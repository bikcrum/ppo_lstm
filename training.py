import argparse
import datetime
import gc
import random

import numpy as np
import tqdm

from ppo import PPO
from utils import *

logging.getLogger().setLevel(logging.DEBUG)


def main(args, env_name):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    max_reward = float('-inf')
    time_now = datetime.datetime.now()

    args.run_name = str(time_now)

    env = gym.make(env_name)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    log = {'state_dim': args.state_dim,
           'action_dim': args.action_dim}

    logging.info(log)

    dev_cpu, dev_gpu = get_device()

    agent = PPO(args, dev_gpu)

    logging.info(f'Using device:{dev_cpu}(CPU), {dev_gpu}(CUDA)')

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)

    run, epochs, total_steps, trajectory_count = init_logger(args, agent)

    pbar = tqdm.tqdm(total=args.max_steps)

    prev_total_steps = 0
    prev_save_steps = 0

    render = False

    wandb.watch(models=agent.actor, log_freq=1)

    actor_global = deepcopy(agent.actor).to(dev_cpu)
    critic_global = deepcopy(agent.critic).to(dev_cpu)

    dispatcher = Dispatcher.remote()

    collectors = [Worker.remote(env_name, dispatcher, actor_global, args, dev_cpu, i) for i in
                  range(args.n_collectors)]

    evaluators = [Worker.remote(env_name, dispatcher, actor_global, args, dev_cpu, i) for i in
                  range(args.n_evaluators)]

    replay_buffer = ReplayBuffer(args, buffer_size=args.buffer_size)

    while total_steps < args.max_steps:
        actor_param_id = ray.put(list(actor_global.parameters()))

        evaluator_ids = []
        if total_steps - prev_total_steps >= args.evaluate_freq:
            """Evaluation"""
            logging.info("Evaluating")
            time_evaluating = datetime.datetime.now()

            # Copy the latest actor to all evaluators
            for evaluator in evaluators:
                evaluator.update_model.remote(actor_param_id)

            # Evaluate policy
            ray.get(dispatcher.set_evaluating.remote(True))
            evaluator_ids = [
                evaluator.evaluate.remote(max_ep_len=min(args.time_horizon, args.eval_buffer_size), render=render) for
                evaluator in evaluators]

            prev_total_steps = total_steps

        """Collect data"""
        logging.info("Collecting")
        time_collecting = datetime.datetime.now()

        # Copy the latest actor to all collectors
        for collector in collectors:
            collector.update_model.remote(actor_param_id)

        # Collect data
        ray.get(dispatcher.set_collecting.remote(True))
        collector_ids = [collector.collect.remote(max_ep_len=min(args.time_horizon, args.buffer_size), render=render)
                         for collector in
                         collectors]

        evaluator_steps = 0
        eval_rewards = []
        eval_lengths = []

        train_rewards = []
        train_lengths = []

        replay_buffer.reset_buffer()

        while evaluator_ids or collector_ids:
            done_ids, remain_ids = ray.wait(collector_ids + evaluator_ids, num_returns=1)

            _replay_buffer, episode_reward, episode_length, worker_id = ray.get(done_ids)[0]

            if _replay_buffer is None:
                # This worker is evaluator

                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)

                evaluator_steps += episode_length

                rem_buffer_size = args.eval_buffer_size - evaluator_steps

                if rem_buffer_size > 0:
                    logging.debug(f"{rem_buffer_size} steps remaining to evaluate")
                    evaluator_ids[worker_id] = evaluators[worker_id].evaluate.remote(
                        max_ep_len=min(args.time_horizon, rem_buffer_size), render=render)
                else:
                    time_evaluating = datetime.datetime.now() - time_evaluating
                    logging.debug('Evaluation done. Cancelling stale evaluators')
                    ray.get(dispatcher.set_evaluating.remote(False))
                    map(ray.cancel, evaluator_ids)
                    evaluator_ids.clear()
            else:
                # This worker is collector

                train_rewards.append(episode_reward)
                train_lengths.append(episode_length)

                replay_buffer.merge(_replay_buffer)

                del _replay_buffer

                if not replay_buffer.is_full():
                    logging.debug(f"{args.buffer_size - replay_buffer.count} steps remaining to collect")
                    collector_ids[worker_id] = collectors[worker_id].collect.remote(
                        max_ep_len=min(args.time_horizon, args.buffer_size - replay_buffer.count),
                        render=render)
                else:
                    time_collecting = datetime.datetime.now() - time_collecting
                    logging.debug('Collector done. Cancelling stale collectors')
                    ray.get(dispatcher.set_collecting.remote(False))
                    map(ray.cancel, collector_ids)
                    collector_ids.clear()

        if evaluator_steps:
            reward, length = np.mean(eval_rewards), np.mean(eval_lengths)

            if reward >= max_reward:
                max_reward = reward
                torch.save(agent.actor.state_dict(), f'saved_models/agent-{run.name}.pth')
                run.save(f'saved_models/agent-{run.name}.pth', policy='now')

            log = {'eval/episode_reward': reward,
                   'eval/episode_length': length,
                   'misc/total_steps': total_steps,
                   'misc/epochs': epochs,
                   'misc/time_evaluating': time_evaluating.total_seconds(),
                   'misc/time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

            logging.info(log)
            run.log(log, step=total_steps)

        mean_train_rewards = np.array(train_rewards).mean()
        mean_train_lens = np.array(train_lengths).mean()

        total_steps += replay_buffer.count
        trajectory_count += len(replay_buffer.ep_lens)

        pbar.update(replay_buffer.count)

        log = {'train/episode_reward': mean_train_rewards,
               'train/episode_length': mean_train_lens,
               'misc/trajectory_count': trajectory_count,
               'misc/total_steps': total_steps,
               'misc/epochs': epochs,
               'misc/time_collecting': time_collecting.total_seconds(),
               'misc/time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

        logging.info(log)
        run.log(log, step=total_steps)

        """Training"""
        logging.info("Training")
        time_training = datetime.datetime.now()

        # Update agent
        actor_loss, entropy_loss, critic_loss, kl, batch_size, train_epoch = agent.update(replay_buffer, total_steps,
                                                                                          check_kl=total_steps >= args.kl_warmup)

        # Copy updated models to global models
        update_model(actor_global, agent.actor.parameters())
        update_model(critic_global, agent.critic.parameters())

        time_training = datetime.datetime.now() - time_training

        log = {'train/actor_loss': actor_loss,
               'train/entropy_loss': entropy_loss,
               'train/critic_loss': critic_loss,
               'train/kl_divergence': kl,
               'misc/total_steps': total_steps,
               'misc/trajectory_count': trajectory_count,
               'misc/batch_size': batch_size,
               'misc/epochs': epochs,
               'misc/train_epoch': train_epoch,
               'misc/time_training': time_training.total_seconds(),
               'misc/time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

        logging.info(log)
        run.log(log, step=total_steps)

        checkpoint = {
            'total_steps': total_steps,
            'epochs': epochs,
            'trajectory_count': trajectory_count,
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
        }

        torch.save(checkpoint, f'checkpoints/checkpoint-{run.name}.pt')

        run.save(f'checkpoints/checkpoint-{run.name}.pt', policy='now')

        if total_steps - prev_save_steps >= args.model_save_freq:
            torch.save(checkpoint, f'checkpoints/checkpoint-{run.name}-{epochs}.pt')

            run.save(f'checkpoints/checkpoint-{run.name}-{epochs}.pt', policy='now')

            prev_save_steps = total_steps

        epochs += 1

        gc.collect()

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters")

    # Training
    parser.add_argument("--model_save_freq", type=int, default=int(1e6), help="Save model frequency")
    parser.add_argument("--max_steps", type=int, default=int(4e9), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1600 * 8 * 2, help="Policy evaluation frequency")
    parser.add_argument("--n_collectors", type=int, default=8, help="Number of collectors")
    parser.add_argument("--n_evaluators", type=int, default=4, help="Number of evaluators")
    parser.add_argument("--buffer_size", type=int, default=1600 * 8, help="Total steps to collect before training")
    parser.add_argument("--eval_buffer_size", type=int, default=1600 * 4, help="Total steps to evaluate the policy")
    parser.add_argument("--mini_batch_size", type=int, default=256,
                        help="Total number of sequences to sample from buffer")
    parser.add_argument("--empty_cuda_cache", type=bool, default=False, help="Whether to empty cuda cache")
    parser.add_argument("--time_horizon", type=int, default=1600, help="The maximum length of the episode")
    parser.add_argument("--device", type=str, default='cuda', help="Device name")

    # Network
    parser.add_argument("--hidden_dim", type=int, default=64, help="Output dimension of CNN and input to transformer")
    parser.add_argument('--lstm_num_layers', type=int, default=1, help='Number of layers in lstm')
    parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='Hidden dimension in lstm')

    # Optimizer
    parser.add_argument('--drop_short_sequence', type=bool, default=False,
                        help='Drop sequences less than transformer window')
    parser.add_argument('--loss_only_at_end', type=bool, default=True,
                        help='Compute loss only at the end of sequence except the first sequence in episode')
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Whether to override dam epsilon")
    parser.add_argument("--eps", type=float, default=1e-5, help="epsilon of Adam optimizer")
    parser.add_argument("--std", type=int, default=0.13, help="Std for action")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization for FF")
    parser.add_argument("--seed", type=int, default=0, help="Seed")

    # PPO
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--num_epoch", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Whether to use advantage normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Whether to use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--grad_clip", type=str, default=0.05, help="Gradient clip value")
    parser.add_argument("--kl_check", action='store_true', help="Whether to check kl divergence")
    parser.add_argument("--kl_threshold", type=float, default=0.2, help="KL threshold of early stopping")
    parser.add_argument("--kl_warmup", type=float, default=1600 * 2,
                        help="Time steps after which kl check is done")

    # Wandb
    parser.add_argument("--project_name", type=str, default='Gym_env', help="Name of project")
    parser.add_argument("--previous_run", type=str, default=None, help="Name of previous run")
    parser.add_argument("--parent_run", type=str, default=None, help="Name of parent run")
    parser.add_argument("--previous_checkpoint", type=str, default=None, help="Timestep of bootstrap checkpoint")
    parser.add_argument("--wandb_mode", type=str, default='online', help="Wandb mode")

    args = parser.parse_args()

    ray.init(num_cpus=min(args.n_collectors, 56))

    main(args, env_name='BipedalWalker-v3')
