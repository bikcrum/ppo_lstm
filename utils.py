import glob
import logging
import os
from copy import deepcopy

import gym
import torch
import wandb
from torch import nn
import ray
from normalization import RewardScaling
from replaybuffer import ReplayBuffer

logging.getLogger().setLevel(logging.DEBUG)


@ray.remote
class Worker:
    def __init__(self, env_name, dispatcher, actor, args, device, worker_id):
        self.env = gym.make(env_name)

        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        self.dispatcher = dispatcher
        if args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
        self.args = args
        self.device = device
        self.actor = deepcopy(actor).to(device)
        self.worker_id = worker_id

    @staticmethod
    def scale_action(y1, y2, x1, x2, x):
        return (x - x1) * (y2 - y1) / (x2 - x1) + y1

    def update_model(self, new_actor_params):
        for p, new_p in zip(self.actor.parameters(), new_actor_params):
            p.data.copy_(new_p)

    def get_action(self, s, deterministic=False):
        with torch.inference_mode():
            if deterministic:
                a, _ = self.actor.forward(s)
                # Get output from last observation
            else:
                dist = self.actor.pdf(s)
                a = dist.sample()
                # a: [1, seq_len, action_dim]

            a = a.squeeze(0)
            # a_logprob: [action_dim]

            a = self.scale_action(y1=self.env.action_space.low, y2=self.env.action_space.high,
                                  x1=-1.0, x2=1.0, x=a)

            return a

    def collect(self, max_ep_len, render=False):
        with torch.inference_mode():
            replay_buffer = ReplayBuffer(self.args, buffer_size=max_ep_len)

            episode_reward = 0

            s, _ = self.env.reset()

            self.actor.reset_hidden_state(self.device)

            if self.args.use_reward_scaling:
                self.reward_scaling.reset()

            for step in range(max_ep_len):
                s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

                a = self.get_action(s, deterministic=False)

                try:
                    s_, r, done, trunc, _ = self.env.step(a.numpy())

                    if render:
                        self.env.render()

                except Exception as e:
                    # Previous state becomes the last state and episode ends
                    if replay_buffer.count > 0:
                        replay_buffer.count -= 1
                        replay_buffer.store_last_state(s)
                    logging.error(e)
                    return replay_buffer, episode_reward, step + 1, self.worker_id

                episode_reward += r

                # Same as done and not trunc
                if done and step != self.args.time_horizon - 1:
                    dw = True
                else:
                    dw = False

                if self.args.use_reward_scaling:
                    r = self.reward_scaling(r)

                r = torch.tensor(r, dtype=torch.float32, device=self.device)
                replay_buffer.store_transition(s, a, r, dw)

                s = s_

                if done:
                    break

                if not ray.get(self.dispatcher.is_collecting.remote()):
                    del replay_buffer
                    return

            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

            replay_buffer.store_last_state(s)

            return replay_buffer, episode_reward, step + 1, self.worker_id

    def evaluate(self, max_ep_len, render=False):
        with torch.inference_mode():
            s, _ = self.env.reset()

            self.actor.reset_hidden_state(self.device)

            episode_reward = 0

            for step in range(max_ep_len):
                s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

                a = self.get_action(s, deterministic=True)

                try:
                    s, r, done, trunc, _ = self.env.step(a.numpy())

                    if render:
                        self.env.render()

                except Exception as e:
                    logging.error(e)
                    break

                episode_reward += r

                if done:
                    break

                if not ray.get(self.dispatcher.is_evaluating.remote()):
                    return

            return None, episode_reward, step + 1, self.worker_id


@ray.remote
class Dispatcher:
    def __init__(self):
        self.collecting = False
        self.evaluating = False

    def is_collecting(self):
        return self.collecting

    def is_evaluating(self):
        return self.evaluating

    def set_collecting(self, val):
        self.collecting = val

    def set_evaluating(self, val):
        self.evaluating = val


def get_device():
    if torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cuda:0")
    else:
        try:
            # For apple silicon
            if torch.backends.mps.is_available():
                # May not require in future pytorch after fix
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
                return torch.device("cpu"), torch.device("mps")
            else:
                return torch.device("cpu"), torch.device("cpu")
        except Exception as e:
            logging.error(e)
            return torch.device("cpu"), torch.device("cpu")


def optimizer_to_device(optimizer, device):
    state_dict = optimizer.state_dict()

    if 'state' not in state_dict:
        logging.warning(f'No state in optimizer. Not converting to {device}')
        return

    states = state_dict['state']

    for k, state in states.items():
        for key, val in state.items():
            states[k][key] = val.to(device)


def update_model(model, new_model_params):
    for p, new_p in zip(model.parameters(), new_model_params):
        p.data.copy_(new_p)


def init_logger(args, agent):
    epochs = 0
    total_steps = 0
    trajectory_count = 0

    # 4 ways to initialize wandb
    # 1. Parent run is not given, previous run is not given -> Start a new run from scratch
    # 2. Parent run is given, previous run is not given -> Create a new run resumed but detached from parent
    # 3. Parent run is not given, previous run is given -> Resume previous run attached to same parent
    # 4. Parent run is given, previous run is given -> Start a new run from previous run attached to same parent

    if args.parent_run is None and args.previous_run is None:
        run = wandb.init(
            project=args.project_name,
            name=args.run_name,
            mode=args.wandb_mode,
            config={**args.__dict__, 'parent_run': args.run_name},
            id=args.run_name.replace(':', '_'),
        )
    elif args.previous_run is None:
        wandb.login()

        run = wandb.Api().run(os.path.join(args.project_name, args.parent_run.replace(':', '_')))

        logging.info(f'Checkpoint loaded from: {args.parent_run}')

        if args.previous_checkpoint:
            checkpoint_name = f'checkpoints/checkpoint-{args.parent_run}-{args.previous_checkpoint}.pt'
        else:
            checkpoint_name = f'checkpoints/checkpoint-{args.parent_run}.pt'

        run.file(name=checkpoint_name).download(replace=True)

        with open(checkpoint_name, 'rb') as r:
            checkpoint = torch.load(r, map_location=agent.device)

        # Create new run
        run = wandb.init(
            project=args.project_name,
            name=args.run_name,
            config={**args.__dict__, 'parent_run': args.parent_run},
            id=args.run_name.replace(':', '_'),
        )

        # Since we start a new run detached from parent, we don't load run state
        total_steps = 0
        trajectory_count = 0
        epochs = 0
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
    elif args.parent_run is None:
        run = wandb.init(
            project=args.project_name,
            resume='allow',
            id=args.previous_run.replace(':', '_'),
        )

        if run.resumed:
            run_ = wandb.Api().run(os.path.join(args.project_name, args.previous_run.replace(':', '_')))

            logging.info(f'Checkpoint loaded from: {args.previous_run}')

            if args.previous_checkpoint:
                checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}-{args.previous_checkpoint}.pt'
            else:
                checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}.pt'

            run_.file(name=checkpoint_name).download(replace=True)

            with open(checkpoint_name, 'rb') as r:
                checkpoint = torch.load(r, map_location=agent.device)

            logging.info(f'Resuming from the run: {run.name} ({run.id})')
            total_steps = checkpoint['total_steps']
            trajectory_count = checkpoint['trajectory_count']
            epochs = checkpoint['epochs']
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        else:
            logging.error(f'Run: {args.previous_run} did not resume')
            raise Exception(f'Run: {args.previous_run} did not resume')
    else:
        wandb.login()

        run = wandb.Api().run(os.path.join(args.project_name, args.previous_run.replace(':', '_')))

        logging.info(f'Checkpoint loaded from: {args.previous_run}')

        if args.previous_checkpoint:
            checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}-{args.previous_checkpoint}.pt'
        else:
            checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}.pt'

        run.file(name=checkpoint_name).download(replace=True)

        with open(checkpoint_name, 'rb') as r:
            checkpoint = torch.load(r, map_location=agent.device)

        # Create new run
        run = wandb.init(
            project=args.project_name,
            name=args.run_name,
            config={**args.__dict__, 'parent_run': args.parent_run},
            id=args.run_name.replace(':', '_'),
        )

        total_steps = checkpoint['total_steps']
        trajectory_count = checkpoint['trajectory_count']
        epochs = checkpoint['epochs']
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

    # Save files to wandb
    cwd = os.getcwd()

    exclude = {'checkpoints', 'saved_models', 'wandb', '.idea', '.git', 'pretrained_models', 'trained_models',
               'offline_data', 'old_files'}

    dirs = [d for d in os.listdir(cwd) if d not in exclude and os.path.isdir(os.path.join(cwd, d))]

    for d in dirs:
        logging.info('Saving files in dir:{} to wandb'.format(d))
        base_paths = [os.path.join(cwd, d, '**', ext) for ext in ['*.py', '*.yaml', '*.yml', '*.sh']]
        for base_path in base_paths:
            for file in glob.glob(base_path, recursive=True):
                file_path = os.path.relpath(file, start=cwd)
                run.save(file_path, policy='now')

    return run, epochs, total_steps, trajectory_count
