import argparse
import logging
import os.path

import cv2
import gym
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn

import wandb
from network import Actor_LSTM

logging.basicConfig(level=logging.INFO)


def get_action(actor, env, s, deterministic=False):
    def scale_action(y1, y2, x1, x2, x):
        return (x - x1) * (y2 - y1) / (x2 - x1) + y1

    with torch.inference_mode():
        if deterministic:
            a, _ = actor.forward(s)
            # Get output from last observation
        else:
            dist = actor.pdf(s)
            a = dist.sample()
            # a: [1, seq_len, action_dim]

        a = a.squeeze(0)
        # a_logprob: [action_dim]

        a = scale_action(y1=env.action_space.low, y2=env.action_space.high,
                         x1=-1.0, x2=1.0, x=a)

        return a


def evaluate_policy(env_name, run_name, replace=True, best=True, render=True):
    parser = argparse.ArgumentParser("Hyperparameters")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Output dimension of CNN and input to transformer")
    parser.add_argument("--time_horizon", type=int, default=1600, help="The maximum length of the episode")
    parser.add_argument('--lstm_num_layers', type=int, default=1, help='Number of layers in lstm')
    parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='Hidden dimension of lstm')
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization for FF")
    parser.add_argument("--std", type=int, default=0.13, help="Std for action")
    parser.add_argument("--project_name", type=str, default='Gym_env', help="Name of project")

    args = parser.parse_args()

    env = gym.make(env_name, render_mode='human')

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    logging.info({'state_dim': args.state_dim,
                  'action_dim': args.action_dim})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Actor_LSTM(args)
    actor.to(device)

    wandb.login()

    run = wandb.Api().run(os.path.join(args.project_name, run_name.replace(':', '_')))

    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    logging.info(f'Checkpoint loaded from: {run_name}')
    if best:
        if replace or not os.path.exists(f'saved_models/agent-{run_name}.pth'):
            run.file(name=f'saved_models/agent-{run_name}.pth').download(replace=replace)

        with open(f'saved_models/agent-{run_name}.pth', 'rb') as r:
            checkpoint = torch.load(r, map_location=device)

        actor.load_state_dict(checkpoint)
    else:
        if replace or not os.path.exists(f'checkpoints/checkpoint-{run_name}.pt'):
            run.file(name=f'checkpoints/checkpoint-{run_name}.pt').download(replace=replace)

        with open(f'checkpoints/checkpoint-{run_name}.pt', 'rb') as r:
            checkpoint = torch.load(r, map_location=device)

        actor.load_state_dict(checkpoint['actor_state_dict'])

    n_epoch = 50

    for _ in tqdm.tqdm(range(n_epoch)):

        for step in range(args.time_horizon):

            actor.reset_hidden_state(device)

            # s, _ = env.reset(options={"randomize": True})
            s, _ = env.reset()
            done = False

            episode_length = 0
            episode_reward = 0

            with torch.inference_mode():
                while not done:
                    s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

                    a = get_action(actor, env, s, deterministic=True)

                    try:
                        s, r, done, trunc, _ = env.step(a.numpy())

                        if render:
                            env.render()

                    except Exception as e:
                        logging.error(e)
                        break

                    episode_reward += r

            logging.info({'episode_reward': episode_reward, 'episode_length': episode_length})


if __name__ == '__main__':
    with torch.inference_mode():
        evaluate_policy(env_name='BipedalWalker-v3',
                        run_name='2024-02-06 00:43:50.309341',
                        replace=True,
                        best=False,
                        render=True)
