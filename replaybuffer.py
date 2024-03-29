from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class ReplayBuffer:
    def __init__(self, args, buffer_size):
        self.args = args
        self.buffer_size = buffer_size
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = dict(
            s=torch.zeros(self.buffer_size, self.args.state_dim, dtype=torch.float32),
            a=torch.zeros(self.buffer_size, self.args.action_dim, dtype=torch.float32),
            r=torch.zeros(self.buffer_size, dtype=torch.float32),
            dw=torch.ones(self.buffer_size, dtype=torch.bool))
        self.count = 0
        self.s_last = []
        self.ep_lens = []

    # Expecting all child of s in dictionary

    def store_transition(self, s, a, r, dw):
        self.buffer['s'][self.count] = s
        self.buffer['a'][self.count] = a
        self.buffer['r'][self.count] = r
        self.buffer['dw'][self.count] = dw

        self.count += 1

    def store_last_state(self, s):
        self.s_last.append(s)
        self.ep_lens.append(self.count)

    def merge(self, replay_buffer):
        rem_count = self.buffer_size - self.count

        rem_count = min(rem_count, replay_buffer.count)

        self.buffer['s'][self.count:self.count + rem_count] = replay_buffer.buffer['s'][:rem_count]
        self.s_last.extend(replay_buffer.s_last)
        self.buffer['a'][self.count:self.count + rem_count] = replay_buffer.buffer['a'][:rem_count]
        self.buffer['r'][self.count:self.count + rem_count] = replay_buffer.buffer['r'][:rem_count]
        self.buffer['dw'][self.count:self.count + rem_count] = replay_buffer.buffer['dw'][:rem_count]

        self.count += rem_count

        self.ep_lens.append(rem_count)

    def is_full(self):
        return self.count >= self.buffer_size

    @staticmethod
    def get_adv(v, v_next, r, dw, active, args):
        # Calculate the advantage using GAE
        adv = torch.zeros_like(r, device=r.device)
        gae = 0
        with torch.no_grad():
            deltas = r + args.gamma * v_next * ~dw - v

            for t in reversed(range(r.size(1))):
                gae = deltas[:, t] + args.gamma * args.lamda * gae
                adv[:, t] = gae
            v_target = adv + v
            if args.use_adv_norm:
                mean = adv[active].mean()
                std = adv[active].std() + 1e-8
                adv = (adv - mean) / std
        return adv, v_target

    @staticmethod
    def unfold(x, size, step):
        return x.unfold(dimension=0, size=min(x.size(0), size), step=step).permute(0, -1, *torch.arange(1, x.dim()))

    @staticmethod
    def pad_sequence(x, length):
        return F.pad(x, [0] * (x.dim() - 2) * 2 + [0, length], value=0)

    @staticmethod
    def create_buffer(replay_buffer, args, value_function, device):
        with torch.inference_mode():
            s = replay_buffer.buffer['s']
            s_last = np.vstack(replay_buffer.s_last)
            a = replay_buffer.buffer['a']
            r = replay_buffer.buffer['r']
            dw = replay_buffer.buffer['dw']
            ep_lens = replay_buffer.ep_lens

            # Send to device
            s = s.to(device)
            s_last = torch.tensor(s_last, dtype=torch.float32, device=device)
            a = a.to(device)
            r = r.to(device)
            dw = dw.to(device)
            active = torch.ones_like(r, dtype=torch.bool, device=device)

            # Split buffer into episodes
            s = s.split(ep_lens)
            r = r.split(ep_lens)
            dw = dw.split(ep_lens)
            active = active.split(ep_lens)

            v = []

            # Compute value function
            for i in range(len(ep_lens)):
                # Add last state to the end of the episode
                _s = torch.vstack((s[i], s_last[i].unsqueeze(0)))

                value_function.reset_hidden_state(device=device)

                _v = value_function.forward(_s).squeeze(-1)
                v.append(_v)

            # Pad episode to the same length to compute value function
            v = pad_sequence(v, padding_value=0, batch_first=True)
            r = pad_sequence(r, padding_value=0, batch_first=True)
            dw = pad_sequence(dw, padding_value=1, batch_first=True)
            active = pad_sequence(active, padding_value=0, batch_first=True)

            # Compute v_next (exclude first state, include last state), v (exclude last state, include first state)
            v_next = v[:, 1:]
            v = v[:, :-1]

            # Mask out inactive transitions
            v[~active] = 0
            v_next[~active] = 0

            # Compute advantages
            adv, v_target = ReplayBuffer.get_adv(v, v_next, r, dw, active, args)

            s = torch.cat(s, dim=0)
            adv = adv[active]
            v_target = v_target[active]

            ep_lens = torch.tensor(ep_lens, dtype=torch.long, device=device)
            buffer = dict(s=s, a=a, adv=adv, v_target=v_target, ep_lens=ep_lens)

            return buffer
