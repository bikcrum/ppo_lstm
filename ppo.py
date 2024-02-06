import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.distributions import kl_divergence
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from network import Actor_LSTM, Critic_LSTM
from replaybuffer import ReplayBuffer

logging.basicConfig(level=logging.INFO)


class PPO:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.actor = Actor_LSTM(args)
        self.critic = Critic_LSTM(args)

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        if self.args.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_a, eps=self.args.eps)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c, eps=self.args.eps)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c)

        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()

    def update(self, replay_buffer, total_steps, check_kl):
        losses = []
        kls = []

        window_size = torch.inf

        # Create buffer
        buffer = ReplayBuffer.create_buffer(replay_buffer, self.args, self.critic, device=self.device)

        ep_lens = buffer['ep_lens'].to(torch.int32)

        # Get buffer size
        buffer_size = buffer['s'].size(0)

        # Create an episode lookup tensor
        episode_lookup = torch.arange(ep_lens.size(0), device=self.device).repeat_interleave(ep_lens)

        # Calculate episode start indices
        ep_start_indices = torch.cat((torch.tensor([0], dtype=torch.int32, device=self.device), ep_lens.cumsum(0)[:-1]))

        # Create sampling indices for sequences
        sampling_indices = torch.cat([torch.arange(s, max(s, s + l) + 1, device=self.device) for s, l in
                                      zip(ep_start_indices, ep_lens - window_size)], dim=0)

        # Determine the sequence length
        seq_len = min(ep_lens.max(), window_size)

        # Create a range of indices for selecting sequences
        select_range = torch.arange(seq_len, device=self.device)

        # Get the sequence lengths
        seq_lens = ep_lens[episode_lookup[sampling_indices]].clamp_max(seq_len)

        # Create an active mask for valid sequence indices
        active = select_range < seq_lens.unsqueeze(-1)

        # Create the indices for selecting sequences from the buffer
        select_indices = (sampling_indices.unsqueeze(-1) + select_range).clamp_max(buffer_size - 1)

        # Create a mask for start sequences
        start_sequence = ep_start_indices[episode_lookup[sampling_indices]] == sampling_indices

        if self.args.loss_only_at_end:
            # Set active mask to False for non-start sequences
            active[~start_sequence] = False
            # Set active mask to True for last time step of each sequence
            active[~start_sequence, seq_lens[~start_sequence] - 1] = True

        # Drop sequences that are short
        if self.args.drop_short_sequence:
            valid_mask = seq_lens >= window_size
            active = active[valid_mask]
            select_indices = select_indices[valid_mask]

        batch_size = active.size(0)

        self.actor_old.load_state_dict(self.actor.state_dict())

        for i in range(self.args.num_epoch):
            early_stop = False
            sampler = tqdm.tqdm(BatchSampler(SubsetRandomSampler(range(batch_size)), self.args.mini_batch_size, False))
            for index in sampler:
                # Reset hidden state
                self.actor.reset_hidden_state(device=self.device, batch_size=len(index))
                self.critic.reset_hidden_state(device=self.device, batch_size=len(index))
                self.actor_old.reset_hidden_state(device=self.device, batch_size=len(index))

                # Get active mask for selected indices
                _active = active[index].to(self.device)

                # Get transitions for selected indices
                s = buffer['s'][select_indices[index]].to(self.device)
                # s: [batch_size, seq_len, args.state_dim]

                a = buffer['a'][select_indices[index]].to(self.device)
                # a: [batch_size, seq_len, args.action_dim]

                with torch.inference_mode():
                    dist = self.actor_old.pdf(s)
                    a_logprob = dist.log_prob(a)

                # a_logprob = buffer['a_logprob'][select_indices[index]].to(self.device)
                # a_logprob: [batch_size, seq_len]

                adv = buffer['adv'][select_indices[index]].to(self.device)
                # adv: [batch_size, seq_len]

                v_target = buffer['v_target'][select_indices[index]].to(self.device)
                # v_target: [batch_size, seq_len]

                # Forward pass
                dist_now = self.actor.pdf(s)
                values_now = self.critic(s).squeeze(-1)

                ratios = (dist_now.log_prob(a).sum(-1) - a_logprob.sum(-1)).exp()
                # ratios = torch.exp(dist_now.log_prob(a)[_active].sum(-1) - a_logprob[_active].sum(-1))

                del a_logprob

                # actor loss
                # adv = adv[_active]
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv
                entropy_loss = - self.args.entropy_coef * dist_now.entropy().sum(-1)
                actor_loss = -torch.min(surr1, surr2)

                actor_loss = actor_loss[_active].mean()
                entropy_loss = entropy_loss[_active].mean()
                critic_loss = 0.5 * F.mse_loss(values_now, v_target, reduction='none')[_active].mean()

                with torch.inference_mode():
                    kl = kl_divergence(dist_now, dist).sum(-1)[_active].mean().item()
                    kls.append(kl)

                if self.args.kl_check and check_kl and kl > self.args.kl_threshold:
                    logging.warning(f'Early stopping at epoch {i} due to reaching max kl.')
                    early_stop = True
                    break

                log = {'epochs': i, 'actor_loss': actor_loss.item(), 'entropy_loss': entropy_loss.item(),
                       'critic_loss': critic_loss.item(), 'batch_size': batch_size, 'kl_divergence': kl,
                       'active_count': len(adv), 'active_shape': _active.shape}

                sampler.set_description(str(log))

                # Check for error
                if not (torch.isfinite(s).all()
                        and torch.isfinite(a).all()
                        and torch.isfinite(adv).all()
                        and torch.isfinite(v_target).all()):
                    torch.save({'s': s, 'a': a, 'adv': adv, 'v_target': v_target, 'values_now': values_now,
                                'actor_old_state_dict': self.actor_old.state_dict(),
                                'actor_state_dict': self.actor.state_dict(),
                                'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
                                'surr1': surr1, 'surr2': surr2, 'entropy_loss': entropy_loss, 'actor_loss': actor_loss,
                                'critic_loss': critic_loss, 'kl': kl, '_active': _active},
                               f'training_logs/training_error_{self.args.run_name}.pt')
                    raise RuntimeError(
                        f"Non-finite values detected in training. Saved to training_logs/training_error_{self.args.run_name}.pt'")

                losses.append((log['actor_loss'], log['entropy_loss'], log['critic_loss']))

                # Update
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()

                (actor_loss + entropy_loss).backward()
                critic_loss.backward()

                if self.args.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_clip)

                if any([((~param.grad.isfinite()).any()).item() for param in self.actor.parameters() if
                        param.grad is not None]):

                    # collect gradients in array
                    gradients = []
                    for param in self.actor.parameters():
                        if param.grad is None:
                            continue
                        gradients.append(param.grad)

                    torch.save({'s': s, 'a': a, 'adv': adv, 'v_target': v_target, 'values_now': values_now,
                                'actor_old_state_dict': self.actor_old.state_dict(),
                                'actor_state_dict': self.actor.state_dict(),
                                'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
                                'gradients': gradients,
                                'surr1': surr1, 'surr2': surr2, 'entropy_loss': entropy_loss, 'actor_loss': actor_loss,
                                'critic_loss': critic_loss, 'kl': kl, '_active': _active},
                               f'training_logs/training_error_{self.args.run_name}.pt')
                    # raise RuntimeError(
                    #     f"Non-finite values detected in gradients. Saved to training_logs/training_error_{self.args.run_name}.pt'")
                    logging.warning(
                        f"Non-finite values detected in gradients. Saved to training_logs/training_error_{self.args.run_name}.pt'")
                    early_stop = True
                    break

                self.optimizer_actor.step()
                self.optimizer_critic.step()

                del adv, actor_loss, entropy_loss, critic_loss, values_now, v_target, _active, dist_now

                if self.args.empty_cuda_cache and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            if early_stop:
                break

        if self.args.use_lr_decay:
            self.lr_decay(total_steps)

        a_loss, e_loss, c_loss = zip(*losses)
        kl = np.mean(kls)

        del losses, kls, buffer, episode_lookup, ep_start_indices, ep_lens, sampling_indices, \
            select_range, seq_lens, active, select_indices, start_sequence

        if self.args.empty_cuda_cache and self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return np.mean(a_loss), np.mean(e_loss), np.mean(c_loss), kl, batch_size, i

    def lr_decay(self, total_steps):
        lr_a_now = self.args.lr_a * (1 - total_steps / self.args.max_steps)
        lr_c_now = self.args.lr_c * (1 - total_steps / self.args.max_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
