import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Agent:
    def __init__(
            self,
            obs_space,
            action_space,
            hidden,
            gamma,
            clip_coef,
            lr,
            value_coef,
            entropy_coef,
            batch_size,
            ppo_epochs,
            lam,
            seed
    ):
        # EPSILON = 1e-8
        # DEFAULT_BATCH_SIZE = 64
        # DEFAULT_PPO_EPOCHS = 5

        # Initialize seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            T.manual_seed(seed)


        # Use GPU if available
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.action_dim = int(getattr(action_space, "n", action_space))

        # Initialize the policy and the critic networks
        self.policy = Policy(obs_space.shape, self.action_dim, hidden).to(self.device)
        self.critic = Critic(obs_space.shape, hidden).to(self.device)

        # Set optimizer for policy and critic networks
        self.opt = optim.SGD(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            lr=lr,
            momentum=0.9  # Common default for SGD
        )

        self.gamma = gamma
        self.clip = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.sigma_history = []
        self.loss_history = []
        self.policy_loss_history = []
        self.ppo_avg_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        # self.EPSILON = EPSILON
        # self.DEFAULT_BATCH_SIZE = DEFAULT_BATCH_SIZE
        # self.DEFAULT_PPO_EPOCHS = DEFAULT_PPO_EPOCHS
        self.observeNorm = ObservationNorm()
        self.advantageNorm = AdvantageNorm()
        self.returnNorm = ReturnNorm()
        self.running_return_stats = RunningStats()
        self.adv_mean = []
        self.adv_std = []
        self.target_kl = 0.015

        self.memory = Memory()

    # Function to choose action based on current policy
    # Returns: action, log probabilitiy, value of the state
    def choose_action(self, observation):
        state = T.as_tensor(observation, dtype=T.float32, device=self.device)
        with T.no_grad():
            dist = self.policy.next_action(state)
            action = dist.sample()
            logp = dist.log_prob(action)
            value = self.critic.evaluated_state(state)
        return int(action.item()), float(logp.item()), float(value.item())

    # Store reward, state, action in memory
    def remember(self, state, action, reward, done, log_prob, value, next_state):
        with T.no_grad():
            ns = T.as_tensor(next_state, dtype=T.float32, device=self.device)
            next_value = self.critic.evaluated_state(ns).item()
        self.memory.store(state, action, reward, done, log_prob, value, next_value)

    def _prepare_batch_data(self):
        """Convert memory to tensors."""
        states = T.as_tensor(np.array(self.memory.states), dtype=T.float32, device=self.device)
        actions = T.as_tensor(self.memory.actions, dtype=T.long, device=self.device)
        rewards = T.as_tensor(self.memory.rewards, dtype=T.float32, device=self.device)
        dones = T.as_tensor(self.memory.dones, dtype=T.float32, device=self.device)
        old_logp = T.as_tensor(self.memory.log_probs, dtype=T.float32, device=self.device)
        values = T.as_tensor(self.memory.values, dtype=T.float32, device=self.device)
        return states, actions, rewards, dones, old_logp, values

    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        with T.no_grad():
            next_values = T.cat([values[1:], values[-1:].clone()])
            deltas = rewards + self.gamma * next_values * (1 - dones) - values

            adv = T.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
                adv[t] = gae

            returns = adv + values
            return adv, returns

    def _compute_ppo_loss(self, states, actions, old_logp, returns, advantages):
        """Compute PPO loss components."""
        dist = self.policy.next_action(states)
        new_logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        ratio = (new_logp - old_logp).exp()

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = T.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
        policy_loss = -T.min(surr1, surr2).mean()

        # Critic loss
        value_pred = self.critic.evaluated_state(states)
        value_loss = 0.5 * (returns - value_pred).pow(2).mean()

        # Total loss
        total_loss = (
                policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
        )

        return total_loss, policy_loss, value_loss

    def _ppo_update_loop(self, states, actions, old_logp, returns, adv, use_grad_clip=False):
        """Run PPO training loop over multiple epochs and minibatches."""
        total_loss_epoch = 0.0
        num_samples = len(states)
        batch_size = min(self.batch_size, num_samples)

        for epoch in range(self.ppo_epochs):
            # We track KL for the entire epoch to decide on early stopping
            epoch_kls = []

            idxs = T.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                batch_idx = idxs[start:start + batch_size]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logp = old_logp[batch_idx]
                b_returns = returns[batch_idx]
                b_adv = adv[batch_idx]

                # 1. Get new distribution for KL calculation
                dist = self.policy.next_action(b_states)
                new_logp = dist.log_prob(b_actions)

                # 2. Calculate Approximate KL Divergence
                # Formula: mean(exp(log_ratio) - 1 - log_ratio)
                log_ratio = new_logp - b_old_logp
                approx_kl = ((T.exp(log_ratio) - 1) - log_ratio).mean()
                epoch_kls.append(approx_kl.item())

                # 3. Standard Loss calculation and update
                total_loss, policy_loss, value_loss = self._compute_ppo_loss(
                    b_states, b_actions, b_old_logp, b_returns, b_adv
                )

                self.opt.zero_grad(set_to_none=True)
                total_loss.backward()

                if use_grad_clip:
                    T.nn.utils.clip_grad_norm_(
                        list(self.policy.parameters()) + list(self.critic.parameters()),
                        0.5
                    )
                self.opt.step()

                total_loss_epoch += total_loss.item()

            # 4. Early Stopping Check
            # If the average KL for this epoch exceeds our target, we stop
            # avg_kl = np.mean(epoch_kls)
            # if avg_kl > self.target_kl:
            #     print(f"Early stopping at epoch {epoch} due to KL {avg_kl:.4f} > {self.target_kl}")
            #     break

        return total_loss_epoch

    # Basic PPO update function
    def vanilla_ppo_update(self):
        if len(self.memory.states) == 0:
            return 0.0

        states, actions, rewards, dones, old_logp, values = self._prepare_batch_data()
        adv, returns = self._compute_gae(rewards, values, dones)

        # No normalization or scaling applied to adv or returns
        self.adv_mean.append(adv.mean().item())
        self.adv_std.append(adv.std().item())

        avg_total_loss = self._ppo_update_loop(states, actions, old_logp, returns, adv)
        self.ppo_avg_loss_history.append(avg_total_loss)
        self.memory.clear()
        return avg_total_loss

    # Return Based Scaling PPO update function
    def update_rbs(self):
        if len(self.memory.states) == 0:
            return 0.0

        states, actions, rewards, dones, old_logp, values = self._prepare_batch_data()
        adv, returns = self._compute_gae(rewards, values, dones)

        with T.no_grad():
            # Update running stats with the current batch of returns
            self.running_return_stats.update(returns)

            # Scale by the global standard deviation (as per the paper)
            # global_std = self.running_return_stats.std
            returns = returns / returns.std()
            adv = adv / adv.std()

            # self.sigma_history.append(global_std.item())
        self.adv_mean.append(adv.mean().item())
        self.adv_std.append(adv.std().item())

        avg_loss = self._ppo_update_loop(states, actions, old_logp, returns, adv)
        self.memory.clear()
        return avg_loss


    def update_adv_norm(self):
        if len(self.memory.states) == 0:
            return 0.0

        states, actions, rewards, dones, old_logp, values = self._prepare_batch_data()
        adv, returns = self._compute_gae(rewards, values, dones)

        with T.no_grad():
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.adv_mean.append(adv.mean().item())
        self.adv_std.append(adv.std().item())
        avg_loss = self._ppo_update_loop(states, actions, old_logp, returns, adv)
        self.memory.clear()
        return avg_loss


    # Reward Gradient Clipping PPO update function
    def update_gradient_clipping(self):
        if len(self.memory.states) == 0:
            return 0.0

        states, actions, rewards, dones, old_logp, values = self._prepare_batch_data()
        adv, returns = self._compute_gae(rewards, values, dones)

        with T.no_grad():
            # Advantage normalization
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        self.adv_mean.append(adv.mean().item())
        self.adv_std.append(adv.std().item())

        avg_loss = self._ppo_update_loop(states, actions, old_logp, returns, adv, use_grad_clip=True)
        self.memory.clear()
        return avg_loss



    def update_obs_norm(self):
        if len(self.memory.states) == 0:
            return 0.0

            # Convert memory to tensors
        states = T.as_tensor(np.array(self.memory.states), dtype=T.float32, device=self.device)
        actions = T.as_tensor(self.memory.actions, dtype=T.long, device=self.device)
        rewards = T.as_tensor(self.memory.rewards, dtype=T.float32, device=self.device)
        dones = T.as_tensor(self.memory.dones, dtype=T.float32, device=self.device)
        old_logp = T.as_tensor(self.memory.log_probs, dtype=T.float32, device=self.device)
        values = T.as_tensor(self.memory.values, dtype=T.float32, device=self.device)

        with T.no_grad():
            # Compute next values (bootstrap for final step)
            next_values = T.cat([values[1:], values[-1:].clone()])
            deltas = rewards + self.gamma * next_values * (1 - dones) - values

            # --- GAE-Lambda ---
            adv = T.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
                adv[t] = gae

            returns = adv + values

            # --- observation normalization ---
            states = self.observeNorm.normalize(states)
            # Advantage normalization
        # adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # --- PPO Multiple Epochs + Minibatch ---
        total_loss_epoch = 0.0
        num_samples = len(states)
        batch_size = min(32, num_samples)
        ppo_epochs = 5

        for _ in range(ppo_epochs):
            # Shuffle indices
            idxs = T.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                batch_idx = idxs[start:start + batch_size]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logp = old_logp[batch_idx]
                b_returns = returns[batch_idx]
                b_adv = adv[batch_idx]

                dist = self.policy.next_action(b_states)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                ratio = (new_logp - b_old_logp).exp()

                # --- Clipped surrogate objective ---
                surr1 = ratio * b_adv
                surr2 = T.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_adv
                policy_loss = -T.min(surr1, surr2).mean()

                # --- Critic loss ---
                value_pred = self.critic.evaluated_state(b_states)
                value_loss = 0.5 * (b_returns - value_pred).pow(2).mean()

                # --- Total loss ---
                total_loss = (
                        policy_loss +
                        self.value_coef * value_loss -
                        self.entropy_coef * entropy
                )

                # Debug: track individual loss components
                self.policy_loss_history.append(policy_loss.item())
                self.value_loss_history.append(value_loss.item())

                self.opt.zero_grad(set_to_none=True)
                total_loss.backward()
                self.opt.step()
                total_loss_epoch += total_loss.item()

        # Clear memory after full PPO update
        self.memory.clear()

        return total_loss_epoch / (ppo_epochs * (num_samples / batch_size))



    def update_return_norm(self):
        if len(self.memory.states) == 0:
            return 0.0

            # Convert memory to tensors
        states = T.as_tensor(np.array(self.memory.states), dtype=T.float32, device=self.device)
        actions = T.as_tensor(self.memory.actions, dtype=T.long, device=self.device)
        rewards = T.as_tensor(self.memory.rewards, dtype=T.float32, device=self.device)
        dones = T.as_tensor(self.memory.dones, dtype=T.float32, device=self.device)
        old_logp = T.as_tensor(self.memory.log_probs, dtype=T.float32, device=self.device)
        values = T.as_tensor(self.memory.values, dtype=T.float32, device=self.device)

        with T.no_grad():
            # Compute next values (bootstrap for final step)
            next_values = T.cat([values[1:], values[-1:].clone()])
            deltas = rewards + self.gamma * next_values * (1 - dones) - values

            # --- GAE-Lambda ---
            adv = T.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
                adv[t] = gae

            returns = adv + values

            # --- returns normalization ---
            returns = self.returnNorm.normalize(returns)

            # Advantage normalization
            # adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # --- PPO Multiple Epochs + Minibatch ---
        total_loss_epoch = 0.0
        num_samples = len(states)
        batch_size = min(32, num_samples)
        ppo_epochs = 5

        for _ in range(ppo_epochs):
            # Shuffle indices
            idxs = T.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                batch_idx = idxs[start:start + batch_size]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logp = old_logp[batch_idx]
                b_returns = returns[batch_idx]
                b_adv = adv[batch_idx]

                dist = self.policy.next_action(b_states)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                ratio = (new_logp - b_old_logp).exp()

                # --- Clipped surrogate objective ---
                surr1 = ratio * b_adv
                surr2 = T.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_adv
                policy_loss = -T.min(surr1, surr2).mean()

                # --- Critic loss ---
                value_pred = self.critic.evaluated_state(b_states)
                value_loss = 0.5 * (b_returns - value_pred).pow(2).mean()

                # --- Total loss ---
                total_loss = (
                        policy_loss +
                        self.value_coef * value_loss -
                        self.entropy_coef * entropy
                )

                # Debug: track individual loss components
                self.policy_loss_history.append(policy_loss.item())
                self.value_loss_history.append(value_loss.item())

                self.opt.zero_grad(set_to_none=True)
                total_loss.backward()
                self.opt.step()
                total_loss_epoch += total_loss.item()

        # Clear memory after full PPO update
        self.memory.clear()

        return total_loss_epoch / (ppo_epochs * (num_samples / batch_size))

    def update_reward_norm(self):
        if len(self.memory.states) == 0:
            return 0.0

        states = T.as_tensor(np.array(self.memory.states), dtype=T.float32, device=self.device)
        actions = T.as_tensor(self.memory.actions, dtype=T.long, device=self.device)
        rewards = T.as_tensor(self.memory.rewards, dtype=T.float32, device=self.device)
        dones = T.as_tensor(self.memory.dones, dtype=T.float32, device=self.device)
        old_logp = T.as_tensor(self.memory.log_probs, dtype=T.float32, device=self.device)
        values = T.as_tensor(self.memory.values, dtype=T.float32, device=self.device)

        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)

        with T.no_grad():
            next_values = T.cat([values[1:], values[-1:].clone()])
            deltas = rewards + self.gamma * next_values * (1 - dones) - values

            adv = T.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
                adv[t] = gae

            returns = adv + values

        total_loss_epoch = 0.0
        num_samples = len(states)
        batch_size = min(self.batch_size, num_samples)
        ppo_epochs = self.ppo_epochs

        for _ in range(ppo_epochs):
            idxs = T.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                batch_idx = idxs[start:start + batch_size]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logp = old_logp[batch_idx]
                b_returns = returns[batch_idx]
                b_adv = adv[batch_idx]

                dist = self.policy.next_action(b_states)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                ratio = (new_logp - b_old_logp).exp()

                surr1 = ratio * b_adv
                surr2 = T.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_adv
                policy_loss = -T.min(surr1, surr2).mean()

                value_pred = self.critic.evaluated_state(b_states)
                value_loss = 0.5 * (b_returns - value_pred).pow(2).mean()

                total_loss = (
                        policy_loss +
                        self.value_coef * value_loss -
                        self.entropy_coef * entropy
                )

                self.policy_loss_history.append(policy_loss.item())
                self.value_loss_history.append(value_loss.item())

                self.opt.zero_grad(set_to_none=True)
                total_loss.backward()
                self.opt.step()

                total_loss_epoch += total_loss.item()

        self.memory.clear()
        return total_loss_epoch / (ppo_epochs * (num_samples / batch_size))


# Policy network (CNN)
class Policy(nn.Module):
    def __init__(self, obs_shape: tuple, action_dim: int, hidden: int):
        super().__init__()
        c, h, w = obs_shape
        # Suggested architecture for Atari: https://arxiv.org/pdf/1312.5602
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),  # 2592 → 256
            nn.ReLU(),
        )

        # Final output layer: one logit per action
        self.net = nn.Linear(256, action_dim)

    def next_action(self, state: T.Tensor) -> Categorical:
        # state shape should be (B, C, H, W)
        if state.dim() == 3:
            state = state.unsqueeze(0)

        cnn_out = self.cnn(state)  # [B, 256]
        logits = self.net(cnn_out)  # [B, action_dim]
        return Categorical(logits=logits)


# Critic network (CNN)
class Critic(nn.Module):
    def __init__(self, obs_shape: tuple, hidden: int):
        super().__init__()
        c, h, w = obs_shape
        # Suggested architecture for Atari: https://arxiv.org/pdf/1312.5602
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with T.no_grad():
            cnn_output_dim = self.cnn(T.zeros(1, c, h, w)).shape[1]

        self.net = nn.Sequential(
            nn.Linear(cnn_output_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def evaluated_state(self, x: T.Tensor) -> T.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        cnn_out = self.cnn(x)
        return self.net(cnn_out).squeeze(-1)


class Memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.next_values = []

    def store(self, state, action, reward, done, log_prob, value, next_value):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.next_values = []


class ObservationNorm:

    def normalize(self, x):
        return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)  # We add epsilon to make sure that we don't
        # divide through zero.


class AdvantageNorm:
    '''
    This class implements the Advantage Normalization. The purpose is to normalize either across batches or
    only within the same batch.
    '''

    def normalize(self, x):
        return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)  # We add epsilon to make sure that we don't
        # divide through zero.


class ReturnNorm:
    '''
    This class implements the Return Normalization. The purpose is to normalize either across batches or
    only within the same batch.
    '''

    def normalize(self, x):
        return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


class RunningStats:
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = T.mean(x)
        batch_var = T.var(x)
        batch_count = x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self):
        return T.sqrt(self.var + 1e-8)
