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
            lr,
            value_coef,
            entropy_coef,
            seed,
            lam

    ):
        EPSILON = 1e-8

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
        self.opt = optim.Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.sigma_history = []
        self.loss_history = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.lam = lam
        self.EPSILON = EPSILON
        self.observeNorm = ObservationNorm()
        self.advantageNorm = AdvantageNorm()
        self.returnNorm = ReturnNorm()

        self.memory = Memory()

    # Function to choose action based on current policy
    # Returns: action, log probabilitiy, value of the state
    def choose_action(self, observation):
        state = T.as_tensor(observation, dtype=T.float32, device=self.device)
        with T.no_grad():
            dist = self.policy.next_action(state)
            action = dist.sample()
            value = self.critic.evaluated_state(state)
        return int(action.item()), float(value.item())

    # Store reward, state, action in memory
    def remember(self, state, action, reward, done, value, next_state):
        with T.no_grad():
            ns = T.as_tensor(next_state, dtype=T.float32, device=self.device)
            next_value = self.critic.evaluated_state(ns).item()
        self.memory.store(state, action, reward, done, value, next_value)

    def _prepare_batch_data(self):
        """Convert memory to tensors."""
        states = T.as_tensor(np.array(self.memory.states), dtype=T.float32, device=self.device)
        actions = T.as_tensor(self.memory.actions, dtype=T.long, device=self.device)
        rewards = T.as_tensor(self.memory.rewards, dtype=T.float32, device=self.device)
        dones = T.as_tensor(self.memory.dones, dtype=T.float32, device=self.device)
        values = T.as_tensor(self.memory.values, dtype=T.float32, device=self.device)
        return states, actions, rewards, dones, values

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

    def _compute_a2c_loss(self, states, actions, returns, advantages):
        """Compute A2C loss components."""
        dist = self.policy.next_action(states)
        new_logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Simple policy gradient (no clipping)
        policy_loss = -(new_logp * advantages).mean()

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

    def _a2c_update(self, states, actions, returns, adv, use_grad_clip=False):
        """Run single A2C update (no multiple epochs)."""
        total_loss, policy_loss, value_loss = self._compute_a2c_loss(
            states, actions, returns, adv
        )

        self.policy_loss_history.append(policy_loss.item())
        self.value_loss_history.append(value_loss.item())

        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()

        self.opt.step()

        return total_loss.item()

    def vanilla_a2c_update(self):
        if len(self.memory.states) == 0:
            return 0.0

        states, actions, rewards, dones, values = self._prepare_batch_data()
        adv, returns = self._compute_gae(rewards, values, dones)

        # with T.no_grad():
        #     adv = (adv - adv.mean()) / (adv.std(unbiased=False) + self.EPSILON)

        avg_loss = self._a2c_update(states, actions, returns, adv)  # changed
        self.memory.clear()
        return avg_loss


    def update_rbs(self):
        if len(self.memory.states) == 0:
            return 0.0

        states, actions, rewards, dones, values = self._prepare_batch_data()
        adv, returns = self._compute_gae(rewards, values, dones)

        with T.no_grad():
            sigma_t = returns.std(unbiased=False) + 1e-8
            returns = returns / sigma_t
            self.sigma_history.append(sigma_t.item())
            adv = adv / sigma_t

        avg_loss = self._a2c_update(states, actions, returns, adv)  # changed
        self.memory.clear()
        return avg_loss

    def update_adv_norm(self):
        if len(self.memory.states) == 0:
            return 0.0

        states, actions, rewards, dones, values = self._prepare_batch_data()
        adv, returns = self._compute_gae(rewards, values, dones)

        with T.no_grad():
            # --- Advantage normalization ---
            adv = self.advantageNorm.normalize(adv)

        avg_loss = self._a2c_update(states, actions, returns, adv)
        self.memory.clear()
        return avg_loss

    def update_gradient_clipping(self):
        if len(self.memory.states) == 0:
            return 0.0

        states, actions, rewards, dones, values = self._prepare_batch_data()
        adv, returns = self._compute_gae(rewards, values, dones)

        # with T.no_grad():
        #     adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        T.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            0.5
        )

        avg_loss = self._a2c_update(states, actions, returns, adv, use_grad_clip=True)  # changed
        self.memory.clear()
        return avg_loss

    def update_obs_norm(self):
        if len(self.memory.states) == 0:
            return 0.0

        states, actions, rewards, dones, values = self._prepare_batch_data()
        adv, returns = self._compute_gae(rewards, values, dones)

        with T.no_grad():
            # --- observation normalization ---
            states = self.observeNorm.normalize(states)
            # Advantage normalization
            # adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        avg_loss = self._a2c_update(states, actions, returns, adv)
        self.memory.clear()
        return avg_loss



    def update_return_norm(self):
        if len(self.memory.states) == 0:
            return 0.0

        states = T.as_tensor(np.array(self.memory.states), dtype=T.float32, device=self.device)
        actions = T.as_tensor(self.memory.actions, dtype=T.long, device=self.device)
        rewards = T.as_tensor(self.memory.rewards, dtype=T.float32, device=self.device)
        dones = T.as_tensor(self.memory.dones, dtype=T.float32, device=self.device)
        values = T.as_tensor(self.memory.values, dtype=T.float32, device=self.device)

        with T.no_grad():
            next_values = T.cat([values[1:], values[-1:].clone()])
            returns = rewards + self.gamma * next_values * (1 - dones)
            adv = returns - values
            returns = self.returnNorm.normalize(returns)
            # adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        dist = self.policy.next_action(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * adv).mean()

        value_pred = self.critic.evaluated_state(states)
        value_loss = 0.5 * (returns - value_pred).pow(2).mean()

        total_loss = (
                policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
        )

        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.opt.step()

        avg_loss = self._a2c_update(states, actions, returns, adv)
        self.memory.clear()
        return avg_loss

    def update_reward_norm(self):
        if len(self.memory.states) == 0:
            return 0.0

        states = T.as_tensor(np.array(self.memory.states), dtype=T.float32, device=self.device)
        actions = T.as_tensor(self.memory.actions, dtype=T.long, device=self.device)
        rewards = T.as_tensor(self.memory.rewards, dtype=T.float32, device=self.device)
        dones = T.as_tensor(self.memory.dones, dtype=T.float32, device=self.device)
        values = T.as_tensor(self.memory.values, dtype=T.float32, device=self.device)

        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)

        with T.no_grad():
            next_values = T.cat([values[1:], values[-1:].clone()])

            returns = rewards + self.gamma * next_values * (1 - dones)

            adv = returns - values

            # adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        dist = self.policy.next_action(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Actor Loss: - log_prob * Advantage
        policy_loss = -(log_probs * adv).mean()

        value_pred = self.critic.evaluated_state(states)
        value_loss = 0.5 * (returns - value_pred).pow(2).mean()

        total_loss = (
                policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
        )

        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.opt.step()

        avg_loss = self._a2c_update(states, actions, returns, adv)
        self.memory.clear()
        return avg_loss

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
        self.values = []
        self.next_values = []

    def store(self, state, action, reward, done, value, next_value):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        self.next_values.append(float(next_value))

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
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
