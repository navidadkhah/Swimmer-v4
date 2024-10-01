import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 2024

class PPO_Network(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(PPO_Network, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.policy_mean = nn.Linear(64, num_actions)
        self.policy_log_std = nn.Parameter(torch.zeros(num_actions))
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        shared_out = self.shared_layers(x)
        policy_mean = self.policy_mean(shared_out)
        policy_log_std = self.policy_log_std.expand_as(policy_mean)
        value = self.value_head(shared_out)
        return policy_mean, policy_log_std, value


class PPO_Agent:
    def __init__(self, env, clip_epsilon, learning_rate, discount, entropy_coef, value_coef, ppo_epochs, batch_size):
        self.env = env
        self.clip_epsilon = clip_epsilon
        self.discount = discount
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.memory = []

        self.policy_network = PPO_Network(input_dim=self.env.observation_space.shape[0],
                                          num_actions=self.env.action_space.shape[0]).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        policy_mean, policy_log_std, value = self.policy_network(state)
        policy_std = policy_log_std.exp()
        action_dist = torch.distributions.Normal(policy_mean, policy_std)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), log_prob, value

    def store_transition(self, transition):
        self.memory.append(transition)

    def learn(self):
        states, actions, rewards, dones, log_probs, values = zip(*self.memory)

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=device)
        values = torch.tensor(values, dtype=torch.float32, device=device)

        returns = []
        loss_list = []

        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.discount * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = returns - values.squeeze()

        for _ in range(self.ppo_epochs):
            for idx in range(0, len(states), self.batch_size):
                sampled_idx = np.random.choice(len(states), self.batch_size, replace=False)

                sampled_states = states[sampled_idx]
                sampled_actions = actions[sampled_idx]
                sampled_old_log_probs = old_log_probs[sampled_idx]
                sampled_advantages = advantages[sampled_idx]
                sampled_returns = returns[sampled_idx]

                new_policy_mean, new_policy_log_std, new_value = self.policy_network(sampled_states)
                new_policy_std = new_policy_log_std.exp()
                new_action_dist = torch.distributions.Normal(new_policy_mean, new_policy_std)
                new_log_probs = new_action_dist.log_prob(sampled_actions).sum(dim=-1)

                ratio = (new_log_probs - sampled_old_log_probs).exp()
                surr1 = ratio * sampled_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * sampled_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(sampled_returns, new_value.squeeze())
                entropy_loss = new_action_dist.entropy().mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss_list.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory = []
        return  loss_list


class Model_TrainTest:
    def __init__(self, hyperparams):
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]
        self.clip_epsilon = hyperparams["clip_epsilon"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.ppo_epochs = hyperparams["ppo_epochs"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        self.render = hyperparams["render"]
        self.entropy_coef = hyperparams["entropy_coef"]
        self.value_coef = hyperparams["value_coef"]
        self.env = gym.make("Swimmer-v4", render_mode="human" if self.render else None)
        self.env.metadata['render_fps'] = hyperparams["render_fps"]
        self.agent = PPO_Agent(env=self.env, clip_epsilon=self.clip_epsilon, learning_rate=self.learning_rate,
                               discount=self.discount_factor, entropy_coef=self.entropy_coef,
                               value_coef=self.value_coef, ppo_epochs=self.ppo_epochs, batch_size=self.batch_size)

    def train(self):
        self.reward_history = []
        self.loss_history = []
        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            episode_reward = 0
            while not done and not truncation:
                action, log_prob, value = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)
                self.agent.store_transition((state, action, reward, done, log_prob, value))
                state = next_state
                episode_reward += reward

            loss_list = self.agent.learn()
            self.loss_history.extend(loss_list)
            self.reward_history.append(episode_reward)

            if episode % self.save_interval == 0:
                torch.save(self.agent.policy_network.state_dict(), self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode, loss_list)
                print('\n~~~~~~Interval Save: Model saved.\n')

            result = (f"Episode: {episode}, "
                      f"Raw Reward: {episode_reward:.2f}, ")
            print(result)
        self.plot_training(episode, loss_list)

    def plot_training(self, episode, loss_list):
        sma = np.convolve(self.reward_history, np.ones(50) / 50, mode='valid')

        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        if episode == self.max_episodes:
            plt.savefig('./clip_hyper/reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.title("Loss")
        plt.plot(self.loss_history, label='Loss', color='red', alpha=1)
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()

        if episode == self.max_episodes:
            plt.savefig('./Adaptive/loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

    def test(self, max_episodes):
        total_rewards = []
        self.agent.policy_network.load_state_dict(torch.load(self.RL_load_path))
        for episode in range(max_episodes):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            episode_reward = 0
            while not done and not truncation:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                    policy_mean, policy_log_std, _ = self.agent.policy_network(state_tensor)
                    policy_std = policy_log_std.exp()
                    action_dist = torch.distributions.Normal(policy_mean, policy_std)
                    action = action_dist.mean
                next_state, reward, done, truncation, _ = self.env.step(action.cpu().numpy())
                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{max_episodes}, Reward: {episode_reward:.2f}")

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {max_episodes} episodes: {avg_reward:.2f}")

        return total_rewards, avg_reward

if __name__ == '__main__':
    train_mode = False
    render = not train_mode
    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": f'./clip/final_weights' + '_' + '2500' + '.pth',
        "save_path": f'./clip_hyper/final_weights',
        "save_interval": 500,
        "clip_epsilon": 0.05,
        "learning_rate": 3e-4,
        "discount_factor": 0.99,
        "batch_size": 32,
        "ppo_epochs": 10,
        "max_episodes": 3500 if train_mode else 5,
        "max_steps": 500,
        "render": render,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "render_fps": 10,
    }

    DRL = Model_TrainTest(RL_hyperparams)
    if train_mode:
        DRL.train()
    else:
        DRL.test(RL_hyperparams["max_episodes"])
