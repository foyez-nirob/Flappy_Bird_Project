from collections import deque
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
from dqn import DQN
from experience_replay import ReplayMemory 
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random

# device selection 
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else: 
    device = "cpu"

# create run directory 
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]
        self.mini_batch_size = params["mini_batch_size"]

        self.loss_fn = nn.SmoothL1Loss()   # Huber loss: more stable than MSE
        self.optimizer = None

        # Create log file and Model file 
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")


    def run(self, is_training=True, render=False):

        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
            use_lidar=False,
        )

        num_states = env.observation_space.shape[0]  # input dim = 12
        num_actions = env.action_space.n             # output dim = 2

        policy_dqn = DQN(num_states, num_actions).to(device)
        epsilon = 0.0

        if is_training:
            #initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            #initialize target network
            target_dqn = DQN(num_states, num_actions).to(device)
            # copy the wt & bias vals from policy => target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            total_steps = 0
            recent_rewards = deque(maxlen=100)
            best_mean_reward = float("-inf")

        else:
            # test : load model file 
            policy_dqn.load_state_dict(
                torch.load(self.MODEL_FILE, map_location=device)
            )
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()             # current state
            state = torch.tensor(state, dtype=torch.float, device=device)       #convert state to tensor 

            episode_reward = 0
            terminated = False

            while (not terminated and episode_reward < self.reward_threshold):  # per episode 

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()                          # Explore
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploit

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                
                episode_reward += reward

                # create tensors
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    total_steps += 1
                    
                    # decay epsilon PER STEP
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                    # train every step
                    if len(memory) > self.mini_batch_size:
                        # get sample
                        mini_batch = memory.sample(self.mini_batch_size)
        
                        self.optimize(mini_batch, policy_dqn, target_dqn)
        
                    # sync target network every N steps
                    if total_steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        total_steps = 0

                state = next_state
                if truncated:
                    break
            
            # ---- end of episode ----

            if is_training:
                recent_rewards.append(episode_reward)
                mean_reward = float(np.mean(recent_rewards))

                if (episode + 1) % 100 == 0:
                    msg = (
                        f"episode={episode+1} steps={total_steps} "
                        f"mean100={mean_reward:.2f} last={episode_reward:.2f} "
                        f"epsilon={epsilon:.3f}"
                    )
                    print(msg)
                    with open(self.LOG_FILE, "a") as f:
                        f.write(msg + "\n")

                # save on best ROLLING MEAN (not a single lucky episode)
                if len(recent_rewards) == 100 and mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    msg = (
                        f"** new best mean100={mean_reward:.2f} "
                        f"at episode={episode+1}, steps={total_steps}"
                    )
                    print(msg)
                    with open(self.LOG_FILE, "a") as f:
                        f.write(msg + "\n")
            else:
                print(f"test episode={episode+1} reward={episode_reward:.2f}")

            # env.close() - manually stop

    # standard DQN 
    # def optimize(self, mini_batch, policy_dqn, target_dqn):
    #     # get batch of experiences
    #     states, actions, next_states, rewards, terminations = zip(*mini_batch)

    #     states = torch.stack(states)
    #     actions = torch.stack(actions)
    #     next_states = torch.stack(next_states)
    #     rewards = torch.stack(rewards)
    #     terminations = torch.tensor(terminations).float().to(device)

    #     # calculate target Q-values - if terminations=true => zero
    #     with torch.no_grad():
    #         target_q = rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

            
    #     # calculate y_pred i.e. Q-value from current policy
    #     current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

    #     # compute loss
    #     loss = self.loss_fn(current_q, target_q)

    #     # optimize model
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    # double deep Q - Network 
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # Calculate Double DQN target Q-values
        with torch.no_grad():

            # 1. Policy network selects the best next action
            best_actions = policy_dqn(next_states).argmax(dim=1)

            # 2. Target network evaluates the selected actions
            next_q_values = target_dqn(next_states).gather(
                dim=1,
                index=best_actions.unsqueeze(dim=1)
            ).squeeze(dim=1)

            # 3. Calculate target Q-values
            target_q = rewards + (1 - terminations) * self.gamma * next_q_values

        # Calculate predicted Q-values
        current_q = policy_dqn(states).gather(
            dim=1,
            index=actions.unsqueeze(dim=1)
        ).squeeze(dim=1)

        # Calculate loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize policy network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 10)         # claude suggested this line for
        self.optimizer.step()



if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)