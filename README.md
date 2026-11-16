# Flappy Bird AI using Deep Q-Network and Double DQN

An implementation of **Deep Q-Network (DQN)** and **Double Deep Q-Network (Double DQN)** to train an AI agent to play Flappy Bird using Reinforcement Learning and PyTorch.

The goal of this project is to understand how an agent learns through trial and error, estimates action values, and improves its decision-making through experience.

## Overview

In this project, an AI agent learns to play Flappy Bird by interacting with the game environment.

At each time step, the agent:
1. Observes the current game state.
2. Selects an action (flap or do nothing).
3. Receives a reward based on its interaction with the environment.
4. Stores the experience in Replay Memory.
5. Learns to improve its action selection using a neural network.

The agent uses an epsilon-greedy strategy to balance exploration and exploitation.

## Algorithms Implemented

### 1. Deep Q-Network (DQN)

DQN combines Q-Learning with a neural network to approximate the action-value function.

Instead of maintaining a Q-table, a neural network estimates the Q-value for each possible action given the current state.

Two important techniques are used to improve training stability:

- **Experience Replay:** Stores past experiences and randomly samples mini-batches for training, reducing correlations between consecutive experiences.
- **Target Network:** Uses a separate network to calculate target Q-values, helping stabilize the learning process.

The standard DQN target is:

\[
y = r + \gamma(1-d)\max_{a'}Q_{\text{target}}(s',a')
\]

### 2. Double Deep Q-Network (Double DQN)

Standard DQN can suffer from overestimation bias because the same network is used to select and evaluate the next action.

Double DQN addresses this by separating these two operations:

- **Policy Network:** Selects the best next action.
- **Target Network:** Evaluates the selected action.

The Double DQN target is:

\[
y = r + \gamma(1-d)
Q_{\text{target}}\left(s',
\arg\max_{a'}Q_{\text{policy}}(s',a')\right)
\]

This separation helps reduce overestimation of action values and can improve learning stability.

## Key Concepts Explored

- Q-Learning and Bellman Equation
- Deep Q-Networks (DQN)
- Double Deep Q-Networks (Double DQN)
- Experience Replay and Replay Memory
- Target Network Synchronization
- Exploration vs. Exploitation
- Epsilon-Greedy Strategy
- Q-Value Estimation and Overestimation Bias
- Neural Network Optimization and Backpropagation

## Technologies Used

- **Python** – Main programming language
- **PyTorch** – Neural network implementation and optimization
- **Gymnasium** – Reinforcement Learning environment interface
- **Flappy Bird Gymnasium** – Flappy Bird game environment
- **NumPy** – Numerical operations
- **PyYAML** – Hyperparameter configuration

## Project Structure

```text
FlappyBird-DoubleDQN/
│
├── main.py              # Main training and testing script
├── dqn.py               # DQN neural network architecture
├── ReplayMemory.py      # Experience replay implementation
├── parameters.yaml      # Training hyperparameters
│
├── runs/                # Saved models and training logs
│   ├── model.pt
│   └── training.log
│
├── requirements.txt     # Project dependencies
└── README.md
```

## Installation

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd FlappyBird-DoubleDQN
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
```

Activate it:

For macOS/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

If you haven't created a `requirements.txt` yet, install the main packages:

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml numpy
```

## Training the Agent

To train the agent, run:

```bash
python main.py YOUR_HYPERPARAMETER_SET --train
```

Replace `YOUR_HYPERPARAMETER_SET` with a valid configuration name defined in `parameters.yaml`.

For example:

```bash
python main.py default --train
```

During training, the agent explores the environment, collects experiences, and updates its neural network using sampled mini-batches.

The best-performing policy network is saved in the `runs/` directory.

## Testing the Trained Agent

To evaluate the trained agent:

```bash
python main.py default
```

This loads the saved model and runs the agent in the Flappy Bird environment.

The agent uses the learned policy to select actions without exploration.

## Model Saving and Training Logs

The project automatically creates a `runs/` directory to organize training outputs.

- **Model checkpoint (.pt):** Stores the learned neural network parameters.
- **Training log (.log):** Records improvements in the best episode reward.

The model checkpoint can be loaded later without training the agent again.

## Key Learnings

Working on this project helped me understand several Reinforcement Learning concepts beyond their theoretical definitions.

Some of my most important learnings were:

- Understanding how Q-values guide an agent's decision-making.
- Why consecutive experiences are correlated and how Experience Replay helps address this problem.
- Why DQN needs a separate Target Network.
- How overestimation bias occurs and how Double DQN reduces it.
- How epsilon-greedy strategy balances exploration and exploitation.
- How neural networks learn through loss calculation, backpropagation, and gradient-based optimization.
- How hyperparameters affect the agent's learning process and performance.

## Future Improvements

Some possible directions for further experimentation:

- Experiment with different hyperparameters and training strategies.
- Compare standard DQN and Double DQN performance.
- Improve training stability and agent performance.
- Explore additional DQN improvements such as Dueling DQN and Prioritized Experience Replay.

---

**Built as a hands-on learning project to understand and implement Deep Reinforcement Learning.**
