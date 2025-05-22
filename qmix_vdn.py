import argparse
import os

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.utils.save_video import save_video

import wandb
from env import Environment


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_state(state, max_time_steps, max_packages=20, max_packages_in_obs=10):
    """
    Convert the raw environment state into per-agent observations and a global state vector.

    Args:
        state (dict): Contains keys 'robots', 'packages', 'time_step', 'map'.
        max_time_steps (int): Used to normalize the time_step.
        max_packages (int): Max packages to include in global state.
        max_packages_in_obs (int): Max packages per-agent observation.

    Returns:
        observations (np.ndarray): Shape (n_agents, obs_dim), each row is an agent’s obs.
        global_state (np.ndarray): Flattened vector of global features for the mixer.
    """
    # 1) Parse and cast raw state arrays
    robots = np.array(state["robots"]).astype(np.float32)
    packages = np.array(state["packages"]).astype(np.float32)
    time_step = np.array([state["time_step"]]).astype(np.float32)
    grid_map = np.array(state["map"]).astype(np.float32)

    # 2) Build global state features
    n_robots = len(robots)
    n_rows, n_cols = len(grid_map), len(grid_map[0])
    global_state = []
    # 2a) normalized time
    global_state.append(time_step / max_time_steps)
    # 2b) entire map flattened
    global_state.append(grid_map.flatten())
    # 2c) robot positions normalized by grid size
    robots_normalized = robots.copy()
    robots_normalized[:, 0] = robots_normalized[:, 0] / n_rows
    robots_normalized[:, 1] = robots_normalized[:, 1] / n_cols
    global_state.append(robots_normalized.flatten())

    # 2d) package features: normalize coords & times, pad or truncate
    if len(packages) > 0:
        packages_normalized = packages.copy()
        # normalize start/target x,y by max dimension
        packages_normalized[:, 1:5] = packages_normalized[:, 1:5] / max(n_rows, n_cols)
        # normalize start_time, deadline by max_time_steps
        packages_normalized[:, 5:7] = packages_normalized[:, 5:7] / max_time_steps
        if len(packages) > max_packages:
            # choose most urgent packages first
            urgency = packages[:, 6] - time_step  # deadline - current time
            most_urgent_indices = np.argsort(urgency)[:max_packages]
            global_state.append(packages_normalized[most_urgent_indices].flatten())
        else:
            # pad with zeros up to max_packages
            padded_packages = np.zeros((max_packages, 7))
            padded_packages[:len(packages)] = packages_normalized
            global_state.append(padded_packages.flatten())
    else:
        # no packages → all zeros
        global_state.append(np.zeros(max_packages * 7))
    
    # flatten the list into one vector
    global_state = np.concatenate(global_state)

    # 3) Build per‐agent observations
    agent_obs = []
    for i in range(n_robots):
        obs_i = []
        # 3a) own position and carrying flag
        robot_x, robot_y, carrying = robots[i]
        obs_i.append(np.array([robot_x/n_rows, robot_y/n_cols, carrying]).astype(np.float32))
        # 3b) current time
        obs_i.append(time_step / max_time_steps)
        # 3c) full map
        obs_i.append(grid_map.flatten())
        # 3d) other robots: relative positions & carrying
        other_robots = np.zeros((n_robots - 1, 3), dtype=np.float32)
        idx = 0
        for j in range(n_robots):
            if i != j:
                other_x, other_y, other_carrying = robots[j]
                # Relative position and carrying status
                other_robots[idx] = [
                    (other_x - robot_x)/n_rows,
                    (other_y - robot_y)/n_cols,
                    other_carrying
                ]
                idx += 1
        obs_i.append(other_robots.flatten())

        # 3e) local package info: relative coords, times, carrying flag
        package_list = []
        carrying_package = None
        for pkg in packages:
            pkg_id, start_x, start_y, target_x, target_y, start_time, deadline = pkg
            normalized_pkg = [
                pkg_id / max_packages,         
                (start_x - robot_x) / n_rows,  
                (start_y - robot_y) / n_cols,  
                (target_x - robot_x) / n_rows, 
                (target_y - robot_y) / n_cols, 
                start_time / max_time_steps,   
                deadline / max_time_steps,
                1.0 if pkg_id == carrying else 0.0
            ]
            if pkg_id == carrying:
                carrying_package = normalized_pkg
            else:
                package_list.append(normalized_pkg)

        # ensure carried package first, pad/truncate to max_packages_in_obs
        if carrying_package:
            final_packages = [carrying_package] + package_list
        else:
            final_packages = package_list
        while len(final_packages) < max_packages_in_obs:
            final_packages.append([0.0] * 8)
        final_packages = final_packages[:max_packages_in_obs]
        obs_i.append(np.array(final_packages, dtype=np.float32).flatten())

        # concatenate all components of observation
        agent_obs.append(np.concatenate([arr.flatten() for arr in obs_i]))

    # stack into array of shape (n_agents, obs_dim)
    observations = np.stack(agent_obs, axis=0)
    return observations, global_state


def reward_shaping(r, env, state, next_state, action):
    """
    Reward shaping function to guide the agent's learning
    r: original reward (10 for on-time delivery, 1 for late delivery)
    env: environment instance
    state: current state before action
    next_state: state after action
    action: current action as tuple (movement, package) for each robot
    """
    # Get state information
    current_robots = np.array(state["robots"])
    next_robots = np.array(next_state["robots"])
    current_packages = np.array(state["packages"])
    packages = np.array(next_state["packages"])
    current_time_step = state["time_step"]
    next_time_step = next_state["time_step"]
    n_robots = len(current_robots)

    # Reward for picking up packages and moving closer to destination
    for i in range(n_robots):
        current_robot_x, current_robot_y, current_carrying = current_robots[i]
        next_robot_x, next_robot_y, next_carrying = next_robots[i]

        # Action for robot i
        movement_action, package_action = action[i]  # Directly unpack the tuple

        # Check for invalid movement actions
        if movement_action != 0:  # Only check if robot attempted to move
            # If position hasn't changed after a movement action, it was invalid
            if current_robot_x == next_robot_x and current_robot_y == next_robot_y:
                r -= 0.1  # Penalty for invalid movement

        # Check for package pickup
        if package_action == 1 and current_carrying == 0 and next_carrying > 0:  # Robot just picked up a package
            # Find the package that was just picked up
            r += 1.0  # Additional reward for picking up
            break

        # Check for package delivery
        if package_action == 2 and current_carrying > 0 and next_carrying == 0:  # Robot just delivered a package
            # Find the package that was just delivered
            for pkg in packages:
                if pkg[0] == current_carrying:  # package was delivered
                    deadline = pkg[6]
                    delivery_time = next_time_step
                    if delivery_time <= deadline:
                        # On-time delivery: keep full reward
                        r += 10.0
                    else:
                        # Late delivery: reduce reward based on how late
                        time_diff = delivery_time - deadline
                        # Reduce reward by 10% for each time step after deadline
                        # Minimum reward is 1.0
                        reduction = min(0.9, time_diff * 0.1)  # 10% reduction per step, max 90%
                        r += max(1.0, 10.0 * (1 - reduction))
                    break

        # If robot is carrying a package, check if it's moving closer to destination
        if current_carrying > 0 and next_carrying > 0:  # Robot is carrying a package in next state
            # Find the package being carried
            for pkg in packages:
                if pkg[0] == current_carrying:  # pkg[0] is package ID
                    target_x, target_y = pkg[3:5]  # target coordinates
                    # Calculate current and previous Manhattan distances to target
                    next_dist = abs(next_robot_x - target_x) + abs(next_robot_y - target_y)
                    current_dist = abs(current_robot_x - target_x) + abs(current_robot_y - target_y)
                    # If robot moved closer to target, give reward
                    if next_dist < current_dist:
                        r += 0.1
                    break

    return r


class Env(gym.Env):
    def __init__(self, *args, **kwargs):
        super(Env, self).__init__()
        self.env = Environment(*args, **kwargs)
        self.action_space = spaces.multi_discrete.MultiDiscrete([5, 3]*self.env.n_robots)
        self.prev_state = self.env.reset()
        obs, global_state = convert_state(self.prev_state, self.env.max_time_steps)
        self.observation_space = spaces.Box(low=-100, high=100, shape=obs.shape, dtype=np.float32)
        self.state_dim = len(global_state)
        self.obs_dim = obs.shape[1]
        self.action_dim = 15
        from sklearn.preprocessing import LabelEncoder
        self.le1, self.le2= LabelEncoder(), LabelEncoder()
        self.le1.fit(['S', 'L', 'R', 'U', 'D'])
        self.le2.fit(['0','1', '2'])

    def reset(self, *args, **kwargs):
        self.prev_state = self.env.reset()
        return convert_state(self.prev_state, self.env.max_time_steps), {}

    def render(self, *args, **kwargs):
        return self.env.render()
    
    def get_state(self):
        return convert_state(self.env.get_state(), self.env.max_time_steps)
    
    def convert_action(self, actions):
        # actions: list of integers in range (0, 14), one per robot
        n_robots = len(actions)
        result = np.zeros(n_robots * 2, dtype=int)
        for i in range(n_robots):
            action_idx = actions[i]
            # Convert from flat index (0-14) to separate movement and package indices
            # Movement: 0-4 (corresponding to 'S', 'L', 'R', 'U', 'D')
            # Package: 0-2 (corresponding to '0', '1', '2')
            movement_idx = action_idx // 3
            package_idx = action_idx % 3
            # Place in the correct positions in the result array
            result[i*2] = movement_idx    # Movement action (even index)
            result[i*2+1] = package_idx   # Package action (odd index)
        return result

    def step(self, action):
        action = self.convert_action(action)
        ret = []
        ret.append(self.le1.inverse_transform(action.reshape(-1, 2).T[0]))
        ret.append(self.le2.inverse_transform(action.reshape(-1, 2).T[1]))
        action = list(zip(*ret))

        # You should not modify the infos object
        s, r, done, infos = self.env.step(action)
        # new_r = reward_shaping(r, self.env, self.prev_state, action)
        self.prev_state = s
        return convert_state(s, self.env.max_time_steps), r, \
            done, False, infos
    
    def render_image(self):
        cell_size = 100 # size of each cell (pixel)
        height = self.env.n_rows * cell_size
        width = self.env.n_cols * cell_size
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = cell_size / 100
        # Draw grid
        for r in range(self.env.n_rows):
            for c in range(self.env.n_cols):
                x0 = c * cell_size
                y0 = r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                if self.env.grid[r][c] == 1:  # Obstacle
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
                else:  # Free cell
                    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
        # Draw robots
        for i, robot in enumerate(self.env.robots):
            x = robot.position[1] * cell_size + cell_size // 2
            y = robot.position[0] * cell_size + cell_size // 2
            radius = cell_size // 2 - 5
            cv2.circle(img, (x, y), radius, (255, 0, 0), -1)
            cv2.putText(img, f"R{i}", (x - radius // 3, y + radius // 3),
                        font, font_scale, (255, 255, 255), 2)
            # Draw package inside robot if carrying
            if robot.carrying != 0:
                package_size = radius // 2
                package_color = (0, 255, 0)  # Green for packages
                cv2.rectangle(img, 
                            (x - package_size, y - package_size), 
                            (x + package_size, y + package_size), 
                            package_color, -1)
                cv2.putText(img, f"P{robot.carrying}", (x - radius // 3, y + radius // 3), 
                            font, font_scale, (255, 255, 255), 2)
        # Draw packages
        for package in self.env.packages:
            if package.status == "waiting":
                r, c = package.start
                color = (0, 255, 0)
            elif package.status == "in_transit":
                r, c = package.target
                color = (0, 165, 255)
            else:
                continue
            x = c * cell_size + cell_size // 2
            y = r * cell_size + cell_size // 2
            size = cell_size // 3
            cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
            cv2.putText(img, f"P{package.package_id}", (x - size // 2, y + size // 2),
                        font, font_scale, (255, 255, 255), 2)
        return img

class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_limit
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.current_size = 0
        self.buffer = {'obs_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.obs_dim]),
                       's': np.zeros([self.buffer_size, self.episode_limit + 1, self.state_dim]),
                       'avail_a_n': np.ones([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                       'last_onehot_a_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),
                       'a_n': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'r': np.zeros([self.buffer_size, self.episode_limit, 1]),
                       'dw': np.ones([self.buffer_size, self.episode_limit, 1]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.buffer_size, self.episode_limit, 1])
                       }
        self.episode_len = np.zeros(self.buffer_size)

    def store_transition(self, episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step + 1] = last_onehot_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw

        self.buffer['active'][self.episode_num][episode_step] = 1.0

    def store_last_step(self, episode_step, obs_n, s, avail_a_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.episode_len[self.episode_num] = episode_step  # Record the length of this episode
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        # Randomly sampling
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        max_episode_len = int(np.max(self.episode_len[index]))
        batch = {}
        for key in self.buffer.keys():
            if key == 'obs_n' or key == 's' or key == 'avail_a_n' or key == 'last_onehot_a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32)
            elif key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32)

        return batch, max_episode_len


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


class QMIX_Net(nn.Module):
    def __init__(self, args):
        super(QMIX_Net, self).__init__()
        self.N = args.N
        self.state_dim = args.state_dim
        self.batch_size = args.batch_size
        self.qmix_hidden_dim = args.qmix_hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.hyper_layers_num = args.hyper_layers_num
        """
        w1:(N, qmix_hidden_dim)
        b1:(1, qmix_hidden_dim)
        w2:(qmix_hidden_dim, 1)
        b2:(1, 1)
        """
        if self.hyper_layers_num == 2:
            print("hyper_layers_num=2")
            self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim * 1))
        elif self.hyper_layers_num == 1:
            print("hyper_layers_num=1")
            self.hyper_w1 = nn.Linear(self.state_dim, self.N * self.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.state_dim, self.qmix_hidden_dim * 1)
        else:
            print("wrong!!!")

        self.hyper_b1 = nn.Linear(self.state_dim, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1))

    def forward(self, q, s):
        # q.shape(batch_size, max_episode_len, N)
        # s.shape(batch_size, max_episode_len,state_dim)
        q = q.view(-1, 1, self.N)  # (batch_size * max_episode_len, 1, N)
        s = s.reshape(-1, self.state_dim)  # (batch_size * max_episode_len, state_dim)

        w1 = torch.abs(self.hyper_w1(s))  # (batch_size * max_episode_len, N * qmix_hidden_dim)
        b1 = self.hyper_b1(s)  # (batch_size * max_episode_len, qmix_hidden_dim)
        w1 = w1.view(-1, self.N, self.qmix_hidden_dim)  # (batch_size * max_episode_len, N,  qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        # torch.bmm: 3 dimensional tensor multiplication
        q_hidden = F.elu(torch.bmm(q, w1) + b1)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        w2 = torch.abs(self.hyper_w2(s))  # (batch_size * max_episode_len, qmix_hidden_dim * 1)
        b2 = self.hyper_b2(s)  # (batch_size * max_episode_len,1)
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)  # (batch_size * max_episode_len, qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)  # (batch_size * max_episode_len, 1， 1)

        q_total = torch.bmm(q_hidden, w2) + b2  # (batch_size * max_episode_len, 1， 1)
        q_total = q_total.view(self.batch_size, -1, 1)  # (batch_size, max_episode_len, 1)
        return q_total


class VDN_Net(nn.Module):
    def __init__(self, ):
        super(VDN_Net, self).__init__()

    def forward(self, q):
        return torch.sum(q, dim=-1, keepdim=True)  # (batch_size, max_episode_len, 1)


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Q_network_RNN(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size*N,input_dim)
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        Q = self.fc2(self.rnn_hidden)
        return Q


class Q_network_MLP(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_MLP, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size,max_episode_len,N,input_dim)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q
    

class QMIX:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.add_last_action = args.add_last_action
        self.add_agent_id = args.add_agent_id
        self.max_train_steps=args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.use_grad_clip = args.use_grad_clip
        self.batch_size = args.batch_size  # batch_size ~ episode
        self.target_update_freq = args.target_update_freq
        self.tau = args.tau
        self.use_hard_update = args.use_hard_update
        self.use_rnn = args.use_rnn
        self.algorithm = args.algorithm
        self.use_double_q = args.use_double_q
        self.use_RMS = args.use_RMS
        self.use_lr_decay = args.use_lr_decay

        # Compute the input dimension
        self.input_dim = self.obs_dim
        if self.add_last_action:
            print("------add last action------")
            self.input_dim += self.action_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.input_dim += self.N
        
        if self.use_rnn:
            print("------use RNN------")
            self.eval_Q_net = Q_network_RNN(args, self.input_dim)
            self.target_Q_net = Q_network_RNN(args, self.input_dim)
        else:
            print("------use MLP------")
            self.eval_Q_net = Q_network_MLP(args, self.input_dim)
            self.target_Q_net = Q_network_MLP(args, self.input_dim)
        self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())

        if self.algorithm == "QMIX":
            print("------algorithm: QMIX------")
            self.eval_mix_net = QMIX_Net(args)
            self.target_mix_net = QMIX_Net(args)
        elif self.algorithm == "VDN":
            print("------algorithm: VDN------")
            self.eval_mix_net = VDN_Net()
            self.target_mix_net = VDN_Net()
        else:
            print("wrong!!!")

        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_Q_net.parameters())
        if self.use_RMS:
            print("------optimizer: RMSprop------")
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.lr)
        else:
            print("------optimizer: Adam------")
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)

        self.train_step = 0

        self.eval_Q_net.to(DEVICE)
        self.target_Q_net.to(DEVICE)
        self.eval_mix_net.to(DEVICE)
        self.target_mix_net.to(DEVICE)
    
    def choose_action(self, obs_n, last_onehot_a_n, avail_a_n, epsilon):
        with torch.no_grad():
            if np.random.uniform() < epsilon:  # epsilon-greedy
                # Only available actions can be chosen
                a_n = [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]
            else:
                inputs = []
                obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
                obs_n = obs_n.to(DEVICE)
                inputs.append(obs_n)
                if self.add_last_action:
                    last_a_n = torch.tensor(last_onehot_a_n, dtype=torch.float32)
                    last_a_n = last_a_n.to(DEVICE)
                    inputs.append(last_a_n)
                if self.add_agent_id:
                    agent_id_one_hot = torch.eye(self.N).to(device=DEVICE)
                    inputs.append(agent_id_one_hot)

                inputs = torch.cat([x for x in inputs], dim=-1)  # inputs.shape=(N,inputs_dim)
                inputs = inputs.to(DEVICE)
                q_value = self.eval_Q_net(inputs)

                avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
                avail_a_n.to(DEVICE)

                q_value[avail_a_n == 0] = -float('inf')  # Mask the unavailable actions
                a_n = q_value.argmax(dim=-1).cpu().numpy()
            return a_n
    
    def train(self, replay_buffer, total_steps, wandb_logger):
        batch, max_episode_len = replay_buffer.sample()  # Get training data
        for k, v in batch.items():
            batch[k] = v.to(DEVICE)
        self.train_step += 1

        inputs = self.get_inputs(batch, max_episode_len)  # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)
        if self.use_rnn:
            self.eval_Q_net.rnn_hidden = None
            self.target_Q_net.rnn_hidden = None
            q_evals, q_targets = [], []
            for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
                q_eval = self.eval_Q_net(inputs[:, t].reshape(-1, self.input_dim))  # q_eval.shape=(batch_size*N,action_dim)
                q_target = self.target_Q_net(inputs[:, t + 1].reshape(-1, self.input_dim))
                q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))  # q_eval.shape=(batch_size,N,action_dim)
                q_targets.append(q_target.reshape(self.batch_size, self.N, -1))

            # Stack them according to the time (dim=1)
            q_evals = torch.stack(q_evals, dim=1)  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
            q_targets = torch.stack(q_targets, dim=1)
        else:
            q_evals = self.eval_Q_net(inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
            q_targets = self.target_Q_net(inputs[:, 1:])

        with torch.no_grad():
            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.input_dim)).reshape(self.batch_size, 1, self.N, -1)
                q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
                q_evals_next[batch['avail_a_n'][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
                q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len, N)
            else:
                q_targets[batch['avail_a_n'][:, 1:] == 0] = -999999
                q_targets = q_targets.max(dim=-1)[0]  # q_targets.shape=(batch_size, max_episode_len, N)

        # batch['a_n'].shape(batch_size,max_episode_len, N)
        q_evals = torch.gather(q_evals, dim=-1, index=batch['a_n'].unsqueeze(-1)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len, N)

        # Compute q_total using QMIX or VDN, q_total.shape=(batch_size, max_episode_len, 1)
        if self.algorithm == "QMIX":
            q_total_eval = self.eval_mix_net(q_evals, batch['s'][:, :-1])
            q_total_target = self.target_mix_net(q_targets, batch['s'][:, 1:])
        else:
            q_total_eval = self.eval_mix_net(q_evals)
            q_total_target = self.target_mix_net(q_targets)
        # targets.shape=(batch_size,max_episode_len,1)
        targets = batch['r'] + self.gamma * (1 - batch['dw']) * q_total_target

        td_error = (q_total_eval - targets.detach())
        mask_td_error = td_error * batch['active']
        loss = (mask_td_error ** 2).sum() / batch['active'].sum()
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        self.optimizer.step()

        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        else:
            # Softly update the target networks
            for param, target_param in zip(self.eval_Q_net.parameters(), self.target_Q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.eval_mix_net.parameters(), self.target_mix_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

         # Log loss and lr
        if wandb_logger and self.train_step % 100 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb_logger.log({
                "train/loss": loss.item(),
                "train/lr": current_lr,
                "train/train_step": self.train_step,
            })
        
        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        inputs = []
        inputs.append(batch['obs_n'].to(DEVICE))
        if self.add_last_action:
            inputs.append(batch['last_onehot_a_n'].to(DEVICE))
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len + 1, 1, 1)
            inputs.append(agent_id_one_hot.to(DEVICE))

        # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)
        inputs = torch.cat([x for x in inputs], dim=-1)
        inputs = inputs.to(DEVICE)
        return inputs

    def save_model(self, dir_path="./weights", custom_name=None):
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        name_prefix = custom_name if custom_name else f"{self.algorithm}"

        # Add training step information
        step_k = int(self.train_step / 1000)
        name_prefix += f"_step_{step_k}k"

        q_net_path = os.path.join(dir_path, f"{name_prefix}_q_net.pth")
        mixer_net_path = os.path.join(dir_path, f"{name_prefix}_mixer_net.pth")
        try:
            # save Q network
            torch.save(self.eval_Q_net.state_dict(), q_net_path)
            # save Mixer network
            torch.save(self.eval_mix_net.state_dict(), mixer_net_path)

            print("Saved model:")
            print(f"- Q Network: {q_net_path}")
            print(f"- Mixer Network: {mixer_net_path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        return q_net_path, mixer_net_path
    
    def load_model(self, q_net_path, mixer_net_path):
        try:
            # load Q network
            self.eval_Q_net.load_state_dict(torch.load(q_net_path))
            # load Mixer network
            self.eval_mix_net.load_state_dict(torch.load(mixer_net_path))

            # update target networks
            self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
            print("Model loaded successfully:")
            print(f"- Q Network: {q_net_path}")
            print(f"- Mixer: {mixer_net_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")


class Runner_QMIX:
    def __init__(self, args, number, seed):
        self.args = args
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = Env(
            map_file=self.args.map_file, max_time_steps=self.args.max_time_steps, 
            n_robots=self.args.n_robots, n_packages=self.args.n_packages
        )
        self.args.N = self.env.env.n_robots
        self.args.obs_dim = self.env.obs_dim
        self.args.state_dim = self.env.state_dim
        self.args.action_dim = self.env.action_dim
        self.args.episode_limit = self.env.env.max_time_steps

        print(f"map_file={self.args.map_file}")
        print(f"max_time_steps={self.args.max_time_steps}")
        print(f"n_robots={self.args.n_robots}")
        print(f"n_packages={self.args.n_packages}")
        
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = QMIX(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.epsilon = self.args.epsilon 

        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
    
    def run(self, wandb_logger=None):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy(wandb_logger)  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps = self.run_episode(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps, wandb_logger)  # Training

        self.evaluate_policy(wandb_logger)
        # Save model
        map_file = os.path.basename(self.args.map_file).split('.')[0]
        file_name = f"{self.args.algorithm}-{map_file}-{self.args.n_robots}r-{self.args.n_packages}p-{self.args.max_time_steps}t"
        q_net_path, mixer_net_path = self.agent_n.save_model(dir_path=self.args.model_save_dir, custom_name=file_name)

        if wandb_logger:
            q_artifact = wandb.Artifact(name=f"{file_name}_q_net", type="model")
            q_artifact.add_file(q_net_path)
            wandb_logger.log_artifact(q_artifact)

            mixer_artifact = wandb.Artifact(name=f"{file_name}_mix_net", type="model")
            mixer_artifact.add_file(mixer_net_path)
            wandb_logger.log_artifact(mixer_artifact)

            video_path = self.evaluate_and_record()
            wandb_logger.log({"video_result": wandb.Video(video_path)})

        self.env.close()

    def evaluate_policy(self, wandb_logger=None):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode(evaluate=True)
            evaluate_reward += episode_reward
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        if wandb_logger:
            wandb_logger.log({
                "evaluate_reward": evaluate_reward,
                "total_steps": self.total_steps 
            })
    
    def run_episode(self, evaluate=False):
        episode_reward = 0
        self.env.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        for episode_step in range(self.args.episode_limit):
            obs_n, s = self.env.get_state()
            # print(f"obs_n shape: {obs_n.shape}")
            # obs_n.shape = (N, obs_dim), s.shape = (state_dim,)
            epsilon = 0 if evaluate else self.epsilon
            avail_a_n = np.ones((self.args.N, self.args.action_dim))
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            obs, r, done, _, info = self.env.step(a_n)  # Take a step
            episode_reward += r
            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False
                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min
            if done:
                break
        
        if not evaluate:
            obs_n, s = self.env.get_state()
            avail_a_n = np.ones((self.args.N, self.args.action_dim))
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)
        return episode_reward, episode_step + 1

    def evaluate_and_record(self, dir_path="./videos/", prefix_name=""):
        episode_reward = 0
        self.env.reset()
        frames = [self.env.render_image()]
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        for episode_step in range(self.args.episode_limit):
            obs_n, s = self.env.get_state()
            avail_a_n = np.ones((self.args.N, self.args.action_dim))
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, 0)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            obs, r, done, _, info = self.env.step(a_n)  # Take a step
            episode_reward += r
            frames.append(self.env.render_image())

        map_file = os.path.basename(self.args.map_file).split('.')[0]
        file_name = f"{self.args.algorithm}-{map_file}-{self.args.n_robots}r-{self.args.n_packages}p-{self.args.max_time_steps}t"
        file_name = prefix_name + file_name
        fps = self.args.max_time_steps // 100
        save_video(frames, dir_path, fps=fps, name_prefix=file_name)
        full_video_path = dir_path + file_name + "-episode-0.mp4"
        print(f"Saved video: {full_video_path}")
        return full_video_path
    
    def load_weights(self, q_net_path, mixer_net_path):
        self.agent_n.load_model(q_net_path, mixer_net_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX and VDN")

    # Environment parameters
    parser.add_argument("--map_file", type=str, default="map.txt")
    parser.add_argument("--n_robots", type=int, default=2)
    parser.add_argument("--n_packages", type=int, default=5)
    parser.add_argument("--max_time_steps", type=int, default=100)
    parser.add_argument("--model_save_dir", type=str, default="./weights/")
    
    # Training parameters
    parser.add_argument("--max_train_steps", type=int, default=int(1e6) * 5, help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=20000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=int, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    # Algorithm parameters
    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Network architecture
    parser.add_argument("--qmix_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=2, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of MLP")

    # Feature flags
    parser.add_argument("--use_rnn", action="store_true", help="Whether to use RNN")
    parser.add_argument("--use_orthogonal_init", action="store_true", help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", action="store_true", help="Gradient clip")
    parser.add_argument("--use_lr_decay", action="store_true", help="use lr decay")
    parser.add_argument("--use_RMS", action="store_true", help="Whether to use RMS,if False, we will use Adam")
    parser.add_argument("--add_last_action", action="store_true", help="Whether to add last actions into the observation")
    parser.add_argument("--add_agent_id", action="store_true", help="Whether to add agent id into the observation")
    parser.add_argument("--use_double_q", action="store_true", help="Whether to use double q-learning")
    parser.add_argument("--use_reward_norm", action="store_true", help="Whether to use reward normalization")
    parser.add_argument("--use_hard_update", action="store_true", help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--tau", type=float, default=0.005, help="If use soft update")

    # parser.set_defaults(
    #     use_rnn=True,
    #     use_orthogonal_init=True,
    #     use_grad_clip=True,
    #     use_lr_decay=False,
    #     use_RMS=False,
    #     add_last_action=True,
    #     add_agent_id=True,
    #     use_double_q=True,
    #     use_reward_norm=True,
    #     use_hard_update=True
    # )

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    print(vars(args))

    print(f"Using device: {DEVICE}")

    wandb_run = wandb.init(
        project="RL-MAPD",
        config=vars(args),
    )
    runner = Runner_QMIX(args, number=1, seed=59)
    runner.run(wandb_logger=wandb_run)
    wandb_run.finish()
