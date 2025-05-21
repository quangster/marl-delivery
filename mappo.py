import argparse
import copy
import os

from gymnasium.utils.save_video import save_video
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SequentialSampler

import wandb

from qmix_vdn import Env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'obs_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       's': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
                       'v_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N]),
                       'avail_a_n': np.ones([self.batch_size, self.episode_limit, self.N, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                       'a_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'a_logprob_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'r': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'dw': np.ones([self.batch_size, self.episode_limit, self.N]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r, dw):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r'][self.episode_num][episode_step] = np.array(r).repeat(self.N)
        self.buffer['dw'][self.episode_num][episode_step] = np.array(dw).repeat(self.N)

        self.buffer['active'][self.episode_num][episode_step] = np.ones(self.N)

    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len], dtype=torch.long)
            elif key == 'v_n':
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len + 1], dtype=torch.float32)
            else:
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len], dtype=torch.float32)
        return batch


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        x = self.fc2(self.rnn_hidden)
        x[avail_a_n == 0] = -1e10  # Mask the unavailable actions
        prob = torch.softmax(x, dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, max_episode_len, N, actor_input_dim), prob.shape(mini_batch_size, max_episode_len, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        x = self.fc3(x)
        x[avail_a_n == 0] = -1e10  # Mask the unavailable actions
        prob = torch.softmax(x, dim=-1)
        return prob


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, max_episode_len, N, critic_input_dim), value.shape=(mini_batch_size, max_episode_len, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_agent_specific = args.use_agent_specific
        self.use_value_clip = args.use_value_clip
        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N
        if self.use_agent_specific:
            print("------use agent specific global state------")
            self.critic_input_dim += args.obs_dim

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)
        
        self.actor.to(DEVICE)
        self.critic.to(DEVICE)

    def choose_action(self, obs_n, avail_a_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
            obs_n = obs_n.to(DEVICE)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                """
                    Add an one-hot vector to represent the agent_id
                    For example, if N=3:
                    [obs of agent_1]+[1,0,0]
                    [obs of agent_2]+[0,1,0]
                    [obs of agent_3]+[0,0,1]
                    So, we need to concatenate a N*N unit matrix(torch.eye(N))
                """
                agent_id_one_hot = torch.eye(self.N).to(DEVICE)
                actor_inputs.append(agent_id_one_hot)

            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_input.shape=(N, actor_input_dim)
            actor_inputs = actor_inputs.to(DEVICE)
            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
            avail_a_n.to(DEVICE)
            prob = self.actor(actor_inputs, avail_a_n)  # prob.shape=(N, action_dim)
            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                a_n = prob.argmax(dim=-1)
                return a_n.cpu().numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.cpu().numpy(), a_logprob_n.cpu().numpy()

    def get_value(self, s, obs_n):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            s = s.to(DEVICE)
            critic_inputs.append(s)
            if self.use_agent_specific:  # Add local obs of agents
                critic_inputs.append(torch.tensor(obs_n, dtype=torch.float32).to(DEVICE))
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N).to(DEVICE))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
            critic_inputs = critic_inputs.to(DEVICE)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.cpu().numpy().flatten()

    def train(self, replay_buffer, total_steps, wandb_logger=None):
        batch = replay_buffer.get_training_data()  # Get training data
        for k, v in batch.items():
            batch[k] = v.to(DEVICE)
        max_episode_len = replay_buffer.max_episode_len

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len,N)
            deltas = batch['r'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['dw']) - batch['v_n'][:, :-1]
            for t in reversed(range(max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,max_episode_len,N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,max_episode_len,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv_copy = copy.deepcopy(adv.cpu().numpy())
                adv_copy[batch['active'].cpu().numpy() == 0] = np.nan
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch, max_episode_len)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    Get probs_now and values_now
                    probs_now.shape=(mini_batch_size, max_episode_len, N, action_dim)
                    values_now.shape=(mini_batch_size, max_episode_len, N)
                """
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []
                    for t in range(max_episode_len):
                        # prob.shape=(mini_batch_size*N, action_dim)
                        prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1),
                                          batch['avail_a_n'][index, t].reshape(self.mini_batch_size * self.N, -1))
                        probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))  # prob.shape=(mini_batch_size,N,action_dim）
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))  # v.shape=(mini_batch_size*N,1)
                        values_now.append(v.reshape(self.mini_batch_size, self.N))  # v.shape=(mini_batch_size,N)
                    # Stack them according to the time (dim=1)
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    probs_now = self.actor(actor_inputs[index], batch['avail_a_n'][index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, max_episode_len, N)
                # batch['a_n'][index].shape=(mini_batch_size, max_episode_len, N)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])  # a_logprob_n_now.shape=(mini_batch_size, max_episode_len, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())  # ratios.shape=(mini_batch_size, max_episode_len, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss + critic_loss
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'].to(DEVICE))
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1).to(DEVICE))
        if self.use_agent_specific:
            critic_inputs.append(batch['obs_n'].to(DEVICE))
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len, 1, 1)
            actor_inputs.append(agent_id_one_hot.to(DEVICE))
            critic_inputs.append(agent_id_one_hot.to(DEVICE))

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        actor_inputs = actor_inputs.to(DEVICE)
        critic_inputs = critic_inputs.to(DEVICE)
        return actor_inputs, critic_inputs

    # def save_model(self, env_name, number, seed, total_steps):
    #     torch.save(self.actor.state_dict(), "./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))

    # def load_model(self, env_name, number, seed, step):
    #     self.actor.load_state_dict(torch.load("./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))

    def save_model(self, dir_path="./weights", custom_name=None):
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        name_prefix = custom_name if custom_name else f"{self.algorithm}"
        actor_net_path = os.path.join(dir_path, f"{name_prefix}_actor_net.pth")
        critic_net_path = os.path.join(dir_path, f"{name_prefix}_critic_net.pth")
        try:
            # save actor network
            torch.save(self.actor.state_dict(), actor_net_path)
            # save critic network
            torch.save(self.critic.state_dict(), critic_net_path)

            print("Saved model:")
            print(f"- Actor network: {actor_net_path}")
            print(f"- Critic network: {critic_net_path}")
            return actor_net_path, critic_net_path
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        
    def load_model(self, actor_net_path, critic_net_path):
        try:
            # load Q network
            self.actor.load_state_dict(torch.load(actor_net_path))
            # load Mixer network
            self.critic.load_state_dict(torch.load(critic_net_path))

            print("Model loaded successfully:")
            print(f"- Actor network: {actor_net_path}")
            print(f"- Critic network: {critic_net_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")


class Runner_MAPPO:
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
            n_robots=self.args.n_robots, n_packages=self.args.n_packages,
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
        self.agent_n = MAPPO(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, wandb_logger=None):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy(wandb_logger)  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps = self.run_episode(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy(wandb_logger)

        # Save model
        map_file = os.path.basename(self.args.map_file).split('.')[0]
        file_name = f"{self.args.algorithm}-{map_file}-{self.args.n_robots}r-{self.args.n_packages}p-{self.args.max_time_steps}t"
        actor_net_path, critic_net_path = self.agent_n.save_model(custom_name=file_name)

        if wandb_logger:
            actor_artifact = wandb.Artifact(name=f"{file_name}_actor_net", type="model")
            actor_artifact.add_file(actor_net_path)
            wandb_logger.log_artifact(actor_artifact)

            critic_artifact = wandb.Artifact(name=f"{file_name}_critic_net", type="model")
            critic_artifact.add_file(critic_net_path)
            wandb_logger.log_artifact(critic_artifact)

            video_path = self.evaluate_and_record()
            wandb_logger.log({"video_result": wandb.Video(video_path)})
        self.env.close()

    def evaluate_policy(self, wandb_logger):
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
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n, s = self.env.get_state()  # obs_n.shape=(N,obs_dim) s.shape=(state_dim,)
            avail_a_n = np.ones((self.args.N, self.args.action_dim))
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            v_n = self.agent_n.get_value(s, obs_n)  # Get the state values (V(s)) of N agents
            obs, r, done, _, info = self.env.step(a_n)  # Take a step
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                elif args.use_reward_scaling:
                    r = self.reward_scaling(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r, dw)

            if done:
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n, s = self.env.get_state()
            v_n = self.agent_n.get_value(s, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1
    
    def evaluate_and_record(self, dir_path="./videos/", prefix_name=""):
        episode_reward = 0
        frames = []
        self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n, s = self.env.get_state()  # obs_n.shape=(N,obs_dim), s.shape=(state_dim,)
            avail_a_n = np.ones((self.args.N, self.args.action_dim))
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=True)  # Get actions and the corresponding log probabilities of N agents
            # v_n = self.agent_n.get_value(s, obs_n)  # Get the state values (V(s)) of N agents
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO")

    # Environment parameters
    parser.add_argument("--map_file", type=str, default="map.txt")
    parser.add_argument("--n_robots", type=int, default=2)
    parser.add_argument("--n_packages", type=int, default=5)
    parser.add_argument("--max_time_steps", type=int, default=100)
    parser.add_argument("--model_save_dir", type=str, default="./weights/")
    parser.add_argument("--algorithm", type=str, default="MAPPO", help="MAPPO")

    # Training parameters
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=20000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=int, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    # Algorithm parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")

    # Feature flags
    parser.add_argument("--use_adv_norm",         action="store_true", help="Trick 1: advantage normalization")
    parser.add_argument("--use_reward_norm",      action="store_true", help="Trick 3: reward normalization")
    parser.add_argument("--use_reward_scaling",   action="store_true", help="Trick 4: reward scaling. Here, we do not use it.")
    parser.add_argument("--use_lr_decay",         action="store_true", help="Trick 6: learning rate decay")
    parser.add_argument("--use_grad_clip",        action="store_true", help="Trick 7: gradient clip")
    parser.add_argument("--use_orthogonal_init",  action="store_true", help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps",         action="store_true", help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu",             action="store_true", help="Whether to use ReLU (default False)")
    parser.add_argument("--use_rnn",              action="store_true", help="Whether to use RNN")
    parser.add_argument("--add_agent_id",         action="store_true", help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_agent_specific",   action="store_true", help="Whether to use agent-specific global state.")
    parser.add_argument("--use_value_clip",       action="store_true", help="Whether to use value clip.")

    args = parser.parse_args()
    
    print(vars(args))

    print(f"Using device: {DEVICE}")

    wandb_run = wandb.init(
        project="RL-MAPD",
        config=vars(args),
    )
    runner = Runner_MAPPO(args, number=1, seed=59)
    runner.run(wandb_logger=wandb_run)
    wandb_run.finish()


        