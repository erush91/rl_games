import copy
import time
import datetime
import os

import gym
import numpy as np
import torch

from rl_games.algos_torch import model_builder
from rl_games.common import env_configurations, vecenv

from pathlib import Path

import pandas as pd
import dask as dd

import sklearn.preprocessing

import matplotlib
matplotlib.use('TkAgg')  # Replace 'TkAgg' with another backend if needed

from collections import OrderedDict

class BasePlayer(object):

    def __init__(self, params):
        self.config = config = params['config']
        self.load_networks(params)
        self.env_name = self.config['env_name']
        self.player_config = self.config.get('player', {})
        self.env_config = self.config.get('env_config', {})
        self.env_config = self.player_config.get('env_config', self.env_config)
        self.env_info = self.config.get('env_info')
        self.clip_actions = config.get('clip_actions', True)
        self.seed = self.env_config.pop('seed', None)
        if self.env_info is None:
            use_vecenv = self.player_config.get('use_vecenv', False)
            if use_vecenv:
                print('[BasePlayer] Creating vecenv: ', self.env_name)
                self.env = vecenv.create_vec_env(
                    self.env_name, self.config['num_actors'], **self.env_config)
                self.env_info = self.env.get_env_info()
            else:
                print('[BasePlayer] Creating regular env: ', self.env_name)
                self.env = self.create_env()
                self.env_info = env_configurations.get_env_info(self.env)
        else:
            self.env = config.get('vec_env')

        self.num_agents = self.env_info.get('agents', 1)
        self.value_size = self.env_info.get('value_size', 1)
        self.action_space = self.env_info['action_space']

        self.observation_space = self.env_info['observation_space']
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.layers_out = None
        self.player_config = self.config.get('player', {})
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.has_central_value = self.config.get(
            'central_value_config') is not None
        self.device_name = self.config.get('device_name', 'cuda')
        self.render_env = self.player_config.get('render', False)
        self.games_num = self.player_config.get('games_num', 1)
        if 'deterministic' in self.player_config:
            self.is_deterministic = self.player_config['deterministic']
        else:
            self.is_deterministic = self.player_config.get(
                'deterministic', True)
        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.export_data = self.player_config.get('export_data', True)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        if self.env.cfg['name'] == 'AnymalTerrain' or self.env.cfg['name'] == 'A1Terrain':
            self.max_steps = 501 # 3001 # 1501 # 10001 # 1001 # 108000 // 4
        if self.env.cfg['name'] == 'ShadowHand':
            self.max_steps = 501

        self.device = torch.device(self.device_name)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if 'obs' in obs:
                obs = obs['obs']
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert (obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(
                weights['running_mean_std'])

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_deterministic=False, neural_obs_override=None, neural_state_override=None):
        raise NotImplementedError('step')

    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size(
            )[2]), dtype=torch.float32, requires_grad=True).to(self.device) for s in rnn_states]
            print('hi')
            # self.states[0].requires_grad = True
            # self.states[1].requires_grad = True
            # self.states[2].requires_grad = True
            # self.states[3].requires_grad = True

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        DIM_ACT = self.action_space.shape[-1]
        DIM_OBS = self.observation_space.shape[-1] - DIM_ACT

        DIM_A_MLP_XX = 0
        DIM_C_MLP_XX = 0

        DIM_A_LSTM_HC = 0
        DIM_C_LSTM_HC = 0

        DIM_A_LSTM_HX = 0
        DIM_C_LSTM_HX = 0

        DIM_A_LSTM_CX = 0
        DIM_C_LSTM_CX = 0

        DIM_A_GRU_HX = 0
        DIM_C_GRU_HX = 0


        DIM_A_MLP_XX = self.config['network'].network_builder.params['mlp']['units'][-1]
        DIM_C_MLP_XX = self.config['network'].network_builder.params['mlp']['units'][-1]

        if self.model.get_default_rnn_state() == None:
            rnn_type = None
            print("rnn model not supported")
        elif len(self.model.get_default_rnn_state()) == 4:
            rnn_type = 'lstm'

            DIM_A_LSTM_HX = self.model.get_default_rnn_state()[0].size(dim=2) # actor lstm hn  (short-term memory)
            DIM_A_LSTM_CX = self.model.get_default_rnn_state()[1].size(dim=2) # actor lstm cn  (long-term memory)
            DIM_A_LSTM_HC = DIM_A_LSTM_HX + DIM_A_LSTM_CX

            DIM_C_LSTM_HX = self.model.get_default_rnn_state()[2].size(dim=2) # self.model.get_default_rnn_state()[2].size(dim=2) # critic lstm hn (short-term memory)
            DIM_C_LSTM_CX = self.model.get_default_rnn_state()[3].size(dim=2) # self.model.get_default_rnn_state()[3].size(dim=2) # critic lstm cn (short-term memory)
            DIM_C_LSTM_HC = DIM_C_LSTM_HX + DIM_C_LSTM_CX
            
        elif len(self.model.get_default_rnn_state()) == 2:
            rnn_type = 'gru'
            DIM_A_GRU_HX = self.model.get_default_rnn_state()[0].size(dim=2) # gru hn
            DIM_C_GRU_HX = self.model.get_default_rnn_state()[1].size(dim=2) # gru hn
            
        tensor_specs = OrderedDict([
            ('ENV', 1),
            ('TIME', 1),
            ('DONE', 1),
            ('REWARD', 1),
            ('ACT', DIM_ACT),
            ('OBS', DIM_OBS),
            ('A_MLP_XX', DIM_A_MLP_XX),
            ('A_LSTM_HC', DIM_A_LSTM_HC),
            ('A_LSTM_CX', DIM_A_LSTM_CX),
            ('A_LSTM_C1X', DIM_A_LSTM_CX),
            ('A_LSTM_C2X', DIM_A_LSTM_CX),
            ('A_LSTM_HX', DIM_A_LSTM_HX),
            ('C_MLP_XX', DIM_C_MLP_XX),
            ('C_LSTM_HC', DIM_C_LSTM_HC),
            ('C_LSTM_CX', DIM_C_LSTM_CX),
            ('C_LSTM_C1X', DIM_C_LSTM_CX),
            ('C_LSTM_C2X', DIM_C_LSTM_CX),
            ('C_LSTM_HX', DIM_C_LSTM_HX),
            ('A_GRU_HX', DIM_A_GRU_HX),
            ('C_GRU_HX', DIM_C_GRU_HX),
        ])

        if self.env.cfg['name'] == 'AnymalTerrain' or self.env.cfg['name'] == 'A1Terrain':
            new_tensor_specs = OrderedDict()
            for key, value in tensor_specs.items():
                if key == 'ACT':  # Add 'FT_FORCE' after 'ACT'
                    new_tensor_specs['FT_FORCE'] = 4
                    new_tensor_specs['PERTURB_BEGIN'] = 1
                    new_tensor_specs['PERTURB'] = 1
                    new_tensor_specs['STANCE_BEGIN'] = 1
                new_tensor_specs[key] = value
            tensor_specs = new_tensor_specs

        N_STEPS = self.max_steps - 1
        N_ENVS = self.env.num_environments
        
        def create_tensor_dict(tensor_specs):
            tensor_dict = OrderedDict()
            for key, dim in tensor_specs.items():
                if key == 'TIME':

                    # Create an array with a sequence of numbers multiplied by step size repeated over rows
                    arr = np.tile(np.arange(N_STEPS) * self.env.dt, (N_ENVS, 1)).T
                    arr3d = np.expand_dims(arr, axis=2)

                    # convert to torch tensor
                    tensor_data = torch.from_numpy(arr3d)

                    # assign to your dictionary
                    tensor_dict[key] = {'data': tensor_data, 'cols': [key]}
                    
                elif key == 'ENV':
                    # Create an array with a sequence of numbers repeated over rows
                    arr = np.tile(np.arange(N_ENVS), (N_STEPS, 1))
                    arr3d = np.expand_dims(arr, axis=2)

                    # convert to torch tensor
                    tensor_data = torch.from_numpy(arr3d)

                    # assign to your dictionary
                    tensor_dict[key] = {'data': tensor_data, 'cols': [key]}
                elif dim == 1:
                    tensor_dict[key] = {'data': torch.zeros((N_STEPS, N_ENVS, dim)), 'cols': [key]}
                else:
                    tensor_dict[key] = {'data': torch.zeros((N_STEPS, N_ENVS, dim)), 'cols': np.char.mod('%s_RAW_%%03d' % key, np.arange(dim))}
            return tensor_dict

        tensor_dict = create_tensor_dict(tensor_specs)

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        if self.config['name'] == "AnymalTerrain" or self.config['name'] == "A1Terrain":
            tensor_dict['ACT']['cols'] = pd.read_csv('/home/gene/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/legend/act_anymalterrain.csv', header=None).values[:,0]
            tensor_dict['OBS']['cols'] = pd.read_csv('/home/gene/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/legend/obs_anymalterrain.csv', header=None).values[:,0]

        if self.config['name'] == "ShadowHandAsymmLSTM":
            tensor_dict['ACT']['cols'] = pd.read_csv('/home/gene/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/legend/act_shadowhand.csv', header=None).values[:,0]
            tensor_dict['OBS']['cols'] = pd.read_csv('/home/gene/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/legend/obs_shadowhand.csv', header=None).values[:,0]

        if rnn_type == 'lstm':
            # actor lstm weights and biases
            a_w_ih = self.model.a2c_network.a_rnn.rnn.weight_ih_l0.detach()
            # pd.DataFrame(w_ih.cpu()).to_csv('a_w_ih.csv')
            a_w_ii = a_w_ih[0*DIM_A_LSTM_HX:1*DIM_A_LSTM_HX,:]
            a_w_if = a_w_ih[1*DIM_A_LSTM_HX:2*DIM_A_LSTM_HX,:]
            a_w_ig = a_w_ih[2*DIM_A_LSTM_HX:3*DIM_A_LSTM_HX,:]
            a_w_io = a_w_ih[3*DIM_A_LSTM_HX:4*DIM_A_LSTM_HX,:]
            
            a_w_hh = self.model.a2c_network.a_rnn.rnn.weight_hh_l0.detach()
            # pd.DataFrame(w_hh.cpu()).to_csv('a_w_hh.csv')
            a_w_hi = a_w_hh[0*DIM_A_LSTM_CX:1*DIM_A_LSTM_CX,:]
            a_w_hf = a_w_hh[1*DIM_A_LSTM_CX:2*DIM_A_LSTM_CX,:]
            a_w_hg = a_w_hh[2*DIM_A_LSTM_CX:3*DIM_A_LSTM_CX,:]
            a_w_ho = a_w_hh[3*DIM_A_LSTM_CX:4*DIM_A_LSTM_CX,:]

            a_b_ih = self.model.a2c_network.a_rnn.rnn.bias_ih_l0.detach()
            # pd.DataFrame(b_ih.cpu()).to_csv('a_b_ih.csv')
            a_b_ii = a_b_ih[0*DIM_A_LSTM_HX:1*DIM_A_LSTM_HX]
            a_b_if = a_b_ih[1*DIM_A_LSTM_HX:2*DIM_A_LSTM_HX]
            a_b_ig = a_b_ih[2*DIM_A_LSTM_HX:3*DIM_A_LSTM_HX]
            a_b_io = a_b_ih[3*DIM_A_LSTM_HX:4*DIM_A_LSTM_HX]

            a_b_hh = self.model.a2c_network.a_rnn.rnn.bias_hh_l0.detach()
            # pd.DataFrame(b_hh.cpu()).to_csv('a_b_hh.csv')
            a_b_hi = a_b_hh[0*DIM_A_LSTM_CX:1*DIM_A_LSTM_CX]
            a_b_hf = a_b_hh[1*DIM_A_LSTM_CX:2*DIM_A_LSTM_CX]
            a_b_hg = a_b_hh[2*DIM_A_LSTM_CX:3*DIM_A_LSTM_CX]
            a_b_ho = a_b_hh[3*DIM_A_LSTM_CX:4*DIM_A_LSTM_CX]
        
            # critic lstm weights and biases
            c_w_ih = self.model.a2c_network.c_rnn.rnn.weight_ih_l0.detach()
            # pd.DataFrame(w_ih.cpu()).to_csv('c_w_ih.csv')
            c_w_ii = c_w_ih[0*DIM_C_LSTM_HX:1*DIM_C_LSTM_HX]
            c_w_if = c_w_ih[1*DIM_C_LSTM_HX:2*DIM_C_LSTM_HX]
            c_w_ig = c_w_ih[2*DIM_C_LSTM_HX:3*DIM_C_LSTM_HX]
            c_w_io = c_w_ih[3*DIM_C_LSTM_HX:4*DIM_C_LSTM_HX]
            
            c_w_hh = self.model.a2c_network.c_rnn.rnn.weight_hh_l0.detach()
            # pd.DataFrame(w_hh.cpu()).to_csv('c_w_hh.csv')
            c_w_hi = c_w_hh[0*DIM_C_LSTM_CX:1*DIM_C_LSTM_CX]
            c_w_hf = c_w_hh[1*DIM_C_LSTM_CX:2*DIM_C_LSTM_CX]
            c_w_hg = c_w_hh[2*DIM_C_LSTM_CX:3*DIM_C_LSTM_CX]
            c_w_ho = c_w_hh[3*DIM_C_LSTM_CX:4*DIM_C_LSTM_CX]

            c_b_ih = self.model.a2c_network.c_rnn.rnn.bias_ih_l0.detach()
            # pd.DataFrame(b_ih.cpu()).to_csv('c_b_ih.csv')
            c_b_ii = c_b_ih[0*DIM_C_LSTM_HX:1*DIM_C_LSTM_HX]
            c_b_if = c_b_ih[1*DIM_C_LSTM_HX:2*DIM_C_LSTM_HX]
            c_b_ig = c_b_ih[2*DIM_C_LSTM_HX:3*DIM_C_LSTM_HX]
            c_b_io = c_b_ih[3*DIM_C_LSTM_HX:4*DIM_C_LSTM_HX]

            c_b_hh = self.model.a2c_network.c_rnn.rnn.bias_hh_l0.detach()
            # pd.DataFrame(b_hh.cpu()).to_csv('c_b_hh.csv')
            c_b_hi = c_b_hh[0*DIM_C_LSTM_CX:1*DIM_C_LSTM_CX]
            c_b_hf = c_b_hh[1*DIM_C_LSTM_CX:2*DIM_C_LSTM_CX]
            c_b_hg = c_b_hh[2*DIM_C_LSTM_CX:3*DIM_C_LSTM_CX]
            c_b_ho = c_b_hh[3*DIM_C_LSTM_CX:4*DIM_C_LSTM_CX]

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False



            cx_pc_last = None

            import pickle as pk
            # AnymalTerrain (3) (perturb longer w/ noise) (with HC = (HC, CX))
            DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_09-02-37_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'
            # DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-02_10-25-10_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'
            
            # AnymalTerrain (1) (no bias) no bias but pos u and neg u, no noise/perturb (with HC = (HC, CX))
            # DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_22-30-47_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'


            # AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM-DIST) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=horizon_length=16) (w/ bias)
            DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-04_15-17-09_u[0.3,1.0,16]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'

            # AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM16-DIST500) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=horizon_length=16) (w/o bias)
            DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-05_10-56-19_u[0.3,1.0,16]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/' # w/o noise
            # DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-05_11-01-54_u[0.3,1.0,16]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/' # w/ noise

            # AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM4-DIST500) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=4, horizon_length=16) (w/o bias)
            # DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_20-23-04_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]_LSTM4-DIST500-noperturb/' 

            # post_corl_fix


            # **LSTM16-DIST500 4/4 steps, W/ TERRAIN ()
            # lstm_model = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_15-24-12/nn/last_AnymalTerrain_ep_3200_rew_20.145746.pth')
            DATA_PATH = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-08-27-16-41_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/'

            # **LSTM16-NODIST 1/4 steps (CoRL), W/ TERRAIN ()
            # lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_07-03-29-04/nn/last_AnymalTerrain_ep_3800_rew_20.163399.pth')
            # DATA_PATH = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-08-27-16-52_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/'

            # **LSTM16-NODIST 4/4 steps, W/ TERRAIN ()
            # lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/2023-08-27-17-23_AnymalTerrain/nn/last_AnymalTerrain_ep_2900_rew_20.2482.pth')
            # DATA_PATH = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-08-28-08-46_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/'

            # *LSTM16-DIST500 4/4 steps, NO TERRAIN (LESS ROBUST W/O TERRAIN!!!)
            # lstm_model = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_14-17-13/nn/last_AnymalTerrain_ep_900_rew_20.139568.pth')
            # DATA_PATH = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-08-27-16-56_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/'

            # **LSTM16-DIST500 4/4 steps, W/ TERRAIN () 0.4-1.0 w/ -3.5xBW 100ms
            DATA_PATH = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-09-05-15-47_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[4]/'


            ### FRONTIERS
            # **LSTM16-DIST500 4/4 steps, W/ TERRAIN () 0.4-1.0 w/ -3.5xBW 100ms
            DATA_PATH = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-09-27-15-49_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

            # load scaler and pca transforms
            # scl_hx = pk.load(open(DATA_PATH + 'A_LSTM_HX_SPEED_SCL.pkl','rb'))
            # pca_hx = pk.load(open(DATA_PATH + 'A_LSTM_HX_SPEED_PCA.pkl','rb'))
            # scl_cx = pk.load(open(DATA_PATH + 'A_LSTM_CX_SPEED_SCL.pkl','rb'))
            # pca_cx = pk.load(open(DATA_PATH + 'A_LSTM_CX_SPEED_PCA.pkl','rb'))
            scl_hc = pk.load(open(DATA_PATH + 'A_LSTM_HC_SCL.pkl','rb'))
            pca_hc = pk.load(open(DATA_PATH + 'A_LSTM_HC_PCA.pkl','rb'))

            # Import the required library
            import matplotlib.pyplot as plt

            # Create the figure before the loop
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')

            # fig2 = plt.figure()
            # ax2 = fig2.add_subplot(111)
            # ax2.set_ylim(0, 5)

            # # Initial plots for setting up Line2D objects
            # line_LF, = ax2.step([], [], c='r', label='LF')
            # line_RH, = ax2.step([], [], c='k', label='RH')
            # line_RF, = ax2.step([], [], c='g', label='RF')
            # line_LH, = ax2.step([], [], c='b', label='LH')

            # ax2.legend(loc="best")
            # x_coord = 250
            # ax2.axvline(x=x_coord, color='gray', linestyle='--')
            # ax2.legend(loc="best")  # 'best' means matplotlib will choose the best location for the legend

            # Initialize the marker outside the loop
            marker, = ax1.plot([0], [0], [0], 'ro')  # Use 'ro' for red circles

            perturb_idx = 0

            import random
            rand_hn_idx = random.sample(range(0, 127), 12)
            rand_cn_idx = random.sample(range(0, 127), 5)

            print('rand_hn_idx:', rand_hn_idx)
            print('rand_cn_idx:', rand_cn_idx)

            ROBOT_ID_START = 0
            ROBOT_ID_END = 99
            ROBOT_PERTURB_IDX = np.arange(0, 256, 1) # np.arange(0, 512, 2)
            HC_PERTURB_IDX = np.arange(0, 256, 1)
            

            # Sample settings
            N_hn = 0 # Number of random numbers for hn
            N_cn = 0  # Number of random numbers for cn

            # Pre-generating indices
            static_indices_hn = [random.sample(range(0, 128), N_hn) for _ in range(400)]
            static_indices_cn = [random.sample(range(0, 128), N_cn) for _ in range(400)]


            for t in range(self.max_steps - 1):
                print("t:", t)
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:

                    # used in computing internal LSTM states
                    if rnn_type == 'lstm':
                        a_h_last = self.states[0][0,:,:] # self.layers_out['a_rnn'][1][0][0,0,:]
                        a_c_last = self.states[1][0,:,:] # self.layers_out['a_rnn'][1][1][0,0,:]
                        c_h_last = self.states[2][0,:,:] # self.layers_out['c_rnn'][1][0][0,0,:]
                        c_c_last = self.states[3][0,:,:] # self.layers_out['c_rnn'][1][1][0,0,:]
                    
                    # neural_state_override = torch.ones_like(self.states[1][0,:,:].cpu())

                    neural_obs_override = None
                    neural_state_in_override = None
                    neural_state_out_override = None

                    ### NEURAL OVERRIDE OBSERVATIONS ###
                    # neural_obs_override = obses
                    # obses[:,0]=0 # u 234/400 = 58.5%
                    # obses[:,1]=0 # v 0/400 = 0%
                    # obses[:,2]=0 # w 324/400 = 81%
                    # obses[:,3]=0 # p 98/400 = 24.5%
                    # obses[:,4]=0 # q 398/400 = 99.5%
                    # obses[:,5]=0 # r 395/400 = 98.75%
                    # obses[:,6]=0 # cos(pitch) 397/400 = 99.25%
                    # obses[:,7]=0 # cos(roll) 104/400 = 26.5%
                    # obses[:,8]=0 # cos(yaw)  399/400 = 99.75%
                    # obses[:,9]=0 # u* 331/400 = 82.75%
                    # obses[:,10]=0 # v* 400/400 = 100%
                    # obses[:,11]=0 r* # 400/400 = 100%
                    # obses[:,12:24]=0 # joint pos 9/400 = 2.25%
                    # obses[:,24:36]=0 # joint vel 290/400 = 72.5%
                    # obses[:,136:176]=0 # height 397/400 = 99.25%
                
                    
                    ### NEURAL OVERRIDE STATES IN ###
                    # neural_state_in_override = self.states #[self.states[0].detach().to('cpu'), self.states[1].detach().to('cpu')]
                    # neural_state_in_override[0][0,:,13] = (self.env.perturb_idx > 0) * 0.205488821246418
                    # neural_state_in_override[0][0,:,56] = (self.env.perturb_idx > 0) * 0.22776731317200002
                    # neural_state_in_override[0][0,:,101] = (self.env.perturb_idx > 0) * 0.59731554871306
                    # neural_state_in_override[0][0,:,108] = (self.env.perturb_idx > 0) * -0.199395715246838


                    ### NEURAL OVERRIDE STATES OUT ###
                    neural_state_out_override = self.states #[self.states[0].detach().to('cpu'), self.states[1].detach().to('cpu')]
                    neural_state_out_override[0][0,:,13] = self.env.perturb_started * 0.205488821246418
                    neural_state_out_override[0][0,:,56] = self.env.perturb_started * 0.22776731317200002
                    neural_state_out_override[0][0,:,101] = self.env.perturb_started * 0.59731554871306
                    neural_state_out_override[0][0,:,108] = self.env.perturb_started * -0.199395715246838

                    action = self.get_action(obses, is_deterministic, neural_obs_override, neural_state_in_override, neural_state_out_override) # neural_obs_override,neural_state_override
                    
                    # compute internal LSTM states - confirmed that both c1 and c2 contribute, (f != ones) does not forget everything : )
                    if rnn_type == 'lstm':
                        x = self.layers_out['actor_mlp']
                        i = torch.sigmoid(torch.matmul(a_w_ii, x.t()).t() + a_b_ii.repeat(self.env.num_environments, 1) + torch.matmul(a_w_hi, a_h_last.t()).t() + a_b_hi.repeat(self.env.num_environments, 1))
                        f = torch.sigmoid(torch.matmul(a_w_if, x.t()).t() + a_b_if.repeat(self.env.num_environments, 1) + torch.matmul(a_w_hf, a_h_last.t()).t() + a_b_hf.repeat(self.env.num_environments, 1))
                        g = torch.tanh(torch.matmul(a_w_ig, x.t()).t() + a_b_ig.repeat(self.env.num_environments, 1) + torch.matmul(a_w_hg, a_h_last.t()).t() + a_b_hg.repeat(self.env.num_environments, 1))
                        o = torch.sigmoid(torch.matmul(a_w_io, x.t()).t() + a_b_io.repeat(self.env.num_environments, 1) + torch.matmul(a_w_ho, a_h_last.t()).t() + a_b_ho.repeat(self.env.num_environments, 1))
                        c1 = f * a_c_last
                        c2 = i * g
                        c = c1 + c2
                        h = o * torch.tanh(c)
                        tensor_dict['A_LSTM_C1X']['data'][t,:,:] = c1
                        tensor_dict['A_LSTM_C2X']['data'][t,:,:] = c2


                        x = self.layers_out['critic_mlp']
                        i = torch.sigmoid(torch.matmul(c_w_ii, x.t()).t() + c_b_ii.repeat(self.env.num_environments, 1) + torch.matmul(c_w_hi, c_h_last.t()).t() + c_b_hi.repeat(self.env.num_environments, 1))
                        f = torch.sigmoid(torch.matmul(c_w_if, x.t()).t() + c_b_if.repeat(self.env.num_environments, 1) + torch.matmul(c_w_hf, c_h_last.t()).t() + c_b_hf.repeat(self.env.num_environments, 1))
                        g = torch.tanh(torch.matmul(c_w_ig, x.t()).t() + c_b_ig.repeat(self.env.num_environments, 1) + torch.matmul(c_w_hg, c_h_last.t()).t() + c_b_hg.repeat(self.env.num_environments, 1))
                        o = torch.sigmoid(torch.matmul(c_w_io, x.t()).t() + c_b_io.repeat(self.env.num_environments, 1) + torch.matmul(c_w_ho, c_h_last.t()).t() + c_b_ho.repeat(self.env.num_environments, 1))
                        c1 = f * c_c_last
                        c2 = i * g
                        c = c1 + c2
                        h = o * torch.tanh(c)
                        tensor_dict['C_LSTM_C1X']['data'][t,:,:] = c1
                        tensor_dict['C_LSTM_C2X']['data'][t,:,:] = c2
                        
                obses, r, done, info = self.env_step(self.env, action)


                # # Ablate correlates of diff p
                # self.states[0][0,:,15] = 0  # [actor lstm cn (short-term memory)
                # self.states[0][0,:,28] = 0 # [actor lstm cn (short-term memory)
                # self.states[0][0,:,31] = 0 # [actor lstm cn (short-term memory)
                # self.states[0][0,:,72] = 0 # [actor lstm cn (short-term memory)
                # self.states[0][0,:,74] = 0 # [actor lstm cn (short-term memory)
                # self.states[0][0,:,95] = 0 # [actor lstm cn (short-term memory)
                # self.states[0][0,:,122] = 0 # [actor lstm cn (short-term memory)
                # self.states[1][0,:,9] = 0 # [actor lstm cn (short-term memory)
                # self.states[1][0,:,28] = 0 # [actor lstm cn (short-term memory)

                # # Ablate correlates of diff v
                # self.states[1][0,:,33] = 0 # [actor lstm cn (short-term memory)
                # self.states[1][0,:,46] = 0 # [actor lstm cn (short-term memory)
                # self.states[1][0,:,94] = 0 # [actor lstm cn (short-term memory)

                # Ablate random hn neurons                
                # self.states[0][0,:,rand_hn_idx] = 0 * torch.ones_like(self.states[0][0,:,rand_hn_idx]) # [actor lstm hn (short-term memory)
                # self.states[1][0,:,rand_cn_idx] = 0 * torch.ones_like(self.states[0][0,:,rand_cn_idx]) # [actor lstm cn (short-term memory)

                # Ablate first ten cn neurons
                # self.states[1][0,:,:10] = 0 * torch.rand_like(self.states[0][0,:,:10]) # [actor lstm hn (short-term memory)

                # # Ablate neurons contributing to A_LSTM_HC_PC_001 when disturbance
                # self.states[0][0,:,107] = 0 # [actor lstm cn (short-term memory)
                # self.states[0][0,:,114] = 0 # [actor lstm cn (short-term memory)

                # hx = self.states[0][0,:,:]
                # cx = self.states[1][0,:,:]

                # neural perturbations

                # if t > 100 and t % 130 == 0:
                if t > 250:
                    # obses[:,perturb_idx] += 1
                    # hc_pc[ROBOT_PERTURB_IDX,HC_PERTURB_IDX] += 25

                    # hc_pc[ROBOT_ID_START:ROBOT_ID_END,perturb_idx] += 25 # * cx_pc[:,:256]
                    # cx_pc[ROBOT_ID_START:ROBOT_ID_END,2] += 1e3 # * cx_pc[:,:256]
                    # hc_pc[ROBOT_ID_START:ROBOT_ID_END, 0] *= 2.0 # * cx_pc[:,:256]
                    # hx = torch.tensor(scl_hx.inverse_transform(pca_hx.inverse_transform(hx_pc)), dtype=torch.float32).unsqueeze(dim=0)
                    # cx = torch.tensor(scl_cx.inverse_transform(pca_cx.inverse_transform(cx_pc)), dtype=torch.float32).unsqueeze(dim=0)

                    if t > 250:
                        
                        ### Applies to all agents
                        # with torch.no_grad():

                            ### robot stops walking, falls over
                            # self.model.a2c_network.a_rnn.rnn.weight_ih_l0 *= 0
                            
                            ### same behavior, just slightly less robust to perturbations
                            # self.model.a2c_network.a_rnn.rnn.weight_hh_l0 *= 0

                        ### robot walks, but a bit slower
                        # hc_pc[ROBOT_ID_START,:] *= 0 # * cx_pc[:,:256]

                    # with torch.no_grad():
                        # self.model.a2c_network.actor_mlp[0].weight[:,0] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # u # run v fast, much less robust
                        # self.model.a2c_network.actor_mlp[0].weight[:,1] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # v # wals sideways more, much less robust
                        # self.model.a2c_network.actor_mlp[0].weight[:,2] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # w # slightly less robust
                        # self.model.a2c_network.actor_mlp[0].weight[:,3] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # p # much less robust
                        # self.model.a2c_network.actor_mlp[0].weight[:,4] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # q # slightly less robust
                        # self.model.a2c_network.actor_mlp[0].weight[:,5] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # r # turns more, less robust
                        # self.model.a2c_network.actor_mlp[0].weight[:,6] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # proj grav x (pitch), slightly less robust
                        # self.model.a2c_network.actor_mlp[0].weight[:,7] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # proj grav y (roll), much less robust
                        # self.model.a2c_network.actor_mlp[0].weight[:,8] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # proj grav z (pitch/roll), slightly less robust
                        # self.model.a2c_network.actor_mlp[0].weight[:,9] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # u*, less robust (not moving)
                        # self.model.a2c_network.actor_mlp[0].weight[:,10] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # v*, no diff
                        # self.model.a2c_network.actor_mlp[0].weight[:,11] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # w*, no diff
                        # self.model.a2c_network.actor_mlp[0].weight[:,12] = torch.nn.Parameter(torch.zeros(512, dtype=torch.float, device="cuda")) # dof pos 1

                    # when both dof pos and dof vel are zeroed out, robot doesn't walk. When only one or the other is, then robot walks! Strange!
                        # self.model.a2c_network.actor_mlp[0].weight[:,12:24] = torch.nn.Parameter(torch.zeros([512,12], dtype=torch.float, device="cuda")) # dof pos, surprised still walks, but much less robust
                        # self.model.a2c_network.actor_mlp[0].weight[:,24:36] = torch.nn.Parameter(torch.zeros([512,12], dtype=torch.float, device="cuda")) # dof pos, surprised still walks, but less robust

                        # self.model.a2c_network.actor_mlp[0].weight[:,36:176] = torch.nn.Parameter(torch.zeros([512,140], dtype=torch.float, device="cuda")) # depth sensors
                        # self.model.a2c_network.actor_mlp[0].weight[:,176:] = torch.nn.Parameter(torch.zeros([512,12], dtype=torch.float, device="cuda")) # actions




                        # self.states[0][0,:,:] = 0 # [actor lstm hn (short-term memory)
                        # self.states[1][0,:,:] = 0 # [actor lstm cn (long-term memory)
                                            
                        # self.states[0][0,:,rand_hn_idx] = 0 * torch.ones_like(self.states[0][0,:,rand_hn_idx]) # [actor lstm hn (short-term memory)
                        # self.states[1][0,:,rand_cn_idx] = 0 * torch.ones_like(self.states[1][0,:,rand_cn_idx]) # [actor lstm cn (short-term memory)

                        # Ablate neurons contributing to A_LSTM_HC_PC_001 when disturbance

                        # pass

                        # # # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  < -0.03 --> # -3.5BW: 99%
                        # indices_hn = [8, 15, 77, 87, 91]
                        # indices_cn = [9, 46, 58]
                        # for idx in indices_hn:
                        #     self.states[0][0,:,idx] = scl_hc.mean_[idx] * torch.ones_like(self.states[0][0,:,idx].cpu())
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[0][0,:,idx].cpu())


                        # # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  < -0.02 --> # -3.5BW: 97%
                        # indices_hn = [1, 8, 15, 75, 77, 87, 91, 93, 122]
                        # indices_cn = [9, 28, 46, 58, 75, 92, 110, 116, 122]
                        # for idx in indices_hn:
                        #     self.states[0][0,:,idx] = scl_hc.mean_[idx] * torch.ones_like(self.states[0][0,:,idx].cpu())
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[0][0,:,idx].cpu())


                        # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  < -0.01 --> # -3.5BW: 38%
                        # indices_hn = [1,8,15,21,28,29,32,38,43,44,45,62,65,75,77,78,82,87,90,91,93,94,97,106,107,114,118,119,122]
                        # indices_cn = [0,4,6,9,22,28,33,39,41,42,44,46,58,63,72,73,75,87,91,92,98,102,104,105,110,112,114,115,116,118,119,120,122,124,126]
                        # for idx in indices_hn:
                        #     self.states[0][0,:,idx] = scl_hc.mean_[idx] * torch.ones_like(self.states[0][0,:,idx].cpu())
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())


                        # # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  > 0 --> # -3.5BW: 7%
                        # indices_hn = [0,2,4,5,6,7,9,11,12,14,16,17,18,22,24,27,30,31,34,35,36,37,40,41,42,47,48,50,52,53,54,56,57,58,59,60,61,63,66,67,68,70,73,79,80,81,83,85,88,92,95,96,98,99,100,101,102,104,109,110,112,113,115,116,117,123,124,125,126,127]
                        # indices_cn = [1,2,3,7,8,10,13,15,16,17,19,20,21,23,24,26,29,32,34,35,37,40,43,45,47,48,49,50,51,53,55,56,60,61,62,64,65,68,69,74,76,77,78,80,82,86,88,89,90,95,99,100,103,106,108,113,117,121,123,125,127]

                        # for idx in indices_hn:
                        #     self.states[0][0,:,idx] = scl_hc.mean_[idx] * torch.ones_like(self.states[0][0,:,idx].cpu())
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())


                        # # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  > -0.01 --> # -3.5BW: 9%
                        # indices_hn = [0,2,3,4,5,6,7,9,10,11,12,13,14,16,17,18,19,20,22,23,24,25,26,27,30,31,33,34,35,36,37,39,40,41,42,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,63,64,66,67,68,69,70,71,72,73,74,76,79,80,81,83,84,85,86,88,89,92,95,96,98,99,100,101,102,103,104,105,108,109,110,111,112,113,115,116,117,120,121,123,124,125,126,127]
                        # indices_cn = [1,2,3,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,29,30,31,32,34,35,36,37,38,40,43,45,47,48,49,50,51,52,53,54,55,56,57,59,60,61,62,64,65,66,67,68,69,70,71,74,76,77,78,79,80,81,82,83,84,85,86,88,89,90,93,94,95,96,97,99,100,101,103,106,107,108,109,111,113,117,121,123,125,127]

                        # for idx in indices_hn:
                        #     self.states[0][0,:,idx] = scl_hc.mean_[idx] * torch.ones_like(self.states[0][0,:,idx].cpu())
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())


                        # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  > 0.01 --> # -3.5BW: 100%
                        # indices_hn = [17,22,36,52,54,59,60,81,104,115,125,126,127]
                        # indices_cn = [34,60,68,76,82,100,106,121]
                        
                        # for idx in indices_hn:
                        #     self.states[0][0,:,idx] = scl_hc.mean_[idx] * torch.ones_like(self.states[0][0,:,idx].cpu())
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())

                        # Gather statistics on random ablations of N neurons
                        if N_hn > 0:
                            for i in range(400):  # Assuming self.states.size(1) is 400
                                mean_tensor_hn = torch.tensor(scl_hc.mean_[      np.array(static_indices_hn[i])], dtype=self.states[0].dtype).to(self.device)
                                self.states[0][0, i, static_indices_hn[i]] = mean_tensor_hn * torch.ones_like(self.states[0][0, i, static_indices_hn[i]])

                                if done[i]:
                                    print("hn neurons ablated", static_indices_hn[i])

                        if N_cn > 0:
                            for i in range(400):  # Assuming self.states.size(1) is 400
                                mean_tensor_cn = torch.tensor(scl_hc.mean_[128 + np.array(static_indices_cn[i])], dtype=self.states[1].dtype).to(self.device)
                                self.states[1][0, i, static_indices_cn[i]] = mean_tensor_cn * torch.ones_like(self.states[1][0, i, static_indices_cn[i]])
                        
                                if done[i]:
                                    print("cn neurons ablated", static_indices_cn[i])


                        # # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  > 0.01 --> # -3.5BW: 383/400=96%
                        # indices_cn = [18]
                        
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())


                        # # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  > 0.01 --> # -3.5BW: 300/400=75%
                        # indices_cn = [18,73]
                        
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())


                        # # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  > 0.01 --> # -3.5BW: 393/400=98%
                        # indices_cn = [73]
                        
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())


                        # # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  > 0.01 --> # -3.5BW: 231/400=58%
                        # indices_cn = [18,30,54,73,77]
                        
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())


                        # # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  > 0.01 --> # -3.5BW: 206/400=52%
                        # indices_cn = [18,30,54,73,74,77,88,108]
                        
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())



                        # # Ablate [del(RHhip)/del(hc) *47, hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  > 0.01 --> # -3.5BW: 206/400=52%
                        # indices_cn = [2,6,30,42,101,108]
                        
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())



                        # # Ablate [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  > 0.01 --> # -3.5BW: 150/400=38%
                        # indices_cn = [6,18,73,94]
                        
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())


                        # Ablate ALL hn
                        # self.states[0][0,:,:] = torch.from_numpy(scl_hc.mean_[:128]) * torch.ones_like(self.states[0][0,:,:].cpu()) # -3.5BW: 50% 

                        # Ablate ALL cn
                        # self.states[1][0,:,:] = torch.from_numpy(scl_hc.mean_[128:]) * torch.ones_like(self.states[1][0,:,:].cpu()) # -3.5BW: 0% 

                        # based on a_rnn grad and a_rnn activations (MAYBE THE WRONG PLACE TO BE ABLATING. MAYBE NEED TO DO AFTER RNN UPDATE BEFORE ACTION SELECTION. INSIDE GET_ACTION
                        # self.states[0][0,:,10] = scl_hc.mean_[10] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,13] = scl_hc.mean_[13] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,48] = scl_hc.mean_[48] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,56] = scl_hc.mean_[56] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,66] = scl_hc.mean_[66] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,98] = scl_hc.mean_[98] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,101] = scl_hc.mean_[101] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,108] = scl_hc.mean_[108] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,114] = scl_hc.mean_[114] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover

                        # self.states[0][0,:,13] = scl_hc.mean_[13] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # # self.states[0][0,:,47] = scl_hc.mean_[47] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # # self.states[0][0,:,48] = scl_hc.mean_[48] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,56] = scl_hc.mean_[56] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # # self.states[0][0,:,68] = scl_hc.mean_[68] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # # self.states[0][0,:,98] = scl_hc.mean_[98] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,101] = scl_hc.mean_[101] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # # self.states[0][0,:,103] = scl_hc.mean_[103] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # self.states[0][0,:,108] = scl_hc.mean_[108] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover
                        # # self.states[0][0,:,114] = scl_hc.mean_[114] * torch.ones_like(self.states[0][0,:,0].cpu()) # -3.5BW: 50% robots recover


                    # POS
                    # self.states[0][0,:,30] = 0 * torch.ones_like(self.states[0][0,:,30]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,47] = 0 * torch.ones_like(self.states[0][0,:,47]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,54] = 0 * torch.ones_like(self.states[0][0,:,54]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,55] = 0 * torch.ones_like(self.states[0][0,:,55]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,56] = 0 * torch.ones_like(self.states[0][0,:,56]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,67] = 0 * torch.ones_like(self.states[0][0,:,67]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,68] = 0 * torch.ones_like(self.states[0][0,:,68]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,84] = 0 * torch.ones_like(self.states[0][0,:,84]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,98] = 0 * torch.ones_like(self.states[0][0,:,98]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,101] = 0 * torch.ones_like(self.states[0][0,:,101]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,107] = 0 * torch.ones_like(self.states[0][0,:,107]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,112] = 0 * torch.ones_like(self.states[0][0,:,112]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,114] = 0 * torch.ones_like(self.states[0][0,:,114]) # [actor lstm hn (short-term memory)

                    # self.states[1][0,:,21] = 0 * torch.ones_like(self.states[1][0,:,21]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,47] = 0 * torch.ones_like(self.states[1][0,:,47]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,67] = 0 * torch.ones_like(self.states[1][0,:,67]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,68] = 0 * torch.ones_like(self.states[1][0,:,68]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,107] = 0 * torch.ones_like(self.states[1][0,:,107]) # [actor lstm cn (short-term memory)

                    # NEG
                    # self.states[0][0,:,2] = 0 * torch.ones_like(self.states[0][0,:,2]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,9] = 0 * torch.ones_like(self.states[0][0,:,9]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,16] = 0 * torch.ones_like(self.states[0][0,:,16]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,33] = 0 * torch.ones_like(self.states[0][0,:,33]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,42] = 0 * torch.ones_like(self.states[0][0,:,42]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,64] = 0 * torch.ones_like(self.states[0][0,:,64]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,94] = 0 * torch.ones_like(self.states[0][0,:,94]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,109] = 0 * torch.ones_like(self.states[0][0,:,109]) # [actor lstm hn (short-term memory)
                    # self.states[0][0,:,121] = 0 * torch.ones_like(self.states[0][0,:,121]) # [actor lstm hn (short-term memory)

                    # self.states[1][0,:,2] = 0 * torch.ones_like(self.states[1][0,:,2]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,16] = 0 * torch.ones_like(self.states[1][0,:,16]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,18] = 0 * torch.ones_like(self.states[1][0,:,18]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,33] = 0 * torch.ones_like(self.states[1][0,:,33]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,46] = 0 * torch.ones_like(self.states[1][0,:,46]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,74] = 0 * torch.ones_like(self.states[1][0,:,74]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,81] = 0 * torch.ones_like(self.states[1][0,:,81]) # [actor lstm cn (short-term memory)

                    # self.states[1][0,:,8] = 0 * torch.ones_like(self.states[1][0,:,2]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,46] = 0 * torch.ones_like(self.states[1][0,:,46]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,72] = 0 * torch.ones_like(self.states[1][0,:,74]) # [actor lstm cn (short-term memory)
                    # self.states[1][0,:,81] = 0 * torch.ones_like(self.states[1][0,:,81]) # [actor lstm cn (short-term memory)

                hc = torch.cat((self.states[0][0,:,:], self.states[1][0,:,:]), dim=1)
                # # hx_pc = pca_hx.transform(scl_hx.transform(torch.squeeze(hx).detach().cpu().numpy()))
                # # cx_pc = pca_cx.transform(scl_cx.transform(torch.squeeze(cx).detach().cpu().numpy()))
                hc_pc = pca_hc.transform(scl_hc.transform(hc.detach().cpu().numpy()))

                # hc = torch.tensor(scl_hc.inverse_transform(pca_hc.inverse_transform(hc_pc)), dtype=torch.float32).unsqueeze(dim=0)

                    # self.states[0][0,ROBOT_ID_START:ROBOT_ID_END,:] = hx[:,ROBOT_ID_START:ROBOT_ID_END,:]
                    # self.states[1][0,ROBOT_ID_START:ROBOT_ID_END,:] = cx[:,ROBOT_ID_START:ROBOT_ID_END,:]
                    # self.states[0][0,ROBOT_ID_START:ROBOT_ID_END,:] = hc[:,ROBOT_ID_START:ROBOT_ID_END,:DIM_A_LSTM_HX]
                    # self.states[1][0,ROBOT_ID_START:ROBOT_ID_END,:] = hc[:,ROBOT_ID_START:ROBOT_ID_END,DIM_A_LSTM_HX:]

                    # perturb_idx += 1

                # hx_pc_last = hx_pc
                # cx_pc_last = cx_pc
                
                hc_pc_last = hc_pc

                # hc = torch.cat((self.states[0][0,:,:], self.states[1][0,:,:]), dim=1)
                # # hx_pc = pca_hx.transform(scl_hx.transform(torch.squeeze(hx).detach().cpu().numpy()))
                # # cx_pc = pca_cx.transform(scl_cx.transform(torch.squeeze(cx).detach().cpu().numpy()))
                # hc_pc = pca_hc.transform(scl_hc.transform(hc.detach().cpu().numpy()))



                if self.export_data:                  

                    condition = torch.arange(self.env.num_environments)
                    # condition = torch.arange(self.env.num_environments / 5).repeat(5)
                    time = torch.Tensor([t * self.env.dt]).repeat(self.env.num_environments)

                    if self.env.cfg['name'] == 'AnymalTerrain' or self.env.cfg['name'] == 'A1Terrain':
                        tensor_dict['FT_FORCE']['data'][t,:,:] = info['foot_forces']
                        tensor_dict['PERTURB_BEGIN']['data'][t,:,:] = info['perturb_begin'].view(-1, 1)
                        tensor_dict['PERTURB']['data'][t,:,:] = info['perturb'].view(-1, 1)
                        tensor_dict['STANCE_BEGIN']['data'][t,:,:] = info['stance_begin'].view(-1, 1)
                    tensor_dict['OBS']['data'][t,:,:] = obses[:,:DIM_OBS]
                    tensor_dict['ACT']['data'][t,:,:] = action
                    tensor_dict['REWARD']['data'][t,:,:] = torch.unsqueeze(r[:], dim=1)
                    tensor_dict['DONE']['data'][t,:,:] = torch.unsqueeze(done[:], dim=1)

                    tensor_dict['A_MLP_XX']['data'][t,:,:] = self.layers_out['actor_mlp'] # torch.squeeze(self.states[2][0,:,:]) # lstm hn (short-term memory)
                    tensor_dict['C_MLP_XX']['data'][t,:,:] = self.layers_out['critic_mlp'] # lstm cn (long-term memory)

                    if rnn_type == 'lstm':
                        tensor_dict['A_LSTM_HX']['data'][t,:,:] = torch.squeeze(self.states[0][0,:,:]) # lstm hn (short-term memory)
                        tensor_dict['A_LSTM_CX']['data'][t,:,:] = torch.squeeze(self.states[1][0,:,:]) # lstm cn (long-term memory)
                        tensor_dict['A_LSTM_HC']['data'][t,:,:] = torch.cat((tensor_dict['A_LSTM_HX']['data'][t,:,:], tensor_dict['A_LSTM_CX']['data'][t,:,:]), dim=1)

                        tensor_dict['C_LSTM_HX']['data'][t,:,:] = torch.squeeze(self.states[2][0,:,:]) # lstm hn (short-term memory)
                        tensor_dict['C_LSTM_CX']['data'][t,:,:] = torch.squeeze(self.states[3][0,:,:]) # lstm cn (long-term memory)
                        tensor_dict['C_LSTM_HC']['data'][t,:,:] = torch.cat((tensor_dict['C_LSTM_HX']['data'][t,:,:], tensor_dict['C_LSTM_CX']['data'][t,:,:]), dim=1)

                    elif rnn_type == 'gru':
                        tensor_dict['A_GRU_HX']['data'][t,:,:] = torch.squeeze(self.states[0][0,:,:])
                        tensor_dict['C_GRU_HX']['data'][t,:,:] = torch.squeeze(self.states[1][0,:,:])
                    else:
                        print("rnn model not supported")

                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count
                print('gameslost:', games_played, '    games won:', self.env.num_environments - games_played)
                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                          all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards/done_count
                        cur_steps_done = cur_steps/done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1} w: {game_res:.1}')
                        else:
                            print(f'reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break
                
                plot = True

                if plot:

                # if t > 0:
                    # Gait plot
                    
                    # Plot the line of the last agent

                    # force_LF = tensor_dict['FT_FORCE']['data'][:,:,0] # LF
                    # force_LH = tensor_dict['FT_FORCE']['data'][:,:,1] # LH
                    # force_RF = tensor_dict['FT_FORCE']['data'][:,:,2] # RF
                    # force_RH = tensor_dict['FT_FORCE']['data'][:,:,3] # RH
                                        
                    # # Convert each force data to binary foot contact for all agents
                    # contact_LF = np.where(force_LF > 0, 4, 0).sum(axis=1)/100
                    # contact_LH = np.where(force_LH > 0, 1, 0).sum(axis=1)/100
                    # contact_RF = np.where(force_RF > 0, 3, 0).sum(axis=1)/100
                    # contact_RH = np.where(force_RH > 0, 2, 0).sum(axis=1)/100

                    # # # Convert each force data to binary foot contact
                    # # contact_LF = np.where(force_LF > 0, 4, np.nan)
                    # # contact_LH = np.where(force_LH > 0, 1, np.nan)
                    # # contact_RF = np.where(force_RF > 0, 3, np.nan)
                    # # contact_RH = np.where(force_RH > 0, 2, np.nan)
                    
                    # line_LF.set_ydata(contact_LF)
                    # line_LF.set_xdata(range(len(contact_LF)))
                    # line_LH.set_ydata(contact_LH)
                    # line_LH.set_xdata(range(len(contact_LH)))
                    # line_RF.set_ydata(contact_RF)
                    # line_RF.set_xdata(range(len(contact_RF)))
                    # line_RH.set_ydata(contact_RH)
                    # line_RH.set_xdata(range(len(contact_RH)))
                    
                    # # ax2.relim()
                    # # ax2.autoscale_view()
                    # plt.draw()
                    # plt.pause(0.0001)
                    
                    # # Set the title of the plot
                    # ax2.set_title(f'Timestep: {t}')
                    # plt.draw()
                    # plt.pause(0.01)

                #     # Update plot
                    
                    # Plot the line of the last agent
                    ax1.plot(
                        [hc_pc[ROBOT_ID_START, 0], hc_pc_last[ROBOT_ID_START, 0]], 
                        [hc_pc[ROBOT_ID_START, 1], hc_pc_last[ROBOT_ID_START, 1]],
                        [hc_pc[ROBOT_ID_START, 2], hc_pc_last[ROBOT_ID_START, 2]],
                        c='k')

                    # Update the marker position
                    marker.set_data([hc_pc[ROBOT_ID_START, 0]], [hc_pc[ROBOT_ID_START, 1]])
                    marker.set_3d_properties([hc_pc[ROBOT_ID_START, 2]])
                    
                    # Set the title of the plot
                    ax1.set_title(f'Timestep: {t}')
                    plt.draw()
                    plt.pause(0.01)


        print(sum_rewards)
        # if print_game_res:
        #     print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
        #           games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        # else:
        #     print('av reward:', sum_rewards / games_played * n_game_life,
                #   'av steps:', sum_steps / games_played * n_game_life)

        def export_torch2parquet(data: torch.Tensor, columns: np.array, name: str):
            df = pd.DataFrame(data.cpu().numpy())
            df.columns = columns
            df.to_csv(name + '.csv')

        t0 = 0 # 1000 # 500 # 9500 # 950 # 100 # 500
        tf = 500 # 3000 # 1500 # 1500 # 10000 # 1000 # 600 # 527

        if self.export_data:
            
            def extract_tensor_data(tensor_dict_entry: dict, t0: int, tf: int) -> pd.DataFrame:
                tensor = tensor_dict_entry['data']
                columns = tensor_dict_entry['cols']
                data = tensor[t0:tf,:,:].permute(1, 0, 2).contiguous().view(tensor[t0:tf,:,:].size()[0] * tensor[t0:tf,:,:].size()[1], tensor[t0:tf,:,:].size()[2])
                df = pd.DataFrame(data.cpu().numpy(), columns=columns)
                return df

            # Extract the data from all tensors and concatenate into a single dataframe
            data_frames = [extract_tensor_data(v, t0, tf) for v in tensor_dict.values()]
            all_data = pd.concat(data_frames, axis=1)

            # generate a folder to save data in
            current_datetime = datetime.datetime.now()

            # Create a folder name using the current date and time
            date_str = current_datetime.strftime("%Y-%m-%d-%H-%M")

            if self.env.cfg['name'] == 'AnymalTerrain' or self.env.cfg['name'] == 'A1Terrain':
                exp_str = f"_u[\
                    {self.env.specified_command_x_range[0]},\
                    {self.env.specified_command_x_range[1]},\
                    {self.env.specified_command_x_no}]_v[\
                    {self.env.specified_command_y_range[0]},\
                    {self.env.specified_command_y_range[1]},\
                    {self.env.specified_command_y_no}]_r[\
                    {self.env.specified_command_yawrate_range[0]},\
                    {self.env.specified_command_yawrate_range[1]},\
                    {self.env.specified_command_yawrate_no}]_n[\
                    {self.env.specified_command_no_copies}]"

            if self.env.cfg['name'] == 'ShadowHand':
                exp_str = f"_u[\
                    {self.env.specified_command_roll_range[0]},\
                    {self.env.specified_command_roll_range[1]},\
                    {self.env.specified_command_roll_no}]_v[\
                    {self.env.specified_command_pitch_range[0]},\
                    {self.env.specified_command_pitch_range[1]},\
                    {self.env.specified_command_pitch_no}]_r[\
                    {self.env.specified_command_yaw_range[0]},\
                    {self.env.specified_command_yaw_range[1]},\
                    {self.env.specified_command_yaw_no}]"
            
            # Remove the spaces from the string
            exp_str = exp_str.replace(' ', '')
            
            # create data folder (if it does not exist)
            p_data = Path().resolve() / 'data'
            p_data.mkdir(exist_ok=True)

            # create specific folder
            p =  date_str + exp_str
            p = Path().resolve() / 'data' / p
            p.mkdir(exist_ok=True)

            # Save the dataframe to a CSV file
            all_data.to_parquet(str(p / 'RAW_DATA.parquet'))
            all_data.to_csv(str(p / 'RAW_DATA.csv'))

            # export the model used for data collection
            with open(p.joinpath('model.txt'), 'w') as file:
                file.write(self.config['name'])

            self.env.close_viewer()

            # # Normalize the data!
            # scaler = sklearn.preprocessing.StandardScaler()
            # scaled_data_frames = []
            # for v in tensor_dict.values():
            #     if v['data'].shape[-1] > 4:  # Only scale tensors with dim > 4
            #         df = extract_tensor_data(v, t0, tf)
            #         df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            #         scaled_data_frames.append(df_scaled)
            #     else:
            #         scaled_data_frames.append(extract_tensor_data(v, t0, tf))  # Don't scale tensors with dim <=4

            # all_data_scaled = pd.concat(scaled_data_frames, axis=1)
            # all_data_scaled.to_csv(str(p / 'NORM_DATAA.csv'), index=False)

            # # Normalize and save tensors not present in the scaling blocklist
            # scaler = StandardScaler()
            # scaled_data_frames = []
            # for k, v in tensor_dict.items():
            #     if k not in scaling_blocklist:
            #         df = extract_tensor_data(v, t0, tf)
            #         df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            #         scaled_data_frames.append(df_scaled)
            #     else:
            #         scaled_data_frames.append(extract_tensor_data(v, t0, tf))

            # all_data_scaled = pd.concat(scaled_data_frames, axis=1)
            # all_data_scaled.to_csv(str(p / 'NORM_DATA.csv'), index=False)


            # export_torch2parquet(DATA, COLUMNS, 'data/' + 'RAW_DATA')

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if 'obs' in obses:
                obses = obses['obs']
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if 'observation' in obses:
                first_key = 'observation'
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size
