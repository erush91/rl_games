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

        self.ablation_trial = self.player_config.get('ablation_trial', False)
        self.ablation_trial_config = self.player_config.get('ablate', {})
        self.targeted_ablation_trial = self.ablation_trial_config.get('targeted_ablation_trial', False)
        self.wait_until_disturbance = self.ablation_trial_config.get('wait_until_disturbance', False)
        self.ablations_hn_out = self.ablation_trial_config.get('ablations_hn_out', 0)
        self.ablations_hn_in = self.ablation_trial_config.get('ablations_hn_in', 0)
        self.ablations_cn = self.ablation_trial_config.get('ablations_cn', 0)

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
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size()[2]), dtype=torch.float32, requires_grad=True).to(self.device) for s in rnn_states]

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

            ROBOT_ID_PLOT = 0            

            N_ROBOTS = 400

            # Sample settings
            N_hn = 0 #26 # Number of random numbers for hn
            N_cn = 0 # Number of random numbers for cn

            # Pre-generating indices
            # Initialize the tensor with all Falses

            # Generate random row indices for each row
            row_indices = np.tile(np.arange(DIM_A_LSTM_HX), (N_ROBOTS, 1))

            # Shuffle row_indices randomly for each row
            np.apply_along_axis(np.random.shuffle, 1, row_indices)

            # Initialize with mean neural values
            ablate_hn_out_np = np.tile(scl_hc.mean_[:128], (N_ROBOTS,1))
            ablate_hn_in_np = np.tile(scl_hc.mean_[:128], (N_ROBOTS,1))
            ablate_cn_np = np.tile(scl_hc.mean_[128:], (N_ROBOTS,1))

            random_ablations = False
            target_ablations = True

            if self.ablation_trial:

                # TARGETED
                if self.targeted_ablation_trial:

                    hn_out_idx_by_ascending_gradient = [101,56,13,108,68,48,98,103,114,47,83,84,90,30,82,69,85,6,111,42,18,35,4,12,22,109,0,2,87,124,112,104,99,102,59,32,49,72,63,45,110,93,14,70,91,5,106,24,7,127,3,65,97,41,118,117,95,64,39,20,34,27,105,79,94,61,89,31,126,19,25,121,115,96,52,71,1,88,44,46,123,113,8,73,62,37,86,100,119,15,51,125,77,28,116,53,16,80,78,9,122,120,40,50,81,66,33,75,67,60,74,11,92,57,26,36,23,54,58,10,17,21,76,29,43,107,38,55] # hn out
                    targeted_hn_out = hn_out_idx_by_ascending_gradient[self.ablations_hn_out:]
                    # targeted_hn = hn_out_idx_by_ascending_gradient[1:] # 395
                    # targeted_hn = hn_out_idx_by_ascending_gradient[2:] # 397
                    # targeted_hn = hn_out_idx_by_ascending_gradient[3:] # 242
                    # targeted_hn = hn_out_idx_by_ascending_gradient[4:] # 168
                    # targeted_hn = hn_out_idx_by_ascending_gradient[5:] # 270
                    # targeted_hn = hn_out_idx_by_ascending_gradient[6:] # 241
                    # targeted_hn = hn_out_idx_by_ascending_gradient[7:] # 148
                    # targeted_hn = hn_out_idx_by_ascending_gradient[8:] # 188
                    # targeted_hn = hn_out_idx_by_ascending_gradient[9:] # 117
                    # targeted_hn = hn_out_idx_by_ascending_gradient[10:] # 23
                    # targeted_hn = hn_out_idx_by_ascending_gradient[11:] # 46
                    # targeted_hn = hn_out_idx_by_ascending_gradient[12:] # 1
                    # targeted_hn = hn_out_idx_by_ascending_gradient[13:] # 0
                    # targeted_hn = hn_out_idx_by_ascending_gradient[14:] # 0
                    # targeted_hn = hn_out_idx_by_ascending_gradient[15:] # 0
                    # targeted_hn = hn_out_idx_by_ascending_gradient[16:] # 0
                    # targeted_hn = hn_out_idx_by_ascending_gradient[32:] # 0
                    # targeted_hn = hn_out_idx_by_ascending_gradient[64:] # 0
                    # targeted_hn = hn_out_idx_by_ascending_gradient[96:] # 0
                    # targeted_hn = hn_out_idx_by_ascending_gradient[128:] # 0
                    ablate_hn_out_np[:,targeted_hn_out] = torch.nan
                    ablate_hn_out = torch.tensor(ablate_hn_out_np, dtype=torch.float, device='cuda').unsqueeze(0)




                    # hn_in_idx_by_ascending_gradient = [101,56,13,108,68,48,98,103,114,47,83,84,90,30,82,69,85,6,111,42,18,35,4,12,22,109,0,2,87,124,112,104,99,102,59,32,49,72,63,45,110,93,14,70,91,5,106,24,7,127,3,65,97,41,118,117,95,64,39,20,34,27,105,79,94,61,89,31,126,19,25,121,115,96,52,71,1,88,44,46,123,113,8,73,62,37,86,100,119,15,51,125,77,28,116,53,16,80,78,9,122,120,40,50,81,66,33,75,67,60,74,11,92,57,26,36,23,54,58,10,17,21,76,29,43,107,38,55] # hn out

                    hn_in_idx_by_ascending_gradient = [27,126,121,12,11,110,35,70,54,0,13,31,56,114,34,101,115,111,61,26,6,55,90,49,5,98,113,53,64,37,104,43,72,28,22,59,19,21,87,107,63,88,51,76,30,44,50,82,60,123,94,42,52,78,92,109,57,96,77,1,99,95,8,86,9,125,122,91,15,2,71,17,41,62,20,117,79,80,67,24,4,39,116,10,93,65,81,89,46,120,23,118,33,85,66,112,14,73,97,83,38,105,84,102,69,40,3,127,32,103,16,7,124,58,100,47,36,108,29,119,75,106,74,48,45,68,25,18] # hn out
                    targeted_hn_in = hn_in_idx_by_ascending_gradient[self.ablations_hn_in:]
                    # targeted_hn = hn_out_idx_by_ascending_gradient[1:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[2:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[3:] # 399
                    # targeted_hn = hn_out_idx_by_ascending_gradient[4:] # 399
                    # targeted_hn = hn_out_idx_by_ascending_gradient[5:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[6:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[7:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[8:] # 399
                    # targeted_hn = hn_out_idx_by_ascending_gradient[9:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[10:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[11:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[12:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[13:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[14:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[15:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[16:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[32:] # 400
                    # targeted_hn = hn_out_idx_by_ascending_gradient[64:] # 382
                    # targeted_hn = hn_out_idx_by_ascending_gradient[96:] # 384
                    # targeted_hn = hn_out_idx_by_ascending_gradient[128:] # 319
                    ablate_hn_in_np[:,targeted_hn_in] = torch.nan
                    ablate_hn_in = torch.tensor(ablate_hn_in_np, dtype=torch.float, device='cuda').unsqueeze(0)

                    # cn_in_idx_by_ascending_gradient = [6, 18, 73, 94] #cn (from sampling)
                    cn_in_idx_by_ascending_gradient = [108,6,101,2,30,47,13,42,89,90,118,98,85,124,99,68,32,24,10,72,3,61,19,109,21,31,22,103,56,69,114,78,46,81,35,97,84,110,4,104,14,50,63,49,79,77,93,106,44,62,70,73,91,41,11,102,95,120,23,88,26,5,18,80,48,20,125,117,112,52,65,36,37,64,127,96,126,55,59,115,33,58,83,123,71,1,27,100,39,94,8,9,53,116,86,113,67,34,51,76,28,40,25,16,60,15,87,54,105,122,92,0,119,45,75,74,107,29,7,111,66,121,17,82,57,12,38,43] #cn
                    targeted_cn = cn_in_idx_by_ascending_gradient[self.ablations_cn:]
                    # targeted_cn = cn_in_idx_by_ascending_gradient[1:] # 400
                    # targeted_cn = cn_in_idx_by_ascending_gradient[2:] # 400
                    # targeted_cn = cn_in_idx_by_ascending_gradient[3:] # 398
                    # targeted_cn = cn_in_idx_by_ascending_gradient[4:] # 391
                    # targeted_cn = cn_in_idx_by_ascending_gradient[5:] # 392
                    # targeted_cn = cn_in_idx_by_ascending_gradient[6:] # 388
                    # targeted_cn = cn_in_idx_by_ascending_gradient[7:] # 371
                    # targeted_cn = cn_in_idx_by_ascending_gradient[8:] # 364
                    # targeted_cn = cn_in_idx_by_ascending_gradient[9:] # 233
                    # targeted_cn = cn_in_idx_by_ascending_gradient[10:] # 234
                    # targeted_cn = cn_in_idx_by_ascending_gradient[11:] # 295
                    # targeted_cn = cn_in_idx_by_ascending_gradient[12:] # 299
                    # targeted_cn = cn_in_idx_by_ascending_gradient[13:] # 297
                    # targeted_cn = cn_in_idx_by_ascending_gradient[14:] # 306
                    # targeted_cn = cn_in_idx_by_ascending_gradient[15:] # 249
                    # targeted_cn = cn_in_idx_by_ascending_gradient[16:] # 277
                    # targeted_cn = cn_in_idx_by_ascending_gradient[17:] # 269
                    # targeted_cn = cn_in_idx_by_ascending_gradient[18:] # 317
                    # targeted_cn = cn_in_idx_by_ascending_gradient[19:] # 239
                    # targeted_cn = cn_in_idx_by_ascending_gradient[20:] # 191
                    # targeted_cn = cn_in_idx_by_ascending_gradient[21:] # 263
                    # targeted_cn = cn_in_idx_by_ascending_gradient[22:] # 243
                    # targeted_cn = cn_in_idx_by_ascending_gradient[23:] # 265
                    # targeted_cn = cn_in_idx_by_ascending_gradient[24:] # 232
                    # targeted_cn = cn_in_idx_by_ascending_gradient[25:] # 176
                    # targeted_cn = cn_in_idx_by_ascending_gradient[26:] # 191
                    # targeted_cn = cn_in_idx_by_ascending_gradient[27:] # 229
                    # targeted_cn = cn_in_idx_by_ascending_gradient[28:] # 182
                    # targeted_cn = cn_in_idx_by_ascending_gradient[29:] # 182
                    # targeted_cn = cn_in_idx_by_ascending_gradient[30:] # 232
                    # targeted_cn = cn_in_idx_by_ascending_gradient[31:] # 206
                    # targeted_cn = cn_in_idx_by_ascending_gradient[32:] # 276
                    # targeted_cn = cn_in_idx_by_ascending_gradient[64:] # 51
                    # targeted_cn = cn_in_idx_by_ascending_gradient[96:] # 3
                    # targeted_cn = cn_in_idx_by_ascending_gradient[128:] # 0
                    ablate_cn_np[:,targeted_cn] = torch.nan
                    ablate_cn = torch.tensor(ablate_cn_np, dtype=torch.float, device='cuda').unsqueeze(0)
            
                    # neural_state_out_override[0][:, :, targeted_hn] = torch.tensor(scl_hc.mean_[targeted_hn], dtype=torch.float, device='cuda')

                # RANDOM
                else:

                    ablate_hn_out_np[np.arange(N_ROBOTS).reshape(-1, 1), row_indices[:, :DIM_A_LSTM_HX-self.ablations_hn_out]] = torch.nan
                    ablate_hn_out = torch.tensor(ablate_hn_out_np, dtype=torch.float, device='cuda').unsqueeze(0)

                    ablate_hn_in_np[np.arange(N_ROBOTS).reshape(-1, 1), row_indices[:, :DIM_A_LSTM_HX-self.ablations_hn_in]] = torch.nan
                    ablate_hn_in = torch.tensor(ablate_hn_in_np, dtype=torch.float, device='cuda').unsqueeze(0)

                    ablate_cn_np[np.arange(N_ROBOTS).reshape(-1, 1), row_indices[:, :DIM_A_LSTM_CX-self.ablations_cn]] = torch.nan
                    ablate_cn = torch.tensor(ablate_cn_np, dtype=torch.float, device='cuda').unsqueeze(0)

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
                    

                    neural_obs_override = None
                    neural_state_in_override = None
                    neural_state_out_override = None

                    if self.ablation_trial:
                        # ABLATION NEURONS AT SAME TIME AS DISTURBACE
                        if self.wait_until_disturbance:
                            ROBOT_ABLATION_IDX_FOR_MASK = self.env.perturb_started.unsqueeze(-1).repeat(1, 128).unsqueeze(0)
                        # ABLATION NEURONS FOR ENTIRITY OF TRIAL
                        else:
                            ROBOT_ABLATION_IDX_FOR_MASK = torch.ones((1,400,128), dtype=torch.bool)

                        ablate_hn_out_trial = ablate_hn_out.detach().clone()
                        ablate_hn_out_trial[~ROBOT_ABLATION_IDX_FOR_MASK] = torch.nan

                        ablate_hn_in_trial = ablate_hn_in.detach().clone()
                        ablate_hn_in_trial[~ROBOT_ABLATION_IDX_FOR_MASK] = torch.nan

                        ablate_cn_trial = ablate_cn.detach().clone()
                        ablate_cn_trial[~ROBOT_ABLATION_IDX_FOR_MASK] = torch.nan


                        ### NEW NEW NEW ###

                        neural_obs_override = torch.full((N_ROBOTS, DIM_OBS + DIM_ACT), torch.nan, device='cuda')
                        neural_state_in_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        neural_state_out_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        
                        # Now, B will contain updated values from A based on the conditions in C and D

                        neural_state_out_override[0][:, :, :] = ablate_hn_out_trial
                        neural_state_in_override[0][:, :, :] = ablate_hn_in_trial
                        neural_state_in_override[1][:, :, :] = ablate_cn_trial
                        
                        # OBS ABLATION
                        # neural_obs_override[ROBOT_ABLATION_IDX, 0] = 0 # u  317/400 = % (because timing does make a difference!)
                        # neural_obs_override[ROBOT_ABLATION_IDX, 1] = 0 # v  0/400 = 0%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 2] = 0 # w  292/400 = %
                        # neural_obs_override[ROBOT_ABLATION_IDX, 3] = 0 # p  37/400 = 9.25%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 4] = 0 # q  397/400 = 99.25%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 5] = 0 # r  396/400  = 99%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 6] = 0 # cos(pitch)  387/400 = 96.75%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 7] = 0 # cos(roll)  237 = 59.25%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 8] = 0 # cos(yaw)  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 9] = 0 # u*  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 10] = 0 # v*  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 11] = 0 # r*  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 12:24] = 0 # joint pos  0/400 = 0%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 24:36] = 0 # joint vel  315/400 = 78.75%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 136:176] = 0 #

                        ### CN ABLATION ###
                        
                        # CN ABLATE ALL
                        # neural_state_in_override[1][:, :, :] = torch.tensor(scl_hc.mean_[128:], dtype=torch.float, device='cuda')

                        # CN ABLATE RANDOM
                        # neural_state_in_override[1][:, :, :] = ablate_cn_trial
                        # neural_state_in_override[1][:, :, :] = ablate_cn_trial # 0 --> 400
                        # neural_state_in_override[1][:, :, :] = ablate_cn_trial # 32 --> 345
                        # neural_state_in_override[1][:, :, :] = ablate_cn_trial # 64 --> 179
                        # neural_state_in_override[1][:, :, :] = ablate_cn_trial # 96 --> 57
                        # neural_state_in_override[1][:, :, :] = ablate_cn_trial # 128 --> 8


                        # CN ABLATE TARGETED GRADIENT
                        
                        # cn_in_idx_by_ascending_gradient = [6, 18, 73, 94] #cn (from sampling)
                        # cn_in_idx_by_ascending_gradient = [108,6,101,2,30,47,13,42,89,90,118,98,85,124,99,68,32,24,10,72,3,61,19,109,21,31,22,103,56,69,114,78,46,81,35,97,84,110,4,104,14,50,63,49,79,77,93,106,44,62,70,73,91,41,11,102,95,120,23,88,26,5,18,80,48,20,125,117,112,52,65,36,37,64,127,96,126,55,59,115,33,58,83,123,71,1,27,100,39,94,8,9,53,116,86,113,67,34,51,76,28,40,25,16,60,15,87,54,105,122,92,0,119,45,75,74,107,29,7,111,66,121,17,82,57,12,38,43] #cn
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:1] # 400
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:2] # 400
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:3] # 398
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:4] # 391
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:5] # 392
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:6] # 388
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:7] # 371
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:8] # 364
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:9] # 233
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:10] # 234
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:11] # 295
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:12] # 299
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:13] # 297
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:14] # 306
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:15] # 249
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:16] # 277
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:17] # 269
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:18] # 317
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:19] # 239
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:20] # 191
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:21] # 263
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:22] # 243
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:23] # 265
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:24] # 232
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:25] # 176
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:26] # 191
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:27] # 229
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:28] # 182
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:29] # 182
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:30] # 232
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:31] # 206
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:32] # 276
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:64] # 51
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:96] # 3
                        # targeted_cn = cn_in_idx_by_ascending_gradient[:128] # 0

                        # neural_state_in_override[1][:, :, targeted_cn] = torch.tensor(scl_hc.mean_[[x+128 for x in targeted_cn]], dtype=torch.float, device='cuda')

                        ### HN ABLATION ###
                        
                        # HN ABLATE ALL
                        # neural_state_out_override[0][:, :, :] = torch.tensor(scl_hc.mean_[:128], dtype=torch.float, device='cuda')
                        # neural_state_in_override[0][:, :, :] = torch.tensor(scl_hc.mean_[:128], dtype=torch.float, device='cuda')

                        # HN OUT ABLATE RANDOM
                        # neural_state_out_override[0][:, :, :] = ablate_hn_trial
                        # neural_state_out_override[0][:, :, :] = ablate_hn_trial # 0 --> 400
                        # neural_state_out_override[0][:, :, :] = ablate_hn_trial # 32 --> 59
                        # neural_state_out_override[0][:, :, :] = ablate_hn_trial # 64 --> 0
                        # neural_state_out_override[0][:, :, :] = ablate_hn_trial # 96 --> 0
                        # neural_state_out_override[0][:, :, :] = ablate_hn_trial # 128 --> 0
                        
                        # HN IN ABLATE RANDOM
                        # neural_state_in_override[0][:, :, :] = ablate_hn_trial
                        # neural_state_in_override[0][:, :, :] = ablate_hn_trial # 0 --> 400
                        # neural_state_in_override[0][:, :, :] = ablate_hn_trial # 32 --> 394
                        # neural_state_in_override[0][:, :, :] = ablate_hn_trial # 64 --> 353
                        # neural_state_in_override[0][:, :, :] = ablate_hn_trial # 96 --> 283
                        # neural_state_in_override[0][:, :, :] = ablate_hn_trial # 128 --> 182

                        # HN IN ABLATE TARGETED GRADIENT
                        # hn_in_idx_by_ascending_gradient = [27,126,121,12,11,110,35,70,54,0,13,31,56,114,34,101,115,111,61,26,6,55,90,49,5,98,113,53,64,37,104,43,72,28,22,59,19,21,87,107,63,88,51,76,30,44,50,82,60,123,94,42,52,78,92,109,57,96,77,1,99,95,8,86,9,125,122,91,15,2,71,17,41,62,20,117,79,80,67,24,4,39,116,10,93,65,81,89,46,120,23,118,33,85,66,112,14,73,97,83,38,105,84,102,69,40,3,127,32,103,16,7,124,58,100,47,36,108,29,119,75,106,74,48,45,68,25,18] # hn
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:1] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:2] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:3] # 399
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:4] # 399
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:5] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:6] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:7] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:8] # 399
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:9] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:10] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:11] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:12] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:13] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:14] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:15] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:16] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:32] # 400
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:64] # 382
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:96] # 384
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:128] # 319
                        # neural_state_in_override[0][:, :, targeted_hn] = torch.tensor(scl_hc.mean_[targeted_hn], dtype=torch.float, device='cuda')

                        # HN OUT ABLATE TARGETED GRADIENT
                        
                        # hn_out_idx_by_ascending_gradient = [101,56,13,108,68,48,98,103,114,47,83,84,90,30,82,69,85,6,111,42,18,35,4,12,22,109,0,2,87,124,112,104,99,102,59,32,49,72,63,45,110,93,14,70,91,5,106,24,7,127,3,65,97,41,118,117,95,64,39,20,34,27,105,79,94,61,89,31,126,19,25,121,115,96,52,71,1,88,44,46,123,113,8,73,62,37,86,100,119,15,51,125,77,28,116,53,16,80,78,9,122,120,40,50,81,66,33,75,67,60,74,11,92,57,26,36,23,54,58,10,17,21,76,29,43,107,38,55] # hn out
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:1] # 395
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:2] # 397
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:3] # 242
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:4] # 168
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:5] # 270
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:6] # 241
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:7] # 148
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:8] # 188
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:9] # 117
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:10] # 23
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:11] # 46
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:12] # 1
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:13] # 0
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:14] # 0
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:15] # 0
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:16] # 0
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:32] # 0
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:64] # 0
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:96] # 0 
                        # targeted_hn = hn_out_idx_by_ascending_gradient[:128] # 0
                        # neural_state_out_override[0][:, :, targeted_hn] = torch.tensor(scl_hc.mean_[targeted_hn], dtype=torch.float, device='cuda')


                        # HN OUT ABLATE TARGETED
                        # hn_in_idx_by_ascending_gradient = [27, 126, 121, 12, 11, 110, 35, 70, 54, 0, 13, 31, 56, 114, 34, 101, 115, 111, 61, 26, 6, 55, 90, 49, 5, 98, 113, 53, 64, 37, 104, 43, 72, 28, 22, 59, 19] # hn
                        # targeted_hn = hn_in_idx_by_ascending_gradient[:32]
                        # neural_state_out_override[0][:, :, targeted_hn] = torch.tensor(scl_hc.mean_[targeted_hn], dtype=torch.float, device='cuda')

                        # HN OUT ABLATE TARGETED GRADIENT
                        # neuron_idx_by_ascending_gradient = [101, 56, 13, 108, 68, 48, 98, 103, 114, 47, 83, 84, 90]

                        ### REMOVE GROUPS OF NEURONS: TOP 1, 2, ... 13 NEURONS
                        # targeted_hn = neuron_idx_by_ascending_gradient[:1] # 397 400 398 392 398
                        # targeted_hn = neuron_idx_by_ascending_gradient[:2] # 393 387 390 393 392
                        # targeted_hn = neuron_idx_by_ascending_gradient[:3] # 285 251 275 274 280
                        # targeted_hn = neuron_idx_by_ascending_gradient[:4] # 196 187 194 195 214
                        # targeted_hn = neuron_idx_by_ascending_gradient[:5] # 267 262 287 280 258
                        # targeted_hn = neuron_idx_by_ascending_gradient[:6] # 239 238 256 230 235
                        # targeted_hn = neuron_idx_by_ascending_gradient[:7] # 160 164 162 156 167 155 155
                        # targeted_hn = neuron_idx_by_ascending_gradient[:8] # 194 197 164 199 199
                        # targeted_hn = neuron_idx_by_ascending_gradient[:9] # 120 80 111 106 100
                        # targeted_hn = neuron_idx_by_ascending_gradient[:10] # 34 28 32 29 26
                        # targeted_hn = neuron_idx_by_ascending_gradient[:11] # 50 36 47 46 48
                        # targeted_hn = neuron_idx_by_ascending_gradient[:12] # 6 3 0 2 1
                        # targeted_hn = neuron_idx_by_ascending_gradient[:13] # 0 0 0 0 1

                        # targeted_hn = [101, 56, 13, 108] # 196 194
                        # targeted_hn = [13, 56, 101, 108, 68] # 282 281 282 --> 68 is bad!!!
                        # targeted_hn = [13, 56, 101, 108, 48] # 177 160 161
                        # targeted_hn = [13, 56, 101, 108, 98] # 179 170 195
                        # targeted_hn = [13, 56, 101, 108, 68, 48] # 165
                        # targeted_hn = [13, 56, 101, 108, 68, 48, 98] # 171

                        ### REMOVE NEURON 68
                        # targeted_hn = [101, 56, 13, 108, 48] # 186 158 174 150 151
                        # targeted_hn = [101, 56, 13, 108, 48, 98] # 147 138 115 145 163
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 103] # 132 111 101 103 120
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 103, 114] # 102 98 103 97 82
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 103, 114, 47] # 27 25 21 33 34
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 103, 114, 47, 84] # 0 8 0 0 4
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 103, 114, 47, 84, 90] # 63 55 63 56 50 --> thought this would be zero. Some interactivity here, because ablating all [:13] yields 0/400 recovery.

                        ### REMOVE NEURON 103
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 114] # 76 77 97 85 100
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 114, 47] # 15 19 20 12 22
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 114, 47, 84] # 3 7 5 4 7
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 114, 47, 84, 90] # 84 78 64 78 97

                        # neural_state_out_override[0][:, :, targeted_hn] = torch.tensor(scl_hc.mean_[targeted_hn], dtype=torch.float, device='cuda')
                        # neural_state_in_override[0][:, :, targeted_hn] = torch.tensor(scl_hc.mean_[targeted_hn], dtype=torch.float, device='cuda')



















                        ### NEURAL OVERRIDE OBSERVATIONS ### (OLD)
                        # neural_obs_override = obses
                        # neural_obs_override[ROBOT_ABLATION_IDX, 0] = 0 # u  316/400 = 79% (because timing does make a difference!)
                        # neural_obs_override[ROBOT_ABLATION_IDX, 1] = 0 # v  0/400 = 0%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 2] = 0 # w  306/400 = 76.5%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 3] = 0 # p  37/400 = 9.25%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 4] = 0 # q  397/400 = 99.25%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 5] = 0 # r  396/400  = 99%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 6] = 0 # cos(pitch)  387/400 = 96.75%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 7] = 0 # cos(roll)  237 = 59.25%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 8] = 0 # cos(yaw)  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 9] = 0 # u*  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 10] = 0 # v*  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 11] = 0 # r*  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 12:24] = 0 # joint pos  0/400 = 0%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 24:36] = 0 # joint vel  315/400 = 78.75%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 136:176] = 0 # height  400/400 = 100%

                        ### NEURAL OVERRIDE OBSERVATIONS ### (NEW)
                        # neural_obs_override = torch.full((N_ROBOTS, DIM_OBS + DIM_ACT), torch.nan, device='cuda')
                        # neural_obs_override[ROBOT_ABLATION_IDX, 0] = 0 # u  317/400 = % (because timing does make a difference!)
                        # neural_obs_override[ROBOT_ABLATION_IDX, 1] = 0 # v  0/400 = 0%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 2] = 0 # w  292/400 = %

                        # neural_obs_override[ROBOT_ABLATION_IDX, 3] = 0 # p  37/400 = 9.25%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 4] = 0 # q  397/400 = 99.25%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 5] = 0 # r  396/400  = 99%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 6] = 0 # cos(pitch)  387/400 = 96.75%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 7] = 0 # cos(roll)  237 = 59.25%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 8] = 0 # cos(yaw)  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 9] = 0 # u*  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 10] = 0 # v*  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 11] = 0 # r*  400/400 = 100%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 12:24] = 0 # joint pos  0/400 = 0%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 24:36] = 0 # joint vel  315/400 = 78.75%
                        # neural_obs_override[ROBOT_ABLATION_IDX, 136:176] = 0 # height  400/400 = 100%
                        
                        ### NEURAL OVERRIDE STATES IN ###
                        ### Four neurons identified through SAMPLING-BASED METHOD, implicated in more than 40% of the failed trials

                        # # (160, 161, 172 / 400 = 41%) # WHEN ABLATED FOR ALL TIME (AGREES WITH FRONTIERS PAPER 37%)
                        # neural_state_in_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        # neural_state_in_override[1][:, :, 6] = scl_hc.mean_[128+6] # -0.286083278496328
                        # neural_state_in_override[1][:, :, 18] = scl_hc.mean_[128+18] # 4.6484050438199995
                        # neural_state_in_override[1][:, :, 73] = scl_hc.mean_[128+73] # -0.17231659909890198
                        # neural_state_in_override[1][:, :, 94] = scl_hc.mean_[128+94] # 0.420681332036574

                        # # (171, 159, 158, 155, 143 / 400 = 39%) WHEN ABLATED DURING AND AFTER PERTURBATION (NEW, SAME RESULTS AS FRONTIERS PAPER)
                        # neural_state_in_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        # neural_state_in_override[1][:, ROBOT_ABLATION_IDX, 6] = scl_hc.mean_[128+6] # -0.286083278496328
                        # neural_state_in_override[1][:, ROBOT_ABLATION_IDX, 18] = scl_hc.mean_[128+18] # 4.6484050438199995
                        # neural_state_in_override[1][:, ROBOT_ABLATION_IDX, 73] = scl_hc.mean_[128+73] # -0.17231659909890198
                        # neural_state_in_override[1][:, ROBOT_ABLATION_IDX, 94] = scl_hc.mean_[128+94] # 0.420681332036574

                        ## (150 / 400 = 38%)
                        # indices_cn = [6,18,73,94]
                        
                        # for idx in indices_cn:
                        #     self.states[1][0,:,idx] = scl_hc.mean_[128+idx] * torch.ones_like(self.states[1][0,:,idx].cpu())

                        # # Ablate ALL hn (329/400)
                        # self.states[0][0,:,:] = torch.from_numpy(scl_hc.mean_[:128]) * torch.ones_like(self.states[0][0,:,:].cpu()) # -3.5BW: 50%

                        # # Ablate ALL cn (3/400)
                        # self.states[1][0,:,:] = torch.from_numpy(scl_hc.mean_[128:]) * torch.ones_like(self.states[1][0,:,:].cpu()) # -3.5BW: 0%
                        
                        ### ABLATE ALL hn (317/400)
                        # neural_state_in_override[0][:, :, :] = torch.tensor(scl_hc.mean_[:128], dtype=torch.float, device='cuda')

                        ### ABLATE ALL cn (2/400)
                        # neural_state_in_override[1][:, :, :] = torch.tensor(scl_hc.mean_[128:], dtype=torch.float, device='cuda')

                        ### NEURAL OVERRIDE STATES OUT (SPECIFICALLY TARGETED) ###
                        ### Four neurons identified through GRADIENT-BASED METHOD [del(RHhip)/del(hc) * hc]_perturb - [del(RHhip)/del(hc) * hc)]_nom  < -0.08

                        # # WHEN ABLATED FOR ALL TIME (AGREES WITH FRONTIERS PAPER 42%)
                        # # (172, 166, 145, 145, 172, 178, 164, 138, 135 / 400 = 39%)
                        # neural_state_in_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        # neural_state_out_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        # neural_state_in_override_indices = [13, 56, 101, 108] # [0.205488821246418, 0.22776731317200002, 0.59731554871306, -0.199395715246838]
                        # targeted_hn = [13, 56, 101, 108] # [0.205488821246418, 0.22776731317200002, 0.59731554871306, -0.199395715246838]
                        # neural_state_in_override[0][:, :, neural_state_in_override_indices] = torch.tensor(scl_hc.mean_[neural_state_in_override_indices], dtype=torch.float, device='cuda')
                        # neural_state_out_override[0][:, :, targeted_hn] = torch.tensor(scl_hc.mean_[targeted_hn], dtype=torch.float, device='cuda')

                        # # WHEN ABLATED FOR ALL TIME (AGREES WITH FRONTIERS PAPER 42%)
                        # # (172, 166, 145, 145, 172, 178, 164, 138, 135 / 400 = 39%)
                        # neural_state_out_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        # neuron_idx_by_ascending_gradient = [101, 56, 13, 108, 68, 48, 98, 103, 114, 47, 83, 84, 90]
                        
                        ### REMOVE INDIVIDUAL NEURONS: TOP 1st, 2nd, ... 13th NEURONS
                        # targeted_hn = neuron_idx_by_ascending_gradient[0] # 399
                        # targeted_hn = neuron_idx_by_ascending_gradient[1] # 397
                        # targeted_hn = neuron_idx_by_ascending_gradient[2] # 389
                        # targeted_hn = neuron_idx_by_ascending_gradient[3] # 400
                        # targeted_hn = neuron_idx_by_ascending_gradient[4] # 397
                        # targeted_hn = neuron_idx_by_ascending_gradient[5] # 400
                        # targeted_hn = neuron_idx_by_ascending_gradient[6] # 399
                        # targeted_hn = neuron_idx_by_ascending_gradient[7] # 400
                        # targeted_hn = neuron_idx_by_ascending_gradient[8] # 400 
                        # targeted_hn = neuron_idx_by_ascending_gradient[9] # 397 399 397 398 397
                        # targeted_hn = neuron_idx_by_ascending_gradient[10] # 399
                        # targeted_hn = neuron_idx_by_ascending_gradient[11] # 398 399
                        # targeted_hn = neuron_idx_by_ascending_gradient[12] # 400

                        ### REMOVE GROUPS OF NEURONS: TOP 1, 2, ... 13 NEURONS
                        # targeted_hn = neuron_idx_by_ascending_gradient[:1] # 397 400 398 392 398
                        # targeted_hn = neuron_idx_by_ascending_gradient[:2] # 393 387 390 393 392
                        # targeted_hn = neuron_idx_by_ascending_gradient[:3] # 285 251 275 274 280
                        # targeted_hn = neuron_idx_by_ascending_gradient[:4] # 196 187 194 195 214
                        # targeted_hn = neuron_idx_by_ascending_gradient[:5] # 267 262 287 280 258
                        # targeted_hn = neuron_idx_by_ascending_gradient[:6] # 239 238 256 230 235
                        # targeted_hn = neuron_idx_by_ascending_gradient[:7] # 160 164 162 156 167 155 155
                        # targeted_hn = neuron_idx_by_ascending_gradient[:8] # 194 197 164 199 199
                        # targeted_hn = neuron_idx_by_ascending_gradient[:9] # 120 80 111 106 100
                        # targeted_hn = neuron_idx_by_ascending_gradient[:10] # 34 28 32 29 26
                        # targeted_hn = neuron_idx_by_ascending_gradient[:11] # 50 36 47 46 48
                        # targeted_hn = neuron_idx_by_ascending_gradient[:12] # 6 3 0 2 1
                        # targeted_hn = neuron_idx_by_ascending_gradient[:13] # 0 0 0 0 1

                        # targeted_hn = [101, 56, 13, 108] # 196 194
                        # targeted_hn = [13, 56, 101, 108, 68] # 282 281 282 --> 68 is bad!!!
                        # targeted_hn = [13, 56, 101, 108, 48] # 177 160 161
                        # targeted_hn = [13, 56, 101, 108, 98] # 179 170 195
                        # targeted_hn = [13, 56, 101, 108, 68, 48] # 165
                        # targeted_hn = [13, 56, 101, 108, 68, 48, 98] # 171

                        ### REMOVE NEURON 68
                        # targeted_hn = [101, 56, 13, 108, 48] # 186 158 174 150 151
                        # targeted_hn = [101, 56, 13, 108, 48, 98] # 147 138 115 145 163
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 103] # 132 111 101 103 120
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 103, 114] # 102 98 103 97 82
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 103, 114, 47] # 27 25 21 33 34
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 103, 114, 47, 84] # 0 8 0 0 4
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 103, 114, 47, 84, 90] # 63 55 63 56 50 --> thought this would be zero. Some interactivity here, because ablating all [:13] yields 0/400 recovery.

                        ### REMOVE NEURON 103
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 114] # 76 77 97 85 100
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 114, 47] # 15 19 20 12 22
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 114, 47, 84] # 3 7 5 4 7
                        # targeted_hn = [101, 56, 13, 108, 48, 98, 114, 47, 84, 90] # 84 78 64 78 97

                        # neural_state_out_override[0][:, :, :] = torch.tensor(scl_hc.mean_[:128], dtype=torch.float, device='cuda')
                        # neural_state_in_override[0][:, :, :] = torch.tensor(scl_hc.mean_[:128], dtype=torch.float, device='cuda')

                        ### ABLATE ALL
                        # neural_state_out_override[0][:, :, :] = torch.tensor(scl_hc.mean_[128:], dtype=torch.float, device='cuda')

                        # # WHEN ABLATED DURING AND AFTER PERTURBATION (NEW, DISAGREES WITH FRONTIERS PAPER) 
                        # # (381, 386, 391, 391, 389 / 400 = 97%)
                        # # NOT SURE WHY PERFORMANCE IS NOT DEGRADED??? GENE TO DO: LOOK AT HOLISTIC NEURAL STATE???
                        # neural_state_out_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        # targeted_hn = [13, 56, 101, 108] # [0.205488821246418, 0.22776731317200002, 0.59731554871306, -0.199395715246838]
                        # for idx in targeted_hn:
                        #     neural_state_out_override[0][:, ROBOT_ABLATION_IDX, idx] = scl_hc.mean_[idx]




                        ### NEURAL OVERRIDE STATES OUT (RANDOMLY TARGETED) ###
                        # # (303, 306,  / 400 = 75%) (FRONTIERS PAPER 85%)
                        # NOTE: SLOWS DOWN CODE A LOT BECAUSE TORCH.WHERE (CAN WE SPEED UP?)
                        # NOTE: INCLUDE +128 FOR CX
                        # NOTE: REVIEW CODE FOR BUGS
                        # neural_state_out_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), float('nan'), device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), float('nan'), device='cuda')]
                        # neural_state_out_override[0] = torch.where(mask, neural_state_out_override[0], mean_values_expanded)

                        # nantensor = torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), float('nan'), device='cuda')
                        # mask = torch.randint(0, 2, (1, N_ROBOTS, DIM_A_LSTM_HX), dtype=torch.bool, device='cuda')
                        # meanval = torch.tensor(scl_hc.mean_[:128], device='cuda', dtype=torch.float32)
                        # mean_values = meanval.unsqueeze(0).expand_as(mask)

                        # nantensor = torch.where(mask, nantensor, mean_values)



                        # neural_state_out_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        # for idx in range(N_hn):
                        #     neural_state_out_override[0][:, ROBOT_ABLATION_IDX, neural_state_out_override[0][:, idx]] = torch.tensor(scl_hc.mean_[neural_state_out_override[0][:, idx]], dtype=torch.float, device='cuda')
                        # for idx in range(N_hn):
                        #     neural_state_out_override[1][:, ROBOT_ABLATION_IDX, neural_state_out_override[1][:, idx]] = torch.tensor(scl_hc.mean_[neural_state_out_override[1][:, 128+idx]], dtype=torch.float, device='cuda')





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


                # if t > 250:

                #     if t > 250:
                        
                        ### Applies to all agents
                        # with torch.no_grad():

                            ### robot stops walking, falls over
                            # self.model.a2c_network.a_rnn.rnn.weight_ih_l0 *= 0
                            
                            ### same behavior, just slightly less robust to perturbations
                            # self.model.a2c_network.a_rnn.rnn.weight_hh_l0 *= 0

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

                        # Gather statistics on random ablations of N neurons
                        # if N_hn > 0:
                        #     for i in range(400):  # Assuming self.states.size(1) is 400
                        #         mean_tensor_hn = torch.tensor(scl_hc.mean_[      np.array(ablate_hn[i])], dtype=self.states[0].dtype).to(self.device)
                        #         self.states[0][0, i, ablate_hn[i]] = mean_tensor_hn * torch.ones_like(self.states[0][0, i, ablate_hn[i]])

                        #         if done[i]:
                        #             print("hn neurons ablated", ablate_hn[i])

                        # if N_cn > 0:
                        #     for i in range(400):  # Assuming self.states.size(1) is 400
                        #         mean_tensor_cn = torch.tensor(scl_hc.mean_[128 + np.array(static_indices_cn[i])], dtype=self.states[1].dtype).to(self.device)
                        #         self.states[1][0, i, static_indices_cn[i]] = mean_tensor_cn * torch.ones_like(self.states[1][0, i, static_indices_cn[i]])
                        
                        #         if done[i]:
                        #             print("cn neurons ablated", static_indices_cn[i])



                hc = torch.cat((self.states[0][0,:,:], self.states[1][0,:,:]), dim=1)
                hc_pc = pca_hc.transform(scl_hc.transform(hc.detach().cpu().numpy()))
                hc_pc_last = hc_pc

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
                
                plot = False

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
                        [hc_pc[ROBOT_ID_PLOT, 0], hc_pc_last[ROBOT_ID_PLOT, 0]], 
                        [hc_pc[ROBOT_ID_PLOT, 1], hc_pc_last[ROBOT_ID_PLOT, 1]],
                        [hc_pc[ROBOT_ID_PLOT, 2], hc_pc_last[ROBOT_ID_PLOT, 2]],
                        c='k')

                    # Update the marker position
                    marker.set_data([hc_pc[ROBOT_ID_PLOT, 0]], [hc_pc[ROBOT_ID_PLOT, 1]])
                    marker.set_3d_properties([hc_pc[ROBOT_ID_PLOT, 2]])
                    
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
                df = pd.DataFrame(data.detach().cpu().numpy(), columns=columns)
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
