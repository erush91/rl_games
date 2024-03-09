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

import pickle as pk
import pandas as pd

import sklearn.preprocessing

# Import the required library
import matplotlib.pyplot as plt

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
        self.export_data = self.env.cfg['env']['output'].get('export_data', False)
        self.export_data_path = self.env.cfg['env']['output'].get('export_data_path', '')        
        self.export_data_actor = self.env.cfg['env']['output'].get('export_data_actor', False)
        self.export_data_critic = self.env.cfg['env']['output'].get('export_data_critic', False)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        if self.env.cfg['name'] == 'AnymalTerrain' or self.env.cfg['name'] == 'A1Terrain'or self.env.cfg['name'] == 'CassieTerrain':
            self.max_steps = 501 # 3001 # 1501 # 10001 # 1001 # 108000 // 4
        if self.env.cfg['name'] == 'ShadowHand':
            self.max_steps = 501

        self.device = torch.device(self.device_name)

        self.ablation_trial = self.env.cfg['env']['ablate'].get('ablation_trial', False)
        self.targeted_ablation_trial = self.env.cfg['env']['ablate'].get('targeted_ablation_trial', False)
        self.wait_until_disturbance = self.env.cfg['env']['ablate'].get('wait_until_disturbance', False)
        self.ablation_scl_pca_path = self.env.cfg['env']['ablate'].get('ablation_scl_pca_path', '')
        self.ablations_obs_in = self.env.cfg['env']['ablate'].get('ablations_obs_in', 0)
        self.ablations_hn_out = self.env.cfg['env']['ablate'].get('ablations_hn_out', 0)
        self.ablations_hn_in = self.env.cfg['env']['ablate'].get('ablations_hn_in', 0)
        self.ablations_cn_in = self.env.cfg['env']['ablate'].get('ablations_cn_in', 0)

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
            
        tensor_specs = OrderedDict()
        tensor_specs['ENV'] = 1
        tensor_specs['TIME'] = 1
        tensor_specs['DONE'] = 1
        tensor_specs['REWARD'] = 1
        tensor_specs['ACT'] = DIM_ACT
        tensor_specs['OBS'] = DIM_OBS
        if rnn_type == 'lstm':
            if self.export_data_actor:
                tensor_specs['A_MLP_XX'] = DIM_A_MLP_XX
                tensor_specs['A_LSTM_HC'] = DIM_A_LSTM_HC
                tensor_specs['A_LSTM_CX'] = DIM_A_LSTM_CX
                tensor_specs['A_LSTM_C1X'] = DIM_A_LSTM_CX
                tensor_specs['A_LSTM_C2X'] = DIM_A_LSTM_CX
                tensor_specs['A_LSTM_HX'] = DIM_A_LSTM_HX
            if self.export_data_critic:
                tensor_specs['C_MLP_XX'] = DIM_C_MLP_XX
                tensor_specs['C_LSTM_HC'] = DIM_C_LSTM_HC
                tensor_specs['C_LSTM_CX'] = DIM_C_LSTM_CX
                tensor_specs['C_LSTM_C1X'] = DIM_C_LSTM_CX
                tensor_specs['C_LSTM_C2X'] = DIM_C_LSTM_CX
                tensor_specs['C_LSTM_HX'] = DIM_C_LSTM_HX
        if rnn_type == 'gru':
            if self.export_actor:
                tensor_specs['A_GRU_HX'] = DIM_A_GRU_HX
            if self.export_critic:
                tensor_specs['C_GRU_HX'] = DIM_C_GRU_HX

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

        if self.config['name'] == "CassieTerrain":
            tensor_dict['ACT']['cols'] = pd.read_csv('/home/gene/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/legend/act_cassieterrain.csv', header=None).values[:,0]
            tensor_dict['OBS']['cols'] = pd.read_csv('/home/gene/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/legend/obs_cassieterrain.csv', header=None).values[:,0]

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

            # load scaler and pca transforms
            scl_hc = pk.load(open(self.ablation_scl_pca_path + 'A_LSTM_HC_SCL.pkl','rb'))
            pca_hc = pk.load(open(self.ablation_scl_pca_path + 'A_LSTM_HC_PCA.pkl','rb'))

            scl_obs = pk.load(open(self.ablation_scl_pca_path + 'OBS_SCL.pkl','rb'))
            pca_obs = pk.load(open(self.ablation_scl_pca_path + 'OBS_PCA.pkl','rb'))

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
            N_ROBOTS = self.env.num_envs

            # Generate random row indices for each row
            row_indices = np.tile(np.arange(DIM_A_LSTM_HX), (N_ROBOTS, 1))
            row_indices2 = np.tile(np.arange(DIM_OBS + DIM_ACT), (N_ROBOTS, 1))

            # Shuffle row_indices randomly for each row
            np.apply_along_axis(np.random.shuffle, 1, row_indices)

            # Initialize obs with mean neural values
            ablate_obs_in_np = np.tile(scl_obs.mean_, (N_ROBOTS,1))
            zeros_array = np.zeros((ablate_obs_in_np.shape[0], DIM_ACT)) # zeros for actions in obs state
            ablate_obs_in_np = np.hstack((ablate_obs_in_np, zeros_array))

            # Initialize recurrent neurons with mean neural values
            ablate_hn_out_np = np.tile(scl_hc.mean_[:128], (N_ROBOTS,1))
            ablate_hn_in_np = np.tile(scl_hc.mean_[:128], (N_ROBOTS,1))
            ablate_cn_in_np = np.tile(scl_hc.mean_[128:], (N_ROBOTS,1))

            robots_recovered = torch.ones((N_ROBOTS), dtype=torch.bool)

            if self.ablation_trial:

                # TARGETED
                if self.targeted_ablation_trial:

                    # NO OBS ABLATION
                    ablate_obs_in_np[:,:] = torch.nan

                    # OBS ABLATION
                    # ablate_obs_in[:, 0] = 0 # u  317/400 = % (because timing does make a difference!)
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 1] = 0 # v  0/400 = 0%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 2] = 0 # w  292/400 = %
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 3] = 0 # p  37/400 = 9.25%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 4] = 0 # q  397/400 = 99.25%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 5] = 0 # r  396/400  = 99%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 6] = 0 # cos(pitch)  387/400 = 96.75%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 7] = 0 # cos(roll)  237 = 59.25%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 8] = 0 # cos(yaw)  400/400 = 100%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 9] = 0 # u*  400/400 = 100%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 10] = 0 # v*  400/400 = 100%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 11] = 0 # r*  400/400 = 100%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 12:24] = 0 # joint pos  0/400 = 0%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 24:36] = 0 # joint vel  315/400 = 78.75%
                    # neural_obs_override[ROBOT_ABLATION_IDX_FOR_MASK2, 136:176] = 0 #

                    # ### Ablate u # 310
                    # ablate_obs_in_np[:,0] = 0
                    # ablate_obs_in_np[:,1:] = torch.nan

                    # ### Ablate v # 0
                    # ablate_obs_in_np[:,:1] = torch.nan
                    # ablate_obs_in_np[:,1] = 0 # 65
                    # ablate_obs_in_np[:,2:] = torch.nan

                    # ### Ablate w # 300
                    # ablate_obs_in_np[:,:2] = torch.nan
                    # ablate_obs_in_np[:,2] = 0
                    # ablate_obs_in_np[:,3:] = torch.nan

                    # ### Ablate p # 48
                    # ablate_obs_in_np[:,:3] = torch.nan
                    # ablate_obs_in_np[:,3] = 0
                    # ablate_obs_in_np[:,4:] = torch.nan

                    # ### Ablate q # 395
                    # ablate_obs_in_np[:,:4] = torch.nan
                    # ablate_obs_in_np[:,4] = 0
                    # ablate_obs_in_np[:,5:] = torch.nan

                    # ### Ablate r # 399
                    # ablate_obs_in_np[:,:5] = torch.nan
                    # ablate_obs_in_np[:,5] = 0
                    # ablate_obs_in_np[:,6:] = torch.nan

                    # ### Ablate cos(theta) # 390
                    # ablate_obs_in_np[:,:6] = torch.nan
                    # ablate_obs_in_np[:,6] = 0
                    # ablate_obs_in_np[:,7:] = torch.nan

                    # ### Ablate cos(phi) # 221
                    # ablate_obs_in_np[:,:7] = torch.nan
                    # ablate_obs_in_np[:,7] = 0
                    # ablate_obs_in_np[:,8:] = torch.nan

                    # ### Ablate -sqrt(1 - cos^2(phi) - cos^2(theta)) # 400
                    # ablate_obs_in_np[:,:8] = torch.nan
                    # ablate_obs_in_np[:,8] = 0
                    # ablate_obs_in_np[:,9:] = torch.nan

                    # ### Ablate u* # 400
                    # ablate_obs_in_np[:,:9] = torch.nan
                    # ablate_obs_in_np[:,9] = 0
                    # ablate_obs_in_np[:,10:] = torch.nan

                    # ### Ablate v* # 399
                    # ablate_obs_in_np[:,:10] = torch.nan
                    # ablate_obs_in_np[:,10] = 0
                    # ablate_obs_in_np[:,11:] = torch.nan

                    # ### Ablate r* # 400
                    # ablate_obs_in_np[:,:11] = torch.nan
                    # ablate_obs_in_np[:,11] = 0
                    # ablate_obs_in_np[:,12:] = torch.nan

                    # ### Ablate joint pos # 0
                    # ablate_obs_in_np[:,:12] = torch.nan
                    # ablate_obs_in_np[:,24:] = torch.nan

                    # ### Ablate joint vel # 317
                    # ablate_obs_in_np[:,:24] = torch.nan
                    # ablate_obs_in_np[:,36:] = torch.nan

                    # ### Ablate depths # 238
                    # ablate_obs_in_np[:,:36] = torch.nan


                    ablate_obs_in = torch.tensor(ablate_obs_in_np, dtype=torch.float, device='cuda')

                    hn_out_idx_by_ascending_gradient = [101,56,13,108,68,48,98,103,114,47,83,84,90,30,82,69,85,6,111,42,18,35,4,12,22,109,0,2,87,124,112,104,99,102,59,32,49,72,63,45,110,93,14,70,91,5,106,24,7,127,3,65,97,41,118,117,95,64,39,20,34,27,105,79,94,61,89,31,126,19,25,121,115,96,52,71,1,88,44,46,123,113,8,73,62,37,86,100,119,15,51,125,77,28,116,53,16,80,78,9,122,120,40,50,81,66,33,75,67,60,74,11,92,57,26,36,23,54,58,10,17,21,76,29,43,107,38,55] # hn out
                    hn_out_idx_ablated = hn_out_idx_by_ascending_gradient[:self.ablations_hn_out]
                    hn_out_idx_not_ablated = hn_out_idx_by_ascending_gradient[self.ablations_hn_out:]
                    ablate_hn_out_np[:,hn_out_idx_not_ablated] = torch.nan
                    ablate_hn_out = torch.tensor(ablate_hn_out_np, dtype=torch.float, device='cuda').unsqueeze(0)

                    hn_in_idx_by_ascending_gradient = [54,119,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,120,121,122,123,124,125,126,127] #hn in (from sampling, NOT GRADIENT)
                    # hn_in_idx_by_ascending_gradient = [27,126,121,12,11,110,35,70,54,0,13,31,56,114,34,101,115,111,61,26,6,55,90,49,5,98,113,53,64,37,104,43,72,28,22,59,19,21,87,107,63,88,51,76,30,44,50,82,60,123,94,42,52,78,92,109,57,96,77,1,99,95,8,86,9,125,122,91,15,2,71,17,41,62,20,117,79,80,67,24,4,39,116,10,93,65,81,89,46,120,23,118,33,85,66,112,14,73,97,83,38,105,84,102,69,40,3,127,32,103,16,7,124,58,100,47,36,108,29,119,75,106,74,48,45,68,25,18] # hn in
                    hn_in_idx_ablated = hn_in_idx_by_ascending_gradient[:self.ablations_hn_in]
                    hn_in_idx_not_ablated = hn_in_idx_by_ascending_gradient[self.ablations_hn_in:]
                    ablate_hn_in_np[:,hn_in_idx_not_ablated] = torch.nan
                    ablate_hn_in = torch.tensor(ablate_hn_in_np, dtype=torch.float, device='cuda').unsqueeze(0)

                    cn_in_idx_by_ascending_gradient = [6,13,18,54,60,73,94,0,1,2,3,4,5,7,8,9,10,11,12,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,56,57,58,59,61,62,63,64,65,66,67,68,69,70,71,72,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127] #cn (from sampling, NOT GRADIENT)
                    # cn_in_idx_by_ascending_gradient = [108,6,101,2,30,47,13,42,89,90,118,98,85,124,99,68,32,24,10,72,3,61,19,109,21,31,22,103,56,69,114,78,46,81,35,97,84,110,4,104,14,50,63,49,79,77,93,106,44,62,70,73,91,41,11,102,95,120,23,88,26,5,18,80,48,20,125,117,112,52,65,36,37,64,127,96,126,55,59,115,33,58,83,123,71,1,27,100,39,94,8,9,53,116,86,113,67,34,51,76,28,40,25,16,60,15,87,54,105,122,92,0,119,45,75,74,107,29,7,111,66,121,17,82,57,12,38,43] #cn
                    cn_in_idx_ablated = cn_in_idx_by_ascending_gradient[:self.ablations_cn_in]
                    cn_in_idx_not_ablated = cn_in_idx_by_ascending_gradient[self.ablations_cn_in:]
                    ablate_cn_in_np[:,cn_in_idx_not_ablated] = torch.nan
                    ablate_cn_in = torch.tensor(ablate_cn_in_np, dtype=torch.float, device='cuda').unsqueeze(0)
            
                # RANDOM
                else:

                    ablate_obs_in_np[np.arange(N_ROBOTS).reshape(-1, 1), row_indices2[:, :DIM_OBS+DIM_ACT-self.ablations_obs_in]] = torch.nan
                    ablate_obs_in = torch.tensor(ablate_obs_in_np, dtype=torch.float, device='cuda')

                    ablate_hn_out_np[np.arange(N_ROBOTS).reshape(-1, 1), row_indices[:, :DIM_A_LSTM_HX-self.ablations_hn_out]] = torch.nan
                    ablate_hn_out = torch.tensor(ablate_hn_out_np, dtype=torch.float, device='cuda').unsqueeze(0)

                    ablate_hn_in_np[np.arange(N_ROBOTS).reshape(-1, 1), row_indices[:, :DIM_A_LSTM_HX-self.ablations_hn_in]] = torch.nan
                    ablate_hn_in = torch.tensor(ablate_hn_in_np, dtype=torch.float, device='cuda').unsqueeze(0)

                    ablate_cn_in_np[np.arange(N_ROBOTS).reshape(-1, 1), row_indices[:, :DIM_A_LSTM_CX-self.ablations_cn_in]] = torch.nan
                    ablate_cn_in = torch.tensor(ablate_cn_in_np, dtype=torch.float, device='cuda').unsqueeze(0)

            neural_override = {}
            neural_override['obs'] = None
            neural_override['rnn_states_in'] = None
            neural_override['rnn_states_out'] = None

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
                    
                    if self.ablation_trial:

                        # ABLATION NEURONS AT SAME TIME AS DISTURBANCE
                        if self.wait_until_disturbance:
                            ROBOT_ABLATION_IDX_FOR_MASK2 = self.env.perturb_started.unsqueeze(-1).repeat(1, 188)
                            ROBOT_ABLATION_IDX_FOR_MASK = self.env.perturb_started.unsqueeze(-1).repeat(1, 128).unsqueeze(0)
                        # ABLATION NEURONS FOR ENTIRITY OF TRIAL
                        else:
                            ROBOT_ABLATION_IDX_FOR_MASK2 = torch.ones((N_ROBOTS,188), dtype=torch.bool)
                            ROBOT_ABLATION_IDX_FOR_MASK = torch.ones((1,N_ROBOTS,128), dtype=torch.bool)

                        ablate_obs_in_trial = ablate_obs_in.detach().clone()
                        ablate_obs_in_trial[~ROBOT_ABLATION_IDX_FOR_MASK2] = torch.nan

                        ablate_hn_out_trial = ablate_hn_out.detach().clone()
                        ablate_hn_out_trial[~ROBOT_ABLATION_IDX_FOR_MASK] = torch.nan

                        ablate_hn_in_trial = ablate_hn_in.detach().clone()
                        ablate_hn_in_trial[~ROBOT_ABLATION_IDX_FOR_MASK] = torch.nan

                        ablate_cn_trial = ablate_cn_in.detach().clone()
                        ablate_cn_trial[~ROBOT_ABLATION_IDX_FOR_MASK] = torch.nan

                        neural_override['obs'] = torch.full((N_ROBOTS, DIM_OBS + DIM_ACT), torch.nan, device='cuda')
                        neural_override['rnn_states_in'] = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        neural_override['rnn_states_out'] = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        
                        neural_obs_override = ablate_obs_in_trial
                        neural_override['rnn_states_out'][0][:, :, :] = ablate_hn_out_trial
                        neural_override['rnn_states_in'][0][:, :, :] = ablate_hn_in_trial
                        neural_override['rnn_states_in'][1][:, :, :] = ablate_cn_trial
                        
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
                    

                        # # WHEN ABLATED DURING AND AFTER PERTURBATION (NEW, DISAGREES WITH FRONTIERS PAPER) 
                        # # (381, 386, 391, 391, 389 / 400 = 97%)
                        # # NOT SURE WHY PERFORMANCE IS NOT DEGRADED??? GENE TO DO: LOOK AT HOLISTIC NEURAL STATE???
                        # neural_state_out_override = [torch.full((1, N_ROBOTS, DIM_A_LSTM_HX), torch.nan, device='cuda'), torch.full((1, N_ROBOTS, DIM_A_LSTM_CX), torch.nan, device='cuda')]
                        # targeted_hn = [13, 56, 101, 108] # [0.205488821246418, 0.22776731317200002, 0.59731554871306, -0.199395715246838]
                        # for idx in targeted_hn:
                        #     neural_state_out_override[0][:, ROBOT_ABLATION_IDX, idx] = scl_hc.mean_[idx]

                    action = self.get_action(obses, is_deterministic, neural_override) # neural_obs_override,neural_state_override
                    
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
                        if self.export_data_actor:
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
                        if self.export_data_critic:
                            tensor_dict['C_LSTM_C1X']['data'][t,:,:] = c1
                            tensor_dict['C_LSTM_C2X']['data'][t,:,:] = c2
                        
                obses, r, done, info = self.env_step(self.env, action)

                # if t == 499:
                #     print(robots_recovered)
                #     neural_state_in_override[1]
                #     pd.DataFrame(robots_recovered.cpu().numpy()).to_csv('recoveries.csv')
                #     pd.DataFrame(neural_state_in_override[0].squeeze().cpu().numpy()).to_csv(str('hn_in_ablations.csv'))
                #     pd.DataFrame(neural_state_in_override[1].squeeze().cpu().numpy()).to_csv(str('cn_in_ablations.csv'))
                #     pd.DataFrame(neural_state_out_override[0].squeeze().cpu().numpy()).to_csv(str('hn_out_ablations.csv'))


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

                if self.states != None:
                    hc = torch.cat((self.states[0][0,:,:], self.states[1][0,:,:]), dim=1)
                    hc_pc = pca_hc.transform(scl_hc.transform(hc.detach().cpu().numpy()))
                    hc_pc_last = hc_pc

                if self.export_data:

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

                    if self.export_data_actor:
                        tensor_dict['A_MLP_XX']['data'][t,:,:] = self.layers_out['actor_mlp'] # torch.squeeze(self.states[2][0,:,:]) # lstm hn (short-term memory)

                    if self.export_data_critic:
                        tensor_dict['C_MLP_XX']['data'][t,:,:] = self.layers_out['critic_mlp'] # lstm cn (long-term memory)

                    if rnn_type == 'lstm':
                        if self.export_data_actor:
                            tensor_dict['A_LSTM_HX']['data'][t,:,:] = torch.squeeze(self.states[0][0,:,:]) # lstm hn (short-term memory)
                            tensor_dict['A_LSTM_CX']['data'][t,:,:] = torch.squeeze(self.states[1][0,:,:]) # lstm cn (long-term memory)
                            tensor_dict['A_LSTM_HC']['data'][t,:,:] = torch.cat((tensor_dict['A_LSTM_HX']['data'][t,:,:], tensor_dict['A_LSTM_CX']['data'][t,:,:]), dim=1)

                        if self.export_data_critic:
                            tensor_dict['C_LSTM_HX']['data'][t,:,:] = torch.squeeze(self.states[2][0,:,:]) # lstm hn (short-term memory)
                            tensor_dict['C_LSTM_CX']['data'][t,:,:] = torch.squeeze(self.states[3][0,:,:]) # lstm cn (long-term memory)
                            tensor_dict['C_LSTM_HC']['data'][t,:,:] = torch.cat((tensor_dict['C_LSTM_HX']['data'][t,:,:], tensor_dict['C_LSTM_CX']['data'][t,:,:]), dim=1)

                    elif rnn_type == 'gru':
                        if self.export_data_actor:
                            tensor_dict['A_GRU_HX']['data'][t,:,:] = torch.squeeze(self.states[0][0,:,:])

                        if self.export_data_critic:
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
                robots_recovered[done_indices] = 0
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

            # # create specific folder
            p =  self.export_data_path
            p = Path().resolve() / p
            p.mkdir(exist_ok=True)

            # Save the dataframe to a CSV file
            all_data.to_parquet(str(p / 'RAW_DATA.parquet'))
            # all_data.to_csv(str(p / 'RAW_DATA.csv'))

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
