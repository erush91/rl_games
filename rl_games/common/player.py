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
        if self.env.cfg['name'] == 'AnymalTerrain':
            self.max_steps = 1000 # 3001 # 1501 # 10001 # 1001 # 108000 // 4
        if self.env.cfg['name'] == 'ShadowHand':
            self.max_steps = 1001

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

    def get_action(self, obs, is_deterministic=False):
        raise NotImplementedError('step')

    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size(
            )[2]), dtype=torch.float32).to(self.device) for s in rnn_states]

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

        torch.zeros(self.max_steps, self.env.num_environments, 1, dtype=torch.float32).to(self.device)

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

        if self.env.cfg['name'] == 'AnymalTerrain':
            new_tensor_specs = OrderedDict()
            for key, value in tensor_specs.items():
                if key == 'ACT':  # Add 'FT_FORCE' after 'ACT'
                    new_tensor_specs['FT_FORCE'] = 4
                new_tensor_specs[key] = value
            tensor_specs = new_tensor_specs

        N_STEPS = self.max_steps
        N_ENVS = self.env.num_environments
        
        def create_tensor_dict(tensor_specs):
            tensor_dict = OrderedDict()
            for key, dim in tensor_specs.items():
                if dim ==1:
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

        if self.config['name'] == "AnymalTerrain":
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

            # load scaler and pca transforms
            scl_hx = pk.load(open(DATA_PATH + 'A_LSTM_HX_SPEED_SCL.pkl','rb'))
            pca_hx = pk.load(open(DATA_PATH + 'A_LSTM_HX_SPEED_PCA.pkl','rb'))
            scl_cx = pk.load(open(DATA_PATH + 'A_LSTM_CX_SPEED_SCL.pkl','rb'))
            pca_cx = pk.load(open(DATA_PATH + 'A_LSTM_CX_SPEED_PCA.pkl','rb'))

            # Import the required library
            import matplotlib.pyplot as plt

            # Create the figure before the loop
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')

            # Initialize the marker outside the loop
            marker, = ax1.plot([0], [0], [0], 'ro')  # Use 'ro' for red circles


            for t in range(self.max_steps + 1):
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


                    # hx = self.states[0][0,:,:]
                    # cx = self.states[1][0,:,:]
                    # # cx[:,128:] += -1e6 # * cx_pc[:,:256]
                    # # hc_last_pc = pca.transform(scl.transform(torch.squeeze(tensor_dict['A_LSTM_HC']['data'][t-2,:,:]).detach().cpu().numpy()))
                    # hx_pc = pca_hx.transform(scl_hx.transform(torch.squeeze(hx).detach().cpu().numpy()))
                    # cx_pc = pca_cx.transform(scl_cx.transform(torch.squeeze(cx).detach().cpu().numpy()))

                    # ROBOT_ID = 0
                    # # neural perturbations
                    # perturb = True
                    # if perturb and t >= 100:
                    #     # hx_pc[ROBOT_ID,:] = 0 # * cx_pc[:,:256]
                    #     # cx_pc[ROBOT_ID,:] = 0 # * cx_pc[:,:256]
                    #     hx = torch.tensor(scl_hx.inverse_transform(pca_hx.inverse_transform(hx_pc)), dtype=torch.float32).unsqueeze(dim=0)
                    #     cx = torch.tensor(scl_cx.inverse_transform(pca_cx.inverse_transform(cx_pc)), dtype=torch.float32).unsqueeze(dim=0)

                    #     self.states[0][0,ROBOT_ID,:] = hx[:,ROBOT_ID,:]
                    #     self.states[1][0,ROBOT_ID,:] = cx[:,ROBOT_ID,:]
                    #     # self.states[1][0,ROBOT_ID,:] = cx[:,ROBOT_ID,DIM_A_LSTM_HX:]
                        
                    action = self.get_action(obses, is_deterministic)
                    
                    # hx = self.states[0][0,:,:]
                    # cx = self.states[1][0,:,:]
                    # # cx[:,128:] += -1e6 # * cx_pc[:,:256]
                    # # hc_last_pc = pca.transform(scl.transform(torch.squeeze(tensor_dict['A_LSTM_HC']['data'][t-2,:,:]).detach().cpu().numpy()))
                    # hx_pc = pca_hx.transform(scl_hx.transform(torch.squeeze(hx).detach().cpu().numpy()))
                    # cx_pc = pca_cx.transform(scl_cx.transform(torch.squeeze(cx).detach().cpu().numpy()))


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
                        
                        # check to see my computations are correct (norm < 1e-7)
                        # h_error = torch.norm(torch.squeeze(self.states[0][0,0,:])-h)
                        # c_error = torch.norm(torch.squeeze(self.states[1][0,0,:])-c)

                obses, r, done, info = self.env_step(self.env, action)


                # for key, value in self.layers_out.items():
                #     if type(value) is tuple:
                #         for valuee in value:
                #             if type(valuee) is tuple:
                #                 for valueee in valuee:
                #                     print(key, valueee.size())
                #             else:
                #                 print(key, valuee.size())
                #     else:
                #         print(key, value.size())



                # print(self.states[0][0,0,:], self.states[1][0,0,:], self.states[2][0,0,:], self.states[3][0,0,:])
                # self.states[0][0,:,:] += 10 * torch.rand_like(self.states[0][0,:,:]) # [actor lstm hn (short-term memory)
                # self.states[1][0,:,:] += 10 * torch.rand_like(self.states[1][0,:,:]) # [actor lstm cn (long-term memory)
                # self.states[2][0,:,:] += 10 * torch.rand_like(self.states[2][0,:,:]) # [critic lstm hn (short-term memory)
                # self.states[3][0,:,:] += 10 * torch.rand_like(self.states[3][0,:,:]) # [critic lstm cn (long-term memory)

                if self.export_data:                  

                    condition = torch.arange(self.env.num_environments)
                    # condition = torch.arange(self.env.num_environments / 5).repeat(5)
                    time = torch.Tensor([t * self.env.dt]).repeat(self.env.num_environments)

                    tensor_dict['ENV']['data'][t,:,:] = torch.unsqueeze(condition[:], dim=1)
                    tensor_dict['TIME']['data'][t,:,:] = torch.unsqueeze(time[:], dim=1)
                    if self.env.cfg['name'] == 'AnymalTerrain':
                        tensor_dict['FT_FORCE']['data'][t,:,:] = info['info']
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

                    # if t > 0:
                    #     # Update plot
                        
                    #     # Plot the line of the last agent
                    #     ax1.plot(
                    #         [hx_pc[ROBOT_ID, 0], hx_pc_last[ROBOT_ID,0]], 
                    #         [hx_pc[ROBOT_ID, 1], hx_pc_last[ROBOT_ID,1]],
                    #         [hx_pc[ROBOT_ID, 2], hx_pc_last[ROBOT_ID,2]],
                    #         c='k')

                    #     # Update the marker position
                    #     marker.set_data([hx_pc[ROBOT_ID, 0]], [hx_pc[ROBOT_ID, 1]])
                    #     marker.set_3d_properties([hx_pc[ROBOT_ID, 2]])
                        
                    #     # Set the title of the plot
                    #     ax1.set_title(f'Timestep: {t}')
                    #     plt.draw()
                    #     plt.pause(0.01)

                    # hx_pc_last = hx_pc
                    # cx_pc_last = cx_pc

                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

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
        tf = 1000 # 3000 # 1500 # 1500 # 10000 # 1000 # 600 # 527

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
            date_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

            if self.env.cfg['name'] == 'AnymalTerrain':
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
            
            # create folder
            p = date_str + exp_str
            p = Path().resolve() / 'data' / p
            p.mkdir(exist_ok=True)

            # Save the dataframe to a CSV file
            all_data.to_parquet(str(p / 'RAW_DATA.parquet'))

            # export the model used for data collection
            with open(p.joinpath('model.txt'), 'w') as file:
                file.write(self.config['name'])

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
