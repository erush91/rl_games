import copy
import time

import gym
import numpy as np
import torch

from rl_games.algos_torch import model_builder
from rl_games.common import env_configurations, vecenv


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
        self.player_config = self.config.get('player', {})
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.has_central_value = self.config.get(
            'central_value_config') is not None
        self.device_name = self.config.get('device_name', 'cuda')
        self.render_env = self.player_config.get('render', False)
        self.games_num = self.player_config.get('games_num', 2000)
        if 'deterministic' in self.player_config:
            self.is_deterministic = self.player_config['deterministic']
        else:
            self.is_deterministic = self.player_config.get(
                'deterministic', True)
        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.export_data = self.player_config.get('export_data', True)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        self.max_steps = 108000 // 4
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

        obs_dim = self.observation_space.shape[-1] - self.action_space.shape[-1]
        act_dim = self.action_space.shape[-1]
        observations = torch.zeros(self.max_steps, obs_dim, dtype=torch.float32).to(self.device) # act are at end of obs
        actions = torch.zeros(self.max_steps, act_dim, dtype=torch.float32).to(self.device)
        rewards = torch.zeros(self.max_steps, 1, dtype=torch.float32).to(self.device)
        dones = torch.zeros(self.max_steps, 1, dtype=torch.float32).to(self.device)
        arnn_hn = torch.zeros(self.max_steps, self.model.get_default_rnn_state()[0].size(dim=2), dtype=torch.float32).to(self.device)
        arnn_cn = torch.zeros(self.max_steps, self.model.get_default_rnn_state()[1].size(dim=2), dtype=torch.float32).to(self.device)
        crnn_hn = torch.zeros(self.max_steps, self.model.get_default_rnn_state()[2].size(dim=2), dtype=torch.float32).to(self.device)
        crnn_cn = torch.zeros(self.max_steps, self.model.get_default_rnn_state()[3].size(dim=2), dtype=torch.float32).to(self.device)

        agent_id = 2 # 44 # 0 # 1 # 2 # 17F # 48B # 12L # 44R # 13BL

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

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

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)

                if self.export_data:
                    observations[n,:] = obses[agent_id,:obs_dim]
                    actions[n,:] = obses[agent_id,obs_dim:]
                    rewards[n,:] = r[agent_id]
                    dones[n,:] = done[agent_id]
                    arnn_hn[n,:] = torch.squeeze(self.states[0][0,agent_id,:])
                    arnn_cn[n,:] = torch.squeeze(self.states[1][0,agent_id,:])
                    crnn_hn[n,:] = torch.squeeze(self.states[2][0,agent_id,:])
                    crnn_cn[n,:] = torch.squeeze(self.states[3][0,agent_id,:])

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
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

        if self.export_data:
            import pandas as pd
            obs_np = observations.cpu().numpy()
            obs_df = pd.DataFrame(obs_np)
            obs_df = obs_df.loc[~(obs_df==0).all(axis=1)]
            obs_df.to_csv('obs.csv', index=False)
            act_np = actions.cpu().numpy()
            act_df = pd.DataFrame(act_np)
            act_df = act_df.loc[~(act_df==0).all(axis=1)]
            act_df.to_csv('act.csv', index=False)
            rew_np = rewards.cpu().numpy()
            rew_df = pd.DataFrame(rew_np)
            rew_df = rew_df.loc[~(rew_df==0).all(axis=1)]
            rew_df.to_csv('rew.csv', index=False)
            dne_np = dones.cpu().numpy()
            dne_df = pd.DataFrame(dne_np)
            dne_df = dne_df.loc[~(dne_df==0).all(axis=1)]
            dne_df.to_csv('dne.csv', index=False)
            acx_np = arnn_cn.cpu().numpy()
            acx_df = pd.DataFrame(acx_np)
            acx_df = acx_df.loc[~(acx_df==0).all(axis=1)]
            acx_df.to_csv('acx.csv', index=False)
            ahx_np = arnn_hn.cpu().numpy()
            ahx_df = pd.DataFrame(ahx_np)
            ahx_df = ahx_df.loc[~(ahx_df==0).all(axis=1)]
            ahx_df.to_csv('ahx.csv', index=False)
            ccx_np = crnn_cn.cpu().numpy()
            ccx_df = pd.DataFrame(ccx_np)
            ccx_df = ccx_df.loc[~(ccx_df==0).all(axis=1)]
            ccx_df.to_csv('ccx.csv', index=False)
            chx_np = crnn_hn.cpu().numpy()
            chx_df = pd.DataFrame(chx_np)
            chx_df = chx_df.loc[~(chx_df==0).all(axis=1)]
            chx_df.to_csv('chx.csv', index=False)
            arnn_states = torch.cat((arnn_cn, arnn_hn), 1)
            crnn_states = torch.cat((crnn_cn, crnn_hn), 1)
            # arnn = self.model.a2c_network.rnn.rnn
            # arnn = self.model.a2c_network.a_rnn.rnn
            # crnn = self.model.a2c_network.c_rnn.rnn
            # # Instantiate the constant input to the RNNs
            # constant_inputs = torch.zeros(self.env.max_episode_length, arnn.input_size, dtype=torch.float32).to(self.device)
            # self.model.train()

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
