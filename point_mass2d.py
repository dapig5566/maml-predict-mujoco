import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class PointMass2d(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30,
        'total_tasks': 4
    }

    def __init__(self):
        self.delta_t = 0.25
        self.max_position_x = np.inf
        self.min_position_x = -self.max_position_x

        self.max_position_y = np.inf
        self.min_position_y = -self.max_position_y

        self.max_speed = 0.2
        self.min_speed = -self.max_speed
        self.max_acc = 1.0
        self.min_acc = -self.max_acc
        self.radius = 2
        self.screen_width = 600
        self.screen_height = 600
        self.activity_area = 600
        self.shift = np.array([self.screen_width // 2, self.screen_width // 2], dtype=np.int)
        self.scale = self.activity_area / (2.5 * 2)
        self.tasks = [np.array([-2,-2]), np.array([2,-2]), np.array([-2,2]), np.array([2, 2])]

        self.viewer = None
        self.low_state = np.array([self.min_position_x, self.min_position_y])
        self.high_state = np.array([self.max_position_x, self.max_position_y])
        self.low_action = np.array([self.min_speed, self.min_speed])
        self.high_action = np.array([self.max_speed, self.max_speed])
        self.goal = None
        # self.done = None

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)

        self.seed()
        self.reset()

    def set_task(self, task_id):
        assert task_id in range(len(self.tasks)), "task id must be one of {}".format(list(range(len(self.tasks))))
        self.task_id = task_id
        self.goal = np.array(self.tasks[task_id])

    def reset_last(self):
        position = self._last_initial
        speed = np.array([0, 0], dtype=np.float32)
        self.state = np.concatenate([position, speed], axis=-1)
        return np.array(self.state)

    def get_task(self):
        return self.goal

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # self.task_id = task_id = np.random.choice(len(self.tasks))
        # self.done = False
        self._last_action = np.array([0, 0])
        position = np.random.uniform(-0.2, 0.2, [2])
        self._last_initial = np.array(position)
        speed = np.array([0, 0], dtype=np.float32)
        self.state = np.concatenate([position], axis=-1)
        return np.array(self.state)

    def step(self, action):
        assert self.goal is not None, "a goal must be set before take any step."

        # action = np.squeeze([action]).astype("float32")
        # position = self.state[:2]
        # # velocity = self.state[2:]
        # a_norm = np.sqrt((action**2).sum())
        # scale = min(a_norm, self.max_speed)
        # action = action / a_norm * scale
        # dist_old = np.sqrt(np.sum((position - self.goal)**2))
        # self._last_action = np.array(action)
        # # v_norm = np.sqrt((velocity**2).sum())
        # # if v_norm < self.max_speed:
        # #     position += velocity * self.delta_t + 0.5 * action * self.delta_t**2
        # # else:
        # #     position += velocity * self.delta_t
        # # velocity += action * self.delta_t
        # velocity = np.array(action)
        # position += velocity * self.delta_t
        #
        # # v_norm = np.sqrt((velocity ** 2).sum())
        # # scale = min(v_norm, self.max_speed)
        # # velocity = velocity / v_norm * scale
        #
        # min_pos = np.array([self.min_position_x, self.min_position_y], dtype=np.float32)
        # max_pos = np.array([self.max_position_x, self.max_position_y], dtype=np.float32)
        # if (position < min_pos).sum() + (position > max_pos).sum():
        #     velocity = np.array([0, 0], dtype=np.float32)
        #     position = np.clip(position, min_pos, max_pos)
        # dist = np.sqrt(np.sum((position - self.goal)**2))
        # reward = (dist_old - dist) if dist <= self.radius else 0.0
        # if not self.done:
        #     if dist <= 1:
        #         done = True
        #         self.done = True
        #     else:
        #         done = False
        # else:
        #     done = self.done
        # self.state = np.concatenate([position, velocity])
        prev_state = self.state
        self.state = prev_state + np.clip(action, -0.2, 0.2)
        reward = self.reward(prev_state, action, self.state)

        done = self.done(self.state)
        next_observation = np.copy(self.state)


        return np.array(self.state), reward, done, {}

    def done(self, obs):
        if obs.ndim == 1:
            return self.done(np.array([obs]))
        elif obs.ndim == 2:
            goal_distance = np.linalg.norm(obs - self.goal[None,:], axis=1)
            return (goal_distance < 1.0).any()#np.max(self._state) > 3

    def reward(self, obs, act, obs_next):
        if obs_next.ndim == 2:
            goal_distance = np.linalg.norm(obs_next - self.goal[None,:], axis=1)[0]

            dist_from_start = np.sqrt((obs_next**2).sum())#np.linalg.norm(obs_next, ord=1, axis=1)[0]
            pre_dist = np.sqrt((obs**2).sum())
            if dist_from_start <= 0.5:
                if pre_dist > 0.5:
                    cross_pt = self.cross(obs, obs_next, 0.5)
                    return np.linalg.norm(obs - self.goal[None,:], axis=1)[0] - np.linalg.norm(cross_pt - self.goal)
                return 0
            dists = [np.linalg.norm(obs_next - corner[None, :], axis=1) for corner in self.tasks]
            if np.min(goal_distance) == min(dists):
                if pre_dist <= 0.5:
                    cross_pt = self.cross(obs, obs_next, 0.5)
                    return np.linalg.norm(cross_pt - self.goal) - np.linalg.norm(obs_next - self.goal[None,:], axis=1)[0]
                return np.linalg.norm(obs - self.goal[None,:], axis=1)[0] - goal_distance
            return 0
        elif obs_next.ndim == 1:
            return self.reward(np.array([obs]), np.array([act]), np.array([obs_next]))

    def cross(self, x1, x2, r):
        x1 = np.squeeze(x1)
        x2 = np.squeeze(x2)
        dir = x2 - x1
        mat = np.array([[x1[0], dir[0]], [x1[1], dir[1]]])
        a = (mat[:, 1] ** 2).sum()
        b = 2 * (mat[:, 0] * mat[:, 1]).sum()
        c = (mat[:, 0] ** 2).sum() - r ** 2
        delta = np.sqrt(b ** 2 - 4 * a * c)
        root1 = (-b + delta) / (2 * a)
        root2 = (-b - delta) / (2 * a)
        beta = root1 if root1 > 0 else root2
        return x1 + beta * dir

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width, screen_height = self.screen_width, self.screen_height
        position = self.state
        # print(position)
        # velocity = self.state[2:]
        force = self._last_action
        color = np.array([205, 133, 0], dtype=np.float32) / 255
        bg = np.array([1., 1., 1.])
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, b, t, r = self.min_position_x, self.min_position_y, self.max_position_y, self.max_position_x
            points = self._to_screen_coord([(l,b), (l,t), (r,t), (r,b)])
            box = rendering.make_polygon(points, filled=False)
            self.viewer.add_geom(box)
            goals = np.array(self.tasks)

            for i, goal in enumerate(goals):
                cir = rendering.make_circle(self.radius*self.scale, filled=True)
                cir.add_attr(rendering.Transform(translation=self._to_screen_coord(goal)))
                alpha = 0.7
                cir.set_color(*((1-alpha)*color + alpha*bg))
                self.viewer.add_geom(cir)

        cir = rendering.make_circle(self.radius * self.scale, filled=True)
        cir.add_attr(rendering.Transform(translation=self._to_screen_coord(self.goal)))
        alpha = 0.5
        cir.set_color(*((1 - alpha) * color + alpha * bg))
        self.viewer.add_onetime(cir)
        cir_center = rendering.make_circle(2)
        cir_center.add_attr(rendering.Transform(translation=self._to_screen_coord(self.goal)))
        self.viewer.add_onetime(cir_center)

        cir = self.viewer.draw_circle(radius=5)
        trans = rendering.Transform(translation=self._to_screen_coord(position))
        cir.add_attr(trans)
        cir.set_color(1., 0., 0.)
        # velocity_dir = self._to_screen_coord(np.array([[0, 0], velocity])*0.1 + position)
        # line = self.viewer.draw_line(*velocity_dir)
        # line.set_color(0., 1., 0.)
        # force_dir = self._to_screen_coord(np.array([[0, 0], force]) + position)
        # line = self.viewer.draw_line(*force_dir)
        # line.set_color(0., 0., 1.)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _to_screen_coord(self, pos):
        shift, scale = self.shift, self.scale
        pos = np.array(pos)
        pos = (pos * scale).astype("int") + shift
        return pos

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class MetaPointEnvCorner(gym.Env):
    """
    Simple 2D point meta environment. Each meta-task corresponds to a different goal / corner
    (one of the 4 points (-2,-2), (-2, 2), (2, -2), (2,2)) which are sampled with equal probability
    """

    def __init__(self, reward_type='sparse', sparse_reward_radius=0.5):
        assert reward_type in ['dense', 'dense_squared', 'sparse']
        self.reward_type = reward_type
        # print("Point Env reward type is", reward_type)
        self.sparse_reward_radius = sparse_reward_radius
        self.corners = [np.array([-2,-2]), np.array([2,-2]), np.array([-2,2]), np.array([2, 2])]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(2,))
        self.is_done = None
        self.left_steps = None
        self.max_steps = 100

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.

        Args:
            action : an action provided by the environment
        Returns:
            (observation, reward, done, info)
            observation : agent's observation of the current environment
            reward [Float] : amount of reward due to the previous action
            done : a boolean, indicating whether the episode has ended
            info : a dictionary containing other diagnostic information from the previous action
        """
        assert not (np.isnan(action).any() and np.isinf(action).any()), "encountered nan in action."
        prev_state = self._state
        self._state = prev_state + np.clip(action / 5.0, -0.2, 0.2)
        reward = self.reward(prev_state, action, self._state)
        done = self.done(self._state)
        next_observation = np.copy(self._state)
        self.left_steps -= 1
        return np.concatenate([next_observation, np.array([self.left_steps / self.max_steps])], axis=-1), reward*50, self.left_steps <= 0, {}

    def set_state(self, state):
        self._state[:2] = np.array(state)


    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # self._state = np.random.uniform(-0.2, 0.2, size=(2,))
        self.left_steps = 100
        self._state = np.array([np.random.uniform(-0.05, 0.05), 0.0], dtype=np.float32)
        self.is_done = False
        observation = np.copy(self._state)
        return np.concatenate([observation, np.array([self.left_steps / self.max_steps])], axis=-1)

    def done(self, obs):
        if obs.ndim == 1:
            return self.done(np.array([obs]))
        elif obs.ndim == 2:
            if self.is_done:
                return self.is_done
            else:
                goal_distance = np.linalg.norm(obs - self.goal[None, :], axis=1)
                assert not np.isnan(obs).any(), "encountered nan in obs."
                outside = (obs < -2).any() or (obs > 2).any()
                if (goal_distance < 0.3).any() or outside:
                    self.is_done = True
                return self.is_done
            #return #np.max(self._state) > 3

    # def reward(self, obs, act, obs_next):
    #     if obs_next.ndim == 2:
    #         goal_distance = np.linalg.norm(obs_next - self.goal[None,:], axis=1)[0]
    #         if self.reward_type == 'dense':
    #             return - goal_distance
    #         elif self.reward_type == 'dense_squared':
    #             return - goal_distance**2
    #         elif self.reward_type == 'sparse':
    #             dist_from_start = np.linalg.norm(obs_next, ord=1, axis=1)[0]
    #             if dist_from_start < self.sparse_reward_radius:
    #                 return 0
    #             dists = [np.linalg.norm(obs_next - corner[None, :], axis=1) for corner in self.corners]
    #             if np.min(goal_distance) == min(dists):
    #                 return np.linalg.norm(obs - self.goal[None,:], axis=1)[0] - goal_distance
    #             return 0
    #             # return np.maximum(self.sparse_reward_radius - goal_distance, 0)
    #
    #     elif obs_next.ndim == 1:
    #         return self.reward(np.array([obs]), np.array([act]), np.array([obs_next]))
    #     else:
    #         raise NotImplementedError

    def cross(self, x1, x2, r):
        x1 = np.squeeze(x1)
        x2 = np.squeeze(x2)
        dir = x2 - x1
        mat = np.array([[x1[0], dir[0]], [x1[1], dir[1]]])
        a = (mat[:, 1] ** 2).sum()
        b = 2 * (mat[:, 0] * mat[:, 1]).sum()
        c = (mat[:, 0] ** 2).sum() - r ** 2
        delta = b ** 2 - 4 * a * c
        if delta < 0:
            return None, None
        root1 = (-b + np.sqrt(delta)) / (2 * a + 1e-8)
        root2 = (-b - np.sqrt(delta)) / (2 * a + 1e-8)
        # print(root1, root2)

        beta = [i for i in [root1, root2] if i >=0 and i <=1]
        if len(beta) == 0:
            return None, None
        else:
            beta = sorted(beta)[0]

        return x1 + beta * dir, beta

    def cross_xy(self, x1, x2):
        x1 = np.squeeze(x1)
        x2 = np.squeeze(x2)
        beta_y = -x1[0] / (x2[0] - x1[0] + 1e-8)
        beta_x = -x1[1] / (x2[1] - x1[1] + 1e-8)
        return beta_x, beta_y

    def in_goal_plane(self, x):
        dists = [np.linalg.norm(x - corner[None, :], axis=1) for corner in self.corners]
        return np.min(np.linalg.norm(x - self.goal[None, :], axis=1)[0]) == min(dists)

    def reward(self, obs, act, obs_next):
        if obs_next.ndim == 2:
            goal_distance = np.linalg.norm(obs_next - self.goal[None, :], axis=1)[0]
            dist_from_start = np.sqrt((obs_next ** 2).sum())
            pre_dist = np.sqrt((obs ** 2).sum())
            if self.in_goal_plane(obs):
                if self.in_goal_plane(obs_next):
                    cross_pt, b = self.cross(obs, obs_next, 0.5)
                    if cross_pt is None:
                        if dist_from_start <= 0.5:
                            return 0
                        else:
                            return np.linalg.norm(obs - self.goal[None, :], axis=1)[0] - goal_distance
                    else:
                        if dist_from_start <= 0.5:
                            return np.linalg.norm(obs - self.goal[None, :], axis=1)[0] - np.linalg.norm(
                                cross_pt - self.goal)
                        else:
                            return np.linalg.norm(cross_pt - self.goal) - \
                                goal_distance
                else:
                    if pre_dist <= 0.5:
                        return 0
                    else:
                        d = [self.cross(obs, obs_next, 0.5)[1], *self.cross_xy(obs, obs_next)]
                        d = np.array(sorted([i for i in d if i is not None]))
                        idx = np.argmax(np.squeeze(np.logical_and(d >= 0, d <= 1)))
                        beta = d[idx]
                        assert beta <= 1, "wrong dist."
                        cross_pt = obs + beta * (obs_next - obs)
                        return np.linalg.norm(obs - self.goal[None, :], axis=1)[0] - np.linalg.norm(
                            cross_pt - self.goal[None, :], axis=1)[0]
            else:
                if dist_from_start <= 0.5:
                    return 0
                else:
                    if self.in_goal_plane(obs_next):
                        d = [self.cross(obs, obs_next, 0.5)[1], *self.cross_xy(obs, obs_next)]
                        d = np.array(sorted([i for i in d if i is not None], reverse=True))
                        idx = np.argmax(np.logical_and(d >= 0, d <= 1))
                        beta = d[idx]
                        cross_pt = obs + beta * (obs_next - obs)
                        return np.linalg.norm(cross_pt - self.goal[None, :], axis=1)[0] - \
                               goal_distance
                    else:
                        return 0
        elif obs_next.ndim == 1:
            return self.reward(np.array([obs]), np.array([act]), np.array([obs_next]))

    def log_diagnostics(self, *args):
        pass

    def sample_tasks(self, n_tasks):
        return [self.corners[idx] for idx in np.random.choice(range(len(self.corners)), size=n_tasks)]

    def set_task(self, task_id):
        self.goal = np.array(self.corners[task_id])

    def get_task(self):
        return self.goal

def make_env():
    return gym.make("PointMass2d-v1")


def make_vec_env(num_envs):
    from subproc_vec_env import SubprocVecEnv
    # from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    def create_evn(seed):
        def _f():
            env = gym.make("PointMass2d-v1")
            # env.seed(seed)
            return env
        return _f

    envs = [create_evn(seed) for seed in range(num_envs)]
    envs = SubprocVecEnv(envs, init_different_envs=False)
    return envs


from gym.envs.registration import register
from gym import error

try:
    register(**{
        "id": "PointMass2d-v0",
        "entry_point": "point_mass2d:PointMass2d",
        "max_episode_steps": 1000
    })
    register(**{
        "id": "PointMass2d-v1",
        "entry_point": "point_mass2d:MetaPointEnvCorner",
        "max_episode_steps": 1000
    })
except error.Error:
    pass


# if __name__ == "__main__":
# #     import matplotlib.pyplot as plt
#
#
#     # def path_gen_fn(env, policy, start_reset=False, soft=True):
#     #     ac = yield
#     #     obs, dones = env.reset(), [False] * 1
#     #     ob_list = []
#     #     while True:
#     #         ob_list.clear()
#     #         # do NOT use this if environment is parallel env.
#     #         print(env.call_sync("get_task"))
#     #         for _ in range(2):
#     #             ob_list.append(obs)
#     #             obs, rewards, dones, info = env.step(ac)
#     #
#     #         timesteps = yield ob_list
#     #
#     # env = make_vec_env(2)
#     #
#     # path1 = path_gen_fn(env, None)
#     # next(path1)
#     # path2 = path_gen_fn(env, None)
#     # next(path2)
#     # env.call_sync("set_task", task_id=0)
#     # ob1 = path1.send([[1,0], [1, 0]])
#     # ob2 = path2.send([[-1, 0], [-1, 0]])
#     # print(ob1)
#     # print(ob2)
#     import time
#
#
#     def cross(x1, x2, r):
#         x1 = np.squeeze(x1)
#         x2 = np.squeeze(x2)
#         dir = x2 - x1
#         mat = np.array([[x1[0], dir[0]], [x1[1], dir[1]]])
#         a = (mat[:, 1] ** 2).sum()
#         b = 2 * (mat[:, 0] * mat[:, 1]).sum()
#         c = (mat[:, 0] ** 2).sum() - r ** 2
#         delta = np.sqrt(b ** 2 - 4 * a * c)
#         root1 = (-b + delta) / (2 * a)
#         root2 = (-b - delta) / (2 * a)
#         print(root1, root2)
#         if root1 >= 0 and root2 >= 0:
#             beta = min(root1, root2)
#         else:
#             beta = root1 if root1 > 0 else root2
#         beta_y = -x1[0] / (x2[0] - x1[0])
#         beta_x = -x1[1] / (x2[1] - x1[1])
#         print(beta_x, beta_y)
#         return x1 + beta * dir
#
#     def test():
#         return None, None
#
#     a = np.array([-0.53949838, -0.1555534])
#     b = np.array([-0.33949838,  0.04444661])
#     goal = np.array([2, 2])
#     # cross_pt = cross(a, b, 0.5)
#     # print(cross_pt)
#     # print(np.linalg.norm(a - goal) - np.linalg.norm(cross_pt - [2, 2]))
#     env = make_env()
#     ob = env.reset()
#     env.set_task(0)
#     goal = np.array([-2, -2])
#     rs = []
#     for i in range(1500):
#         # env.render()
#         # time.sleep(0.03)
#         action = env.action_space.sample()
#         ob_next, r, d, _ = env.step(action)
#         rs.append(r)
#         print("step {}".format(i), ob, action, r, d)
#         ob = ob_next
#         # assert i<3
#
#     print(np.sum(rs))
#     env.close()
#
#     # env = make_vec_env(2)
#     # ob = env.reset()
#     # print(ob)
#     # env.call_sync("set_task", task_id=0)
#     # ob, _, _, _ = env.step([[1, 0], [1, 0]])
#     # print(ob)
#     # ob = env.call_sync("reset_last")
#     # print(ob)
#     # env = make_env()
#     # ob = env.reset()
#     # env.set_task(0)
#     # for _ in range(100):
#     #     ob, r, d, _ = env.step([-1, 1])
#     #     print(ob, r, d)
