import numpy as np
import logging
from point_mass2d import PointMass2d
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


class EnvOperations:

    @classmethod
    def separate_parallel(cls, arr):
        assert arr.ndim >= 2, "rank must be more than 2."
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=2)
        axes = list(range(arr.ndim))
        axes[:2] = [1, 0]
        return np.squeeze(arr.transpose(axes))

    @classmethod
    def compute_adv(cls, observation, reward, gamma, baseline):
        def discount_cumsum(x, gamma):
            x = x * gamma
            return np.cumsum(x[:, ::-1], axis=-1)[:, ::-1] / gamma

        assert reward.ndim > 1, "reward must greater than 2 dims."
        gammas = np.array([gamma ** i for i in range(reward.shape[1])])
        target = discount_cumsum(reward, gammas)
        baseline.fit(observation, target)
        values = np.concatenate(
            [np.array([baseline.predict(ob) for ob in observation]), np.zeros([observation.shape[0], 1])],
            axis=-1)
        advantage = reward + gamma * values[:, 1:] - values[:, :-1]
        advantage = discount_cumsum(advantage, gammas)
        return advantage

    @classmethod
    def rollout(cls, env, agent, no_reset_when_done=False):

        ob_sub_traj = []
        ac_sub_traj = []
        rw_sub_traj = []
        no_sub_traj = []
        dn_sub_traj = []

        mean_sub_traj = []
        log_std_sub_traj = []

        for _ in range(FLAGS.inner_batch_size):
            done_count = 0
            ob = env.reset()
            data = []
            state = (np.zeros([1, FLAGS.hidden_dim]), np.zeros([1, FLAGS.hidden_dim]))
            ct = np.zeros([1, FLAGS.hidden_dim])
            for time_step in range(FLAGS.inner_time_steps):
                # env.render()
                ac, mean, log_std = agent.act(ob, ct)
                next_ob, rw, dn, _ = env.step(ac)
                if done_count >= 4:
                    dn = True
                data.append((ob, ac, rw, next_ob, dn, mean, log_std))
                ct, state = agent.next_context(state, (ob, ac, rw))
                if dn:
                    assert (time_step+1) % 100 == 0
                    ob = env.reset()
                    done_count += 1
                else:
                    ob = next_ob


            ob, ac, rw, next_ob, dn, mean, log_std = zip(*data)
            ob_sub_traj.append(np.array(ob))
            ac_sub_traj.append(np.array(ac))
            rw_sub_traj.append(np.array(rw))
            no_sub_traj.append(np.array(next_ob))
            dn_sub_traj.append(np.array(dn))
            mean_sub_traj.append(np.array(mean))
            log_std_sub_traj.append(np.array(log_std))

        # observation = cls.separate_parallel(np.array(ob_sub_traj))
        # action = cls.separate_parallel(np.array(ac_sub_traj))
        # reward = cls.separate_parallel(np.array(rw_sub_traj))
        # next_observation = cls.separate_parallel(np.array(no_sub_traj))
        # done = cls.separate_parallel(np.array(dn_sub_traj))
        #
        # mean = cls.separate_parallel(np.array(mean_sub_traj))
        # log_std = cls.separate_parallel(np.array(log_std_sub_traj))

        observation = np.array(ob_sub_traj)
        action = np.array(ac_sub_traj)
        reward = np.array(rw_sub_traj)
        next_observation = np.array(no_sub_traj)
        done = np.array(dn_sub_traj)
        mean = np.array(mean_sub_traj)
        log_std = np.array(log_std_sub_traj)

        returns = []
        # for i, j in zip(reward, done):
        #     # end = np.where(j)[0][0] if np.where(j)[0].size > 0 else j.shape[0]
        #     end = j.shape[0]-1
        #     returns.append(sum(i[:end + 1]))
        avg_return = reward.reshape([-1, 100]).sum(-1).mean()#sum(returns) / len(returns)

        sub_traj = (
            observation,
            action,
            reward,
            next_observation,
            done,

            mean,
            log_std,
            avg_return,
        )

        return sub_traj





    @classmethod
    def perform_inner_trajectories(cls, env, agent, baseline, task_id, add_to_context=True, eval_q=False):
        agent.restore_weights()
        ob_inner_batch = []
        ac_inner_batch = []
        ad_inner_batch = []
        rw_inner_batch = []
        dn_inner_batch = []
        mean_inner_batch = []
        log_std_inner_batch = []
        avg_returns = []
        norm = np.nan
        for inner_step in range(FLAGS.num_inner_loop + 1):
            observation, action, reward, next_observation, done, mean, log_std, avg_return = cls.rollout(env, agent)
            if inner_step == 0:
                agent.replay_buffer.push_back(task_id, (observation, action, reward, next_observation, done))
                # advantage = np.array(agent.compute_adv(task_id, observation, action, reward))
                advantage = np.array(np.zeros_like(reward))
                agent.update_weights((observation, action, reward, advantage, done, mean, log_std), inner_step)
            else:
                advantage = cls.compute_adv(observation.reshape([-1, FLAGS.inner_time_steps // 4, env.observation_space.shape[0]]), reward.reshape([-1, FLAGS.inner_time_steps // 4]), FLAGS.gamma, baseline).reshape([FLAGS.inner_batch_size, FLAGS.inner_time_steps])
            ob_inner_batch.append(observation)
            ac_inner_batch.append(action)
            ad_inner_batch.append(advantage)
            rw_inner_batch.append(reward)
            dn_inner_batch.append(done)
            mean_inner_batch.append(mean)
            log_std_inner_batch.append(log_std)
            avg_returns.append(avg_return)

        inner_batch = (
            np.array(ob_inner_batch),
            np.array(ac_inner_batch),
            np.array(ad_inner_batch),
            np.array(rw_inner_batch),
            np.array(dn_inner_batch),
            np.array(mean_inner_batch),
            np.array(log_std_inner_batch),
        )

        return inner_batch, np.array(avg_returns), norm

    @classmethod
    def sample_meta_batch(cls, env, agent, baseline, eval_q=False):
        task_ids = list(env.get_all_task_idx())
        agent.save_weights()
        ob_meta_batch = []
        ac_meta_batch = []
        ad_meta_batch = []
        rw_meta_batch = []
        dn_meta_batch = []
        mean_meta_batch = []
        log_std_meta_batch = []
        meta_batch_returns = []
        eval_batch_returns = []

        for i in range(FLAGS.meta_batch_size):
            add2ctx = not i >= FLAGS.meta_batch_size - FLAGS.extra_rl_batch_size
            task_id = task_ids[i % len(task_ids)]
            # env.call_sync("reset_task", idx=task_id)
            # env.set_task(task_id=task_id)
            env.reset_task(idx=task_id)

            (ob_inner_batch,
             ac_inner_batch,
             ad_inner_batch,
             rw_inner_batch,
             dn_inner_batch,
             mean_inner_batch,
             log_std_inner_batch), returns, norm = cls.perform_inner_trajectories(env, agent, baseline, task_id, add_to_context=add2ctx)
            if eval_q:
                _, lfb_returns, _ = cls.perform_inner_trajectories(env, agent, baseline, task_id, add_to_context=False, eval_q=eval_q)
                eval_batch_returns.append(lfb_returns)

            ob_meta_batch.append(ob_inner_batch)
            ac_meta_batch.append(ac_inner_batch)
            ad_meta_batch.append(ad_inner_batch)
            rw_meta_batch.append(rw_inner_batch)
            dn_meta_batch.append(dn_inner_batch)
            mean_meta_batch.append(mean_inner_batch)
            log_std_meta_batch.append(log_std_inner_batch)
            if i < FLAGS.meta_batch_size - FLAGS.extra_rl_batch_size:
                meta_batch_returns.append(returns)
            np.set_printoptions(precision=4)
            logging.info("  task [{}/{}] id [{}] average return each update: {}, "
                         "min/max mean {:.4f}/{:.4f}, min/max log_std {:.4f}/{:.4f} "
                         .format(i,
                                 FLAGS.meta_batch_size - 1,
                                 task_id, returns,
                                 np.min(mean_inner_batch),
                                 np.max(mean_inner_batch),
                                 np.min(log_std_inner_batch),
                                 np.max(log_std_inner_batch)))
            if eval_q:
                logging.info("  EVAL [{}/{}] id [{}] average return each update: {}, "
                             .format(i,
                                     FLAGS.meta_batch_size - 1,
                                     task_id, lfb_returns))
        meta_batch = (
            np.array(ob_meta_batch),
            np.array(ac_meta_batch),
            np.array(ad_meta_batch),
            np.array(rw_meta_batch),
            np.array(dn_meta_batch),
            np.array(mean_meta_batch),
            np.array(log_std_meta_batch)
        )
        batch_returns = (np.array(meta_batch_returns),)
        if eval_q:
            batch_returns += (np.array(eval_batch_returns),)
        return meta_batch, batch_returns