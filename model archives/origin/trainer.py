import logging
from dependencies import *
import numpy as np
import json
from gym.spaces import Discrete
from baseline import LinearFeatureBaseline
from maml import MAML, MAMLExploration
from loader import DataManager
# from point_mass2d import make_env, PointMass2d
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from tensorflow.python.platform import flags
from plot import plot_pre_trajs
from timed_wrapper import timed_wrpper
from env_wrappers import EnvOperations
FLAGS = flags.FLAGS


def train_few_shot_classification():
    def preprocess(train_input, train_label, test_input, test_label):
        # size = train.get_shape()[-1]
        # train = tf.reshape(train, [-1, size])
        # train = tf.random.shuffle(train)
        # train_input, train_label = tf.split(train, [dm.size**2, FLAGS.n_way], axis=-1)
        train_input = tf.reshape(train_input, [-1, dm.size, dm.size, dm.channels])
        train_label = tf.reshape(train_label, [-1, FLAGS.n_way])
        # test = tf.reshape(test, [-1, size])
        # test = tf.random.shuffle(test)
        # test_input, test_label = tf.split(test, [dm.size**2, FLAGS.n_way], axis=-1)
        test_input = tf.reshape(test_input, [-1, dm.size, dm.size, dm.channels])
        test_label = tf.reshape(test_label, [-1, FLAGS.n_way])
        return train_input, train_label, test_input, test_label

    def steup_dataset(image_size):
        train_input = tf.placeholder(tf.float32, [None, FLAGS.n_way, FLAGS.k_shot, image_size ** 2])
        train_label = tf.placeholder(tf.float32, [None, FLAGS.n_way, FLAGS.k_shot, FLAGS.n_way])
        test_input = tf.placeholder(tf.float32, [None, FLAGS.n_way, 1, image_size ** 2])
        test_label = tf.placeholder(tf.float32, [None, FLAGS.n_way, 1, FLAGS.n_way])
        dataset = tf.data.Dataset.from_tensor_slices((train_input, train_label, test_input, test_label))
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(FLAGS.meta_batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        return [train_input, train_label, test_input, test_label], next_batch, iterator.initializer

    preload_steps = 1000
    dm = DataManager("datasets/omniglot_resized", FLAGS.n_way, FLAGS.k_shot)
    data_ph, next_batch, dataset_init = steup_dataset(dm.size)
    model = MAML(next_batch, FLAGS.n_way, True)
    saver = tf.train.Saver()
    log_file = open("train_log.log", "w")
    ss = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for i in range(FLAGS.iterations):
        if i % preload_steps == 0:
            if i > 0:
                data = dm.sample_tasks(FLAGS.meta_batch_size, "validation")
                feed_dict = dict(zip(data_ph, data))
                dataset_init.run(feed_dict=feed_dict)
                pre_acc, post_acc = model.validation()
                info = "meta-val pre-update acc: {:.4f}, post-update acc: {:.4f}".format(pre_acc, post_acc)
                print(info)
                log_file.write(info + "\n")
                log_file.flush()
            data = dm.sample_tasks(FLAGS.meta_batch_size * preload_steps)
            feed_dict = dict(zip(data_ph, data))
            dataset_init.run(feed_dict=feed_dict)

        pre_acc, post_acc = model.train_step()

        if i % 10 == 0:
            info = "iteration [{}/{}] pre-update acc: {:4f}, post-update acc: {:4f}".format(i, FLAGS.iterations,
                                                                                            pre_acc, post_acc)
            print(info)
            log_file.write(info + "\n")

    log_file.flush()
    log_file.close()
    saver.save(ss, "maml.ckpt")
    print("model saved.")


def train_2d_exploration():
    # Is it possible to optimize wrt to all steps not only post update rewards?
    # ie. E_t[E_tau~pi_theta_t[R(tau)]]. This contains the first-order derivative like LVC estimator.
    # We can also use lvc estimator to enhance the performance.

    max_score = 0
    saved = False
    #############################################
    # env = make_vec_env(FLAGS.inner_batch_size)
    #############################################
    env = NormalizedBoxEnv(ENVS["ant-dir"](forward_backward=True))
    ob_dim = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, Discrete)
    ac_dim = env.action_space.shape[0] if not discrete else env.action_space.n

    agent = MAMLExploration(ob_dim, ac_dim, len(env.get_all_task_idx()), hidden_dim=256, discrete=discrete)
    baseline = LinearFeatureBaseline()
    saver = tf.train.Saver()
    tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # log params
    with open('logs/param.json', 'w') as fp:
        json.dump(FLAGS.flag_values_dict(), fp)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]:  %(message)s",
        handlers=[
            logging.FileHandler("logs/maml_exp.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("start logging.")
    # logging.info("collecting initial transitions...")
    # _, time_elapsed = timed_wrpper(collect_transition, env, agent)
    # logging.info("done. total time: {:.2f}".format(time_elapsed))
    agent.initialize()
    for iteration in range(FLAGS.iterations):
        logging.info(("=" * 12 + "iteration {}/{}" + "=" * 12).format(iteration, FLAGS.iterations - 1))
        logging.info("collecting samples to meta batch...")

        meta_batch, batch_returns, time_elapsed = timed_wrpper(EnvOperations.sample_meta_batch, env, agent, baseline)
        logging.info("done. total time: {:.2f}".format(time_elapsed))
        for returns, prefix in zip(batch_returns, ("QFNCAVG", "EVALAVG")):
            avg_returns = returns.mean(0)
            per_task_avg_return = returns.reshape(
                [-1, len(env.get_all_task_idx()), FLAGS.num_inner_loop + 1]).mean(0)
            data = [avg_returns, *per_task_avg_return]
            np.set_printoptions(precision=4, suppress=True)
            logging.info((prefix + " returns: {}, per task average return"+" {},"*len(env.get_all_task_idx())).format(*data))

        # if iteration % FLAGS.show_pre_update_every == 0 or iteration == FLAGS.iterations - 1:
        #     plot_pre_trajs(meta_batch[0][0], FLAGS.inner_time_steps // FLAGS.reset_times)

        avg_returns = batch_returns[0].mean(0)
        if avg_returns[1] > 60*4 and avg_returns[1] > max_score:
            saved = True
            max_score = avg_returns[1]
            agent.restore_weights()
            saver.save(tf.get_default_session(), "MAML_Exp.ckpt")
            logging.info("model saved when new high score {} occurred.".format(max_score))
            # plot_pre_trajs(meta_batch[0][0], FLAGS.inner_time_steps // FLAGS.reset_times)
        logging.info("start training...")

        avg_loss, ent, time_elapsed = timed_wrpper(agent.train_step, iteration, meta_batch,
                                             train_policy=iteration > FLAGS.train_value_iters)# train_policy=(iteration + 1) % FLAGS.train_value_iters == 0)
        logging.info("done. total time: {:.2f}".format(time_elapsed))
        # assert iteration < 2, "test done."
        logging.info("qf_loss: {:.4f}, kl_loss: {:.4f}, vf_loss: {:.4f}, n01_loss: {:.4f}".format(*avg_loss, ent))
        logging.info(("=" * 12 + "iteration {}/{}" + "=" * 12).format(iteration, FLAGS.iterations - 1))
    if not saved:
        saver.save(tf.get_default_session(), "MAML_Exp.ckpt")
        logging.info("saved last version")


def eval_maml_exp():
    import time
    env = NormalizedBoxEnv(ENVS["ant-dir"](forward_backward=True))
    num_tasks = len(env.get_all_task_idx())
    ob_dim = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, Discrete)
    ac_dim = env.action_space.shape[0] if not discrete else env.action_space.n
    agent = MAMLExploration(ob_dim, ac_dim, num_tasks, discrete=discrete, training=False)
    baseline = LinearFeatureBaseline()

    saver = tf.train.Saver()
    tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver.restore(tf.get_default_session(), "MAML_Exp.ckpt")

    for i in range(4):
        env.reset_task(idx=i%num_tasks)
        inner_batch, r, norm = EnvOperations.perform_inner_trajectories(env, agent, baseline, i % num_tasks, add_to_context=False, eval_q=True)
        print(r)
        np.save("ob_{}".format(i), inner_batch[0])
        time.sleep(3)
        # np.save("rw_{}".format(i), n)
