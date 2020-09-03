from dependencies import *
import numpy as np
import tensorflow_probability as tfp
import logging
import progressbar
# from progressbar import progressbar
from tensorflow.python.platform import flags
from models import Classifier, Policy, ContextualStateImportance, Qfunction, Vfunction, ContextEncoder
from utils import lvc_estimator, ppo_estimator, squash_backward, vpg_estimator, reverse_action, reparameterization
from experience_replay import MultitaskReplayBuffer, MultitaskTrajectoryBuffer
from timed_wrapper import timed_wrpper

FLAGS = flags.FLAGS


class MAML:
    def __init__(self, input_data, n_way, training=None):
        self.train_input, self.train_label, self.test_input, self.test_label = input_data
        self.n_way = n_way
        self.classifier = Classifier(n_way, 128)
        self.train_op = self.build_train_op(training=training)

    def build_train_op(self, training=None):

        def update_weights(loss, weights):
            names = [i for i in weights]
            w = [weights[i] for i in names]
            grads = tf.gradients(loss, w)
            w = {n: v - FLAGS.inner_lr * g for n, v, g in zip(names, w, grads) if g is not None}
            return w

        def take_inner_steps(inp):
            train_input, train_label, test_input, test_label = inp
            weights = self.classifier.weights
            for i in range(FLAGS.num_inner_loop):
                x = self.classifier.build_convnet(train_input, weights=weights, training=training)
                pre_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_label, logits=x))
                weights = update_weights(pre_loss, weights)
                if i == 0:
                    pre_acc = tf.reduce_mean(
                        tf.to_float(tf.equal(tf.argmax(x, axis=-1), tf.argmax(train_label, axis=-1))))

            x = self.classifier.build_convnet(test_input, weights=weights, training=training)
            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                post_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=test_label, logits=x))
            post_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=-1), tf.argmax(test_label, axis=-1))))
            return pre_loss, post_loss, pre_acc, post_acc

        elems = (self.train_input, self.train_label, self.test_input, self.test_label)
        result = tf.map_fn(take_inner_steps, elems=elems, parallel_iterations=FLAGS.meta_batch_size)
        pre_loss, post_loss, pre_acc, post_acc = result
        self.pre_acc = tf.reduce_mean(pre_acc)
        self.post_acc = tf.reduce_mean(post_acc)
        post_loss = tf.reduce_mean(post_loss)
        train_op = tf.train.AdamOptimizer(FLAGS.meta_lr).minimize(post_loss)
        return train_op

    def train_step(self):
        session = tf.get_default_session()
        _, pre_acc, post_acc = session.run([self.train_op, self.pre_acc, self.post_acc])
        return pre_acc, post_acc

    def validation(self):
        session = tf.get_default_session()
        pre_acc, post_acc = session.run([self.pre_acc, self.post_acc])
        return pre_acc, post_acc


class MAMLExploration:
    def __init__(self, ob_dim, action_dim, num_tasks, hidden_dim=64, discrete=True, training=True):
        self.num_tasks = num_tasks
        self.adaptive_kl_coeff_ph = tf.placeholder(tf.float32)
        self.beta = tf.placeholder(tf.float32)
        self.global_step_ph = tf.placeholder(tf.float32)
        self.decay_step = tf.placeholder(tf.float32)  # useless
        self.embedding = tf.placeholder(tf.float32)

        self.kl_coeff = FLAGS.init_kl_coeff * np.ones([FLAGS.num_inner_loop], dtype=np.float32)
        self.traj_ent_coeff = FLAGS.traj_ent_coeff
        self.global_step = 0
        self.tau = 0.005
        self.training = training
        self.ob_dim = ob_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.obs, self.actions, self.advs, self.rews, self.dns, self.means, self.log_stds, self.init_h, self.init_c, self.context, self.transitions = self.create_phs()
        self.discrete = discrete
        self.func_groups = [self.create_func_group(i) for i in range(2)]
        self.replay_buffer = MultitaskTrajectoryBuffer(self.num_tasks, max_buffer_size=10000, ob_dim=self.ob_dim, action_dim=self.action_dim, context_dim=FLAGS.hidden_dim)

        if FLAGS.use_attention:
            self.csi = ContextualStateImportance()

        self.build_train_op()

    def initialize(self):
        session = tf.get_default_session()
        session.run(self.initialize_target)
        session.run(self.sync_func)

    def create_func_group(self, device_id):
        with tf.device("/GPU:{}".format(device_id)):
            policy = Policy(self.ob_dim, self.action_dim, self.hidden_dim, device_id=device_id)
            qfunc1 = Qfunction(256, "q_func1_GPU_{}".format(device_id))
            qfunc2 = Qfunction(256, "q_func2_GPU_{}".format(device_id))
            vfunc = Vfunction(256, "v_func_GPU_{}".format(device_id))
            target_vfunc = Vfunction(256, "target_v_func_GPU_{}".format(device_id))
            context_encoder = ContextEncoder(FLAGS.hidden_dim, 128, name="rnn_ctx_enc_GPU_{}".format(device_id))
        return (policy, qfunc1, qfunc2, vfunc, target_vfunc, context_encoder)

    def create_phs(self):
        obs = tf.placeholder(tf.float32)
        actions = tf.placeholder(tf.float32)
        advs = tf.placeholder(tf.float32)
        rews = tf.placeholder(tf.float32)
        dns = tf.placeholder(tf.float32)
        means = tf.placeholder(tf.float32)
        log_stds = tf.placeholder(tf.float32)
        init_h = tf.placeholder(tf.float32, [None, FLAGS.hidden_dim])
        init_c = tf.placeholder(tf.float32, [None, FLAGS.hidden_dim])
        context = tf.placeholder(tf.float32)
        transitions = tf.placeholder(tf.float32)
        return obs, actions, advs, rews, dns, means, log_stds, init_h, init_c, context, transitions

    def build_train_op(self):

        def assign_vars(vars1, vars2):
            vars = vars1 if len(vars1) < len(vars2) else vars2
            names = [n for n in vars]
            update_op = tf.group(*[vars1[n].assign(vars2[n]) for n in names])
            return update_op

        def sample_from(x, num_samples=None, discrete=None):
            assert discrete is not None, "must specify discrete or not."
            if discrete:
                distribution = tf.distributions.Categorical(logits=x)
            else:
                x, scale = x
                distribution = tfp.distributions.MultivariateNormalDiag(loc=x, scale_diag=tf.exp(scale))

            if num_samples is None:
                return distribution
            else:
                samples = tf.squeeze(tf.tanh(distribution.sample(num_samples)))
                return samples, distribution

        def update_weights(loss, weights, lr):

            names = [n for n in weights]
            w = [weights[n] for n in names]
            grads = tf.gradients(loss, w)
            if FLAGS.clip_norm:
                grads, _ = tf.clip_by_global_norm(grads, FLAGS.clip_norm)
            w = {n: v - lr * g for n, v, g in zip(names, w, grads) if g is not None}

            return w

        def overwrite_vars(dst_scope, src_scope, operator=None):

            if operator is None:
                operator = lambda x, y: y
            assert hasattr(operator, "__call__"), "operator must be callable."

            s1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, dst_scope)

            s2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, src_scope)

            op = tf.group(*[v1.assign(operator(v1, v2)) for v1, v2 in zip(sorted(s1, key=lambda v: v.name),
                                                                          sorted(s2, key=lambda v: v.name))])
            return op

        def split_inner_step_data(inp):
            obs, acts, advs, rews, dns, means, log_stds, context, transitions = inp
            obs = [tf.squeeze(tensor) for tensor in tf.split(obs, FLAGS.num_inner_loop + 1)]
            actions = [tf.squeeze(tensor) for tensor in tf.split(acts, FLAGS.num_inner_loop + 1)]
            advantages = [tf.squeeze(tensor) for tensor in tf.split(advs, FLAGS.num_inner_loop + 1)]
            rewards = [tf.squeeze(tensor) for tensor in tf.split(rews, FLAGS.num_inner_loop + 1)]
            dones = [tf.squeeze(tensor) for tensor in tf.split(dns, FLAGS.num_inner_loop + 1)]
            means = [tf.squeeze(tensor) for tensor in tf.split(means, FLAGS.num_inner_loop + 1)]
            log_stds = [tf.squeeze(tensor) for tensor in tf.split(log_stds, FLAGS.num_inner_loop + 1)]

            return obs, actions, advantages, rewards, dones, means, log_stds, context, transitions

        def kl_n01(x):
            mean, log_std = x
            return -0.5 + tf.reduce_mean(0.5 * (mean ** 2 + tf.exp(log_std) ** 2) - log_std)

        def kl_n(x, mean, std):
            m, logstd = x
            return -0.5 * tf.reduce_mean(
                2 * logstd - 2 * tf.log(std) - (tf.exp(logstd) ** 2 / std ** 2) - (m - mean) ** 2 / std ** 2 + 1)

        def gather(x, idx):
            if isinstance(x, list) or isinstance(x, tuple):
                return [tf.gather_nd(i, idx) for i in x]
            else:
                return tf.gather_nd(x, idx)

        def take_inner_steps(inp):
            kl_ds = []
            ents = []
            traj_ents = []
            e_maml_loss = 0
            policy, qfunc1, qfunc2, vfunc, target_vfunc, context_encoder = self.func_groups[0]
            w = policy.weights[0]
            obs, actions, advantages, rewards, dones, means, log_stds, context, transitions = split_inner_step_data(inp)

            # pre-update
            for i in range(FLAGS.num_inner_loop):
                tensors = [means[i], log_stds[i]]
                ob = tf.reshape(obs[i], [FLAGS.inner_batch_size, FLAGS.inner_time_steps, self.ob_dim])
                ac = tf.reshape(actions[i], [FLAGS.inner_batch_size, FLAGS.inner_time_steps, self.action_dim])
                rw = tf.reshape(rewards[i], [FLAGS.inner_batch_size, FLAGS.inner_time_steps, 1])
                ad = tf.reshape(advantages[i], [FLAGS.inner_batch_size * FLAGS.inner_time_steps])
                mean, logstd = [tf.reshape(i, [FLAGS.inner_batch_size * FLAGS.inner_time_steps, -1]) for i in tensors]
                tr = tf.concat([ob, ac, rw], axis=-1)
                ct, _, _ = context_encoder.build_forward(tr)
                ct = tf.concat([tf.zeros([FLAGS.inner_batch_size, 1, context_encoder.context_dim]), ct[:, :-1, :]], axis=1)
                ob_ct = tf.concat([ob, ct], axis=-1)
                ac = tf.reshape(ac, [-1, self.action_dim])

                x = policy.build_mlp(ob_ct, weights=w)
                distribution = sample_from(x, discrete=self.discrete)
                distribution_prime = sample_from((mean, logstd), discrete=self.discrete)
                pre_ac = reverse_action(x, ac)
                ac = reparameterization(x, pre_ac)
                ob = tf.reshape(ob, [-1, self.ob_dim])
                ct = tf.reshape(ct, [-1, context_encoder.context_dim])
                saz = tf.concat([ob, ac, ct], axis=-1)
                min_q = tf.minimum(qfunc1.build_forward(saz), qfunc2.build_forward(saz))
                log_pi = squash_backward(distribution, ac, log_prob=True)
                loss = log_pi - min_q

                dn = tf.reshape(dones[i], [FLAGS.inner_batch_size * FLAGS.inner_time_steps])
                idx = tf.where_v2(tf.equal(dn, 0))
                loss = tf.gather(loss, idx, axis=0)
                loss = tf.reduce_mean(loss)

                kl_ds.append(tfp.distributions.kl_divergence(distribution_prime, distribution))
                ents.append(kl_n01(x))
                w = update_weights(loss, w, FLAGS.inner_lr * FLAGS.decay ** i)

            # post-update
            tensors = [means[-1], log_stds[-1]]
            ob = tf.reshape(obs[-1], [FLAGS.inner_batch_size, FLAGS.inner_time_steps, self.ob_dim])
            ac = tf.reshape(actions[-1], [FLAGS.inner_batch_size, FLAGS.inner_time_steps, self.action_dim])
            rw = tf.reshape(rewards[-1], [FLAGS.inner_batch_size, FLAGS.inner_time_steps, 1])
            ad = tf.reshape(advantages[-1], [FLAGS.inner_batch_size * FLAGS.inner_time_steps])
            mean, logstd = [tf.reshape(i, [FLAGS.inner_batch_size * FLAGS.inner_time_steps, -1]) for i in tensors]
            tr = tf.concat([ob, ac, rw], axis=-1)
            ct, _, _ = context_encoder.build_forward(tr)
            ct = tf.concat([tf.zeros([FLAGS.inner_batch_size, 1, context_encoder.context_dim]), ct[:, :-1, :]], axis=1)
            ob_ct = tf.concat([ob, ct], axis=-1)
            ac = tf.reshape(ac, [-1, self.action_dim])

            x = policy.build_mlp(ob_ct, weights=w)

            distribution = sample_from(x, discrete=self.discrete)
            distribution_prime = sample_from((mean, logstd), discrete=self.discrete)

            pi = squash_backward(distribution, ac)
            pi_prime = squash_backward(distribution_prime, ac)

            ad = (ad - tf.reduce_mean(ad)) / (tf.math.reduce_std(ad) + 1e-8)
            post_loss = tf.reduce_mean(ppo_estimator(pi, pi_prime, ad))

            mean_kl_per_step = tf.concat([tf.reduce_mean(x) for x in kl_ds], axis=0)
            kl_constrain = tf.reduce_mean(self.adaptive_kl_coeff_ph * mean_kl_per_step)

            info_bottleneck = tf.reduce_mean(ents)
            traj_entropy = tf.reduce_mean(traj_ents)

            return post_loss, kl_constrain, info_bottleneck, mean_kl_per_step, e_maml_loss, traj_entropy, 0, 0, 0

        def generate_tvf(device_id):
            def train_value_func(inp):
                index, transitions, ct_emb = inp

                policy, qfunc1, qfunc2, vfunc, target_vfunc, context_encoder = self.func_groups[device_id]
                B = tf.shape(transitions)[0]

                trs, _ = [tf.squeeze(tensor) for tensor in tf.split(transitions, [self.ob_dim + self.action_dim + 1, self.ob_dim + 1], axis=-1)]
                trs = tf.reshape(trs, [FLAGS.value_batch_size, -1, self.ob_dim + self.action_dim + 1])
                cts, _, _ = context_encoder.build_forward(trs)
                cts = tf.concat([tf.zeros([FLAGS.value_batch_size, 1, FLAGS.hidden_dim]), cts], axis=1)
                index = tf.to_int32(index)
                idx = tf.transpose(tf.stack([tf.range(FLAGS.value_batch_size), index]))
                transitions = tf.gather_nd(transitions, idx)
                ct = tf.gather_nd(cts, idx)
                idx = tf.transpose(tf.stack([tf.range(FLAGS.value_batch_size), index + 1]))
                next_ct = tf.gather_nd(cts, idx)
                with tf.device("/CPU:0"):
                    o, a, r, no, t = [tf.squeeze(tensor) for tensor in tf.split(transitions, [self.ob_dim,  self.action_dim, 1, self.ob_dim, 1], axis=-1)]
                o = tf.reshape(o, [B, self.ob_dim])
                no = tf.reshape(no, [B, self.ob_dim])
                a = tf.reshape(a, [B, self.action_dim])

                saz = tf.concat([o, a, ct], axis=-1)
                sz2 = tf.concat([no, next_ct], axis=-1)
                sz2 = tf.reshape(sz2, [-1, self.ob_dim + context_encoder.context_dim])
                q1 = qfunc1.build_forward(saz)
                q2 = qfunc2.build_forward(saz)
                target_v = target_vfunc.build_forward(sz2)
                target_q_value = tf.stop_gradient(r * FLAGS.reward_scale + (1 - t) * FLAGS.gamma * target_v)
                qf_loss = tf.reduce_mean((q1 - target_q_value)**2 + (q2 - target_q_value)**2)

                task_z = ct_emb
                ob_ct = tf.concat([o, ct], axis=-1)
                x = policy.build_mlp(ob_ct, weights=policy.weights[0])
                new_actions, dist = sample_from(x, num_samples=16, discrete=self.discrete)
                log_pi = squash_backward(dist, new_actions, log_prob=True)
                bc_z = tf.broadcast_to(task_z, [tf.shape(new_actions)[0], tf.shape(new_actions)[1], context_encoder.context_dim])
                bc_o = tf.broadcast_to(o, [tf.shape(new_actions)[0], tf.shape(new_actions)[1], self.ob_dim])
                sz = tf.concat([o, task_z], axis=-1)
                sz = tf.reshape(sz, [-1, self.ob_dim + context_encoder.context_dim])
                saz = tf.concat([bc_o, new_actions, bc_z], axis=-1)
                saz = tf.reshape(saz, [tf.shape(new_actions)[0], tf.shape(new_actions)[1], self.ob_dim + self.action_dim + context_encoder.context_dim])
                q1 = qfunc1.build_forward(saz)
                q2 = qfunc2.build_forward(saz)
                min_q = tf.minimum(q1, q2)
                v = vfunc.build_forward(sz)
                target_v_value = tf.stop_gradient(tf.reduce_mean(min_q - log_pi, axis=0))
                vf_loss = tf.reduce_mean((v - target_v_value) ** 2)

                return qf_loss, vf_loss, ct
            return train_value_func

        # forward
        x, q1, q2, v, tv, ct, (hs, cs) = self.build_forward(0)
        self.forward_context, self.forward_h_state, self.forward_c_state = ct, hs, cs
        self.forward_action, _ = sample_from(x, 1, discrete=self.discrete)
        self.forward_mean, self.forward_log_std = tf.squeeze(x[0]), tf.squeeze(x[1])
        _ = self.build_forward(1)

        self.initialize_target = overwrite_vars("target_v_func_GPU_0", "v_func_GPU_0")
        self.sync_func = tf.group(overwrite_vars("policy_GPU_1", "policy_GPU_0"),
                                  overwrite_vars("q_func1_GPU_1", "q_func1_GPU_0"),
                                  overwrite_vars("q_func2_GPU_1", "q_func2_GPU_0"),
                                  overwrite_vars("v_func_GPU_1", "v_func_GPU_0"),
                                  overwrite_vars("target_v_func_GPU_1", "target_v_func_GPU_0"),
                                  overwrite_vars("rnn_ctx_enc_GPU_1", "rnn_ctx_enc_GPU_0"))

        soft_update = lambda x, y: (1 - self.tau) * x + self.tau * y
        self.update_target_vfs_op = tf.group(
            overwrite_vars("target_v_func_GPU_0", "v_func_GPU_0", operator=soft_update),
            overwrite_vars("target_v_func_GPU_1", "v_func_GPU_1", operator=soft_update))
        self.update_policy_op = overwrite_vars("policy_GPU_1", "policy_GPU_0")

        with tf.device("/GPU:0"):
            policy = self.func_groups[0][0]
            context_encoder = self.func_groups[0][5]
            qfunc1 = self.func_groups[0][1]
            qfunc2 = self.func_groups[0][2]
            weights, weights_bak = policy.weights[:2]

            # update
            obs = tf.reshape(self.obs, [FLAGS.inner_batch_size, FLAGS.inner_time_steps, self.ob_dim])
            acs = tf.reshape(self.actions, [FLAGS.inner_batch_size, FLAGS.inner_time_steps, self.action_dim])
            rws = tf.reshape(self.rews, [FLAGS.inner_batch_size, FLAGS.inner_time_steps, 1])
            trs = tf.concat([obs, acs, rws], axis=-1)
            cts, _, _ = context_encoder.build_forward(trs)
            cts = tf.concat([tf.zeros([FLAGS.inner_batch_size, 1, context_encoder.context_dim]), cts[:, :-1, :]], axis=1)
            ob_ct = tf.concat([obs, cts], axis=-1)
            acs = tf.reshape(acs, [-1, self.action_dim])

            x = policy.build_mlp(ob_ct, weights=weights)
            distribution = sample_from(x, discrete=self.discrete)
            pre_acs = reverse_action(x, acs)
            acs = reparameterization(x, pre_acs)
            obs = tf.reshape(obs, [-1, self.ob_dim])
            cts = tf.reshape(cts, [-1, context_encoder.context_dim])
            saz = tf.concat([obs, acs, cts], axis=-1)
            min_q = tf.minimum(qfunc1.build_forward(saz), qfunc2.build_forward(saz))
            log_pi = squash_backward(distribution, acs, log_prob=True)
            loss = log_pi - min_q

            dns = tf.reshape(self.dns, [FLAGS.inner_batch_size * FLAGS.inner_time_steps])
            idx = tf.where_v2(tf.equal(dns, 0))
            loss = tf.gather(loss, idx, axis=0)
            loss = tf.reduce_mean(loss)

            w = update_weights(loss, weights, FLAGS.inner_lr * FLAGS.decay ** self.decay_step)
            self.update_weights_op = assign_vars(weights, w)
            self.save_origin = assign_vars(weights_bak, weights)
            self.restore_origin = assign_vars(weights, weights_bak)


        if self.training:
            with tf.device("/GPU:0"):
                elems = (self.obs, self.actions, self.advs, self.rews, self.dns, self.means, self.log_stds, self.context, self.transitions)
                result = tf.map_fn(take_inner_steps, elems=elems, parallel_iterations=FLAGS.meta_batch_size)
                post_loss, kl_constraint, info_bn, mean_kls, e_maml_loss, entropy, offpac_loss, _, _ = result
                self.entropy = info_bn = tf.reduce_mean(info_bn)

                entropy = tf.reduce_mean(entropy)
                loss = tf.reduce_mean(post_loss)
                constraints = tf.reduce_mean(kl_constraint)
                if FLAGS.use_info_bn:
                    constraints += FLAGS.info_bottleneck_coeff * tf.reduce_mean(info_bn)
                if FLAGS.use_min_traj_ent:
                    self.min_ent_op = tf.train.AdamOptimizer(1e-5).minimize(entropy)
                total_loss = loss + constraints
                self.mean_kls = tf.reduce_mean(mean_kls, axis=0)

                opt = tf.train.AdamOptimizer(FLAGS.meta_lr)
                grads_and_vars = opt.compute_gradients(total_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy"))
                if FLAGS.clip_norm:
                    grads = [g for g, _ in grads_and_vars if g is not None]
                    grads = tf.clip_by_global_norm(grads, FLAGS.clip_norm)
                    vars = [v for g, v in grads_and_vars if g is not None]
                    grads_and_vars = zip(grads, vars)
                self.meta_train_op = opt.apply_gradients(grads_and_vars)

            ############################################################################################################
            #                                       DISTRIBUTED VERSION
            ############################################################################################################
            with tf.device("/CPU:0"):
                context_batchs = tf.split(self.context, 2)
                transition_batches = tf.split(self.transitions, 2)
                embedding_batches = tf.split(self.embedding, 2)
                elems1, elems2 = zip(context_batchs, transition_batches, embedding_batches)

            with tf.device("/GPU:0"):
                result1 = tf.map_fn(generate_tvf(0), elems=elems1, parallel_iterations=FLAGS.value_meta_batch_size//2)
                qf_loss1, vf_loss1, ct_emb1 = result1
                qf_loss1, vf_loss1 = tf.reduce_mean(qf_loss1), tf.reduce_mean(vf_loss1)
                enc_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="rnn_ctx_enc_GPU_0")
                qf1_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_func1_GPU_0")
                qf2_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_func2_GPU_0")
                vf_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="v_func_GPU_0")
                var_list1 = enc_vars1 + qf1_vars1 + qf2_vars1
                qf_loss1_grads = tf.gradients(qf_loss1, var_list1)
                vf_loss1_grads = tf.gradients(vf_loss1, vf_vars1)
            with tf.device("/GPU:1"):
                result2 = tf.map_fn(generate_tvf(1), elems=elems2, parallel_iterations=FLAGS.value_meta_batch_size//2)
                qf_loss2, vf_loss2, ct_emb2 = result2
                qf_loss2, vf_loss2 = tf.reduce_mean(qf_loss2), tf.reduce_mean(vf_loss2)
                enc_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="rnn_ctx_enc_GPU_1")
                qf1_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_func1_GPU_1")
                qf2_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_func2_GPU_1")
                vf_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="v_func_GPU_1")
                var_list2 = enc_vars2 + qf1_vars2 + qf2_vars2
                qf_loss2_grads = tf.gradients(qf_loss2, var_list2)
                vf_loss2_grads = tf.gradients(vf_loss2, vf_vars2)

            qf_loss_grads = [(g1 + g2) / 2. for g1, g2 in zip(qf_loss1_grads, qf_loss2_grads)]
            vf_loss_grads = [(g1 + g2) / 2. for g1, g2 in zip(vf_loss1_grads, vf_loss2_grads)]

            vf_grad_and_vars1 = zip(vf_loss_grads, vf_vars1)
            vf_grad_and_vars2 = zip(vf_loss_grads, vf_vars2)

            d1 = dict(zip(var_list1, qf_loss_grads))
            d2 = dict(zip(var_list2, qf_loss_grads))
            var_lists = [enc_vars1, qf1_vars1, qf2_vars1]
            gnvs1 = [[(d1[v], v) for v in var_list] for var_list in var_lists]
            var_lists = [enc_vars2, qf1_vars2, qf2_vars2]
            gnvs2 = [[(d2[v], v) for v in var_list] for var_list in var_lists]
            train_ops = [tf.group(tf.train.AdamOptimizer(FLAGS.value_lr).apply_gradients(gv1),
                                  tf.train.AdamOptimizer(FLAGS.value_lr).apply_gradients(gv2)) for gv1, gv2 in
                         zip(gnvs1, gnvs2)]
            self.train_ctx_enc, self.qf1_op, self.qf2_op = train_ops
            self.vf_op = tf.group(tf.train.AdamOptimizer(FLAGS.value_lr).apply_gradients(vf_grad_and_vars1),
                                  tf.train.AdamOptimizer(FLAGS.value_lr).apply_gradients(vf_grad_and_vars2))
            self.get_embedding = tf.concat([tf.stack(ct_emb1), tf.stack(ct_emb2)], axis=0)
            self.qf_loss = tf.reduce_mean(qf_loss1 + qf_loss2)
            self.vf_loss = tf.reduce_mean(vf_loss1 + vf_loss2)

    def act(self, ob, ct):
        session = tf.get_default_session()
        feed_dict = {self.obs: ob, self.context: ct}
        action, mean, log_std = session.run([self.forward_action, self.forward_mean, self.forward_log_std], feed_dict=feed_dict)
        return action, mean, log_std

    def build_forward(self, device_id):
        with tf.device("/GPU:{}".format(device_id)):
            policy, qfunc1, qfunc2, vfunc, target_vfunc, context_encoder = self.func_groups[device_id]
            weights, weights_bak = policy.weights[:2]
            obs = tf.reshape(self.obs, [1, -1, self.ob_dim])
            acs = tf.reshape(self.actions, [1, -1, self.action_dim])
            rws = tf.reshape(self.rews, [1, -1, 1])
            cts = tf.reshape(self.context, [1, -1, context_encoder.context_dim])
            ob_ct = tf.concat([obs, cts], axis=-1)
            trs = tf.concat([obs, acs, rws], axis=-1)

            obs = tf.reshape(obs, [-1, self.ob_dim])
            acs = tf.reshape(acs, [-1, self.action_dim])
            cts = tf.reshape(cts, [-1, context_encoder.context_dim])
            saz = tf.concat([obs, acs, cts], axis=-1)
            sz = tf.concat([obs, cts], axis=-1)
            x = policy.build_mlp(ob_ct, weights=weights)
            q1, q2 = qfunc1.build_forward(saz), qfunc2.build_forward(saz)
            v = vfunc.build_forward(sz)
            tv = target_vfunc.build_forward(sz)
            ct, hs, cs = context_encoder.build_forward(trs, init_state=[self.init_h, self.init_c])
            return x, q1, q2, v, tv, ct, (hs, cs)

    def next_context(self, state, transition):
        session = tf.get_default_session()
        feed_dict = dict(zip((self.init_h, self.init_c, self.obs, self.actions, self.rews), state + transition))
        ct, hs, cs = session.run([self.forward_context, self.forward_h_state, self.forward_c_state], feed_dict=feed_dict)
        return ct, (hs, cs)

    def update_weights(self, trajectory, decay_step):
        session = tf.get_default_session()
        feed_dict = dict(zip([self.obs, self.actions, self.rews, self.advs, self.dns, self.means, self.log_stds, self.decay_step], trajectory + (decay_step,)))
        session.run(self.update_weights_op, feed_dict=feed_dict)

    def save_weights(self):
        session = tf.get_default_session()
        session.run(self.save_origin)

    def restore_weights(self):
        session = tf.get_default_session()
        session.run(self.restore_origin)


    def adjust_coeff(self, current_kl):
        if current_kl.ndim == 0:
            current_kl = np.expand_dims(current_kl, axis=0)
        for i, kl in enumerate(current_kl):
            if kl < FLAGS.target_kl / 1.5:
                self.kl_coeff[i] /= 2
            elif kl > FLAGS.target_kl * 1.5:
                self.kl_coeff[i] *= 2

    def train_step(self, iteration, meta_batch, train_policy=True, train_value=True):
        def value_train_loop():
            losses = 0

            widgets = [
                'Iter {}'.format(iteration), ' ',
                progressbar.Percentage(), ' ',
                progressbar.widgets.SimpleProgress(
                    format='(%s)' % progressbar.widgets.SimpleProgress.DEFAULT_FORMAT), ' ',
                progressbar.Timer(),
                ' [', progressbar.DynamicMessage("qf_loss"),'] ', '[', progressbar.DynamicMessage("kl_loss"), '] ', '[', progressbar.DynamicMessage("vf_loss"), ']',
            ]
            with progressbar.ProgressBar(max_value=FLAGS.value_iterations, widgets=widgets, redirect_stdout=True) as bar:
                for i in range(FLAGS.value_iterations):
                    loss = self._train_value()
                    losses += loss
                    bar.update(i, qf_loss=loss[0], kl_loss=loss[1], vf_loss=loss[2])
            return losses / FLAGS.value_iterations

        def policy_train_loop():
            ent = np.nan
            for _ in range(FLAGS.num_ppo_steps):
                phs = (self.obs, self.actions, self.advs, self.rews, self.dns, self.means, self.log_stds,
                       self.adaptive_kl_coeff_ph, self.global_step_ph)
                data = meta_batch + (self.kl_coeff, self.global_step)
                feed_dict = dict(zip(phs, data))
                ent = self._train_policy(feed_dict)
            return ent
        ent = np.nan
        self.restore_weights()

        if train_value:
            logging.info("  training value functions...")
            self.update_policy_op.run()
            avg_loss, time_elapsed = timed_wrpper(value_train_loop)
            logging.info("  done. total time: {:.2f}".format(time_elapsed))
        else:
            avg_loss = (np.nan,)*3
            logging.info("  skipped value training...")

        if train_policy:
            logging.info("  training policy...")
            ent, time_elapsed = timed_wrpper(policy_train_loop)
            self.update_policy_op.run()
            logging.info("  done. total time: {:.2f}".format(time_elapsed))
        else:
            logging.info("  skipped policy training...")

        self.global_step += 1
        return list(avg_loss), ent

    def _train_value(self):
        session = tf.get_default_session()
        task_ids = np.tile(np.arange(self.num_tasks), FLAGS.value_meta_batch_size // self.num_tasks)
        transitions, idx = self.replay_buffer.sample(task_ids, FLAGS.value_batch_size)
        emb = np.zeros([FLAGS.value_batch_size, FLAGS.hidden_dim])
        feed_dict = dict(zip((self.context, self.transitions, self.embedding), (idx, transitions, emb)))
        _, _, _, qf_loss, emb = session.run([self.train_ctx_enc, self.qf1_op, self.qf2_op, self.qf_loss, self.get_embedding], feed_dict=feed_dict)
        feed_dict[self.embedding] = emb
        _, vf_loss = session.run([self.vf_op, self.vf_loss], feed_dict=feed_dict)
        session.run(self.update_target_vfs_op)
        return np.array([qf_loss, np.nan, vf_loss])

    def _train_policy(self, feed_dict):
        session = tf.get_default_session()
        _, mean_kls, ent = session.run([self.meta_train_op, self.mean_kls, self.entropy], feed_dict=feed_dict)
        self.adjust_coeff(mean_kls)
        return ent



