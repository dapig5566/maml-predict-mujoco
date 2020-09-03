import tensorflow_probability as tfp
from dependencies import *
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class Classifier:
    def __init__(self, num_classes, feature_dim):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.weights = None
        self.build_initial_weights()

    def build_convnet(self, input_data, weights=None, training=None):
        x = input_data
        if weights is None:
            weights = self.weights
        for i in range(2):
            x = tf.nn.conv2d(x, weights["conv_kernel_{}".format(i)], 2, "SAME") #+ weights["conv_bias_{}".format(i)]
            x = tf.layers.batch_normalization(x, training=training, name="batch_norm_conv_{}".format(i), reuse=tf.AUTO_REUSE)
            x = tf.nn.relu(x)

        x = tf.layers.average_pooling2d(x, 2, 2, "same")
        x = tf.reshape(x, [-1, 4*4*self.feature_dim])

        for i in range(1):
            x = tf.matmul(x, weights["dense_kernel_{}".format(i)])
            x = tf.layers.batch_normalization(x, training=training, name="batch_norm_fc_{}".format(i), reuse=tf.AUTO_REUSE)
            x = tf.nn.relu(x)
        x = tf.matmul(x, weights["dense_kernel_final"]) + weights["dense_bias_final"]

        return x

    def build_initial_weights(self):
        weights = {}
        kernel_init = tf.initializers.he_normal()
        fc_init = tf.initializers.he_normal()
        bias_init = tf.initializers.zeros()
        with tf.variable_scope("classifier"):
            weights["conv_kernel_0"] = tf.get_variable("conv_kernel_0", [4, 4, 1, self.feature_dim // 2], tf.float32,
                                                       initializer=kernel_init, trainable=True)
            weights["conv_bias_0"] = tf.get_variable("conv_bias_0", [self.feature_dim // 2], tf.float32,
                                                     initializer=bias_init, trainable=True)
            weights["conv_kernel_1"] = tf.get_variable("conv_kernel_1", [4, 4, self.feature_dim // 2, self.feature_dim],
                                                       tf.float32, initializer=kernel_init, trainable=True)
            weights["conv_bias_1"] = tf.get_variable("conv_bias_1", [self.feature_dim], tf.float32,
                                                     initializer=bias_init, trainable=True)
            weights["dense_kernel_0"] = tf.get_variable("dense_kernel_0", [4*4*self.feature_dim, 1024], tf.float32,
                                                     initializer=fc_init, trainable=True)
            weights["dense_bias_0"] = tf.get_variable("dense_bias_0", [1024], tf.float32,
                                                        initializer=bias_init, trainable=True)
            weights["dense_kernel_final"] = tf.get_variable("dense_kernel_final", [1024, self.num_classes], tf.float32,
                                                        initializer=fc_init, trainable=True)
            weights["dense_bias_final"] = tf.get_variable("dense_bias_final", [self.num_classes], tf.float32,
                                                      initializer=bias_init, trainable=True)
        self.weights = weights


class Policy:
    def __init__(self, ob_dim, action_dim, hidden_dim, min_std=1e-3, max_std=1.2, memory_size=128):
        # #########################################################################
        # #                              test RNNSGD                              #
        # #########################################################################
        # 暂时搁置
        # self.memory_size = memory_size
        # self.memory = tf.get_variable("memory", [1, self.memory_size], tf.float32, initializer=tf.initializers.random_uniform(-0.1, 0.1), trainable=True)
        # self.memory_bak = tf.get_variable("memory_bak", [1, self.memory_size], tf.float32, initializer=tf.initializers.random_uniform(-0.1, 0.1), trainable=True)
        # #########################################################################

        self.min_std = min_std
        self.max_std = max_std
        self.action_dim = action_dim
        self.ob_dim = ob_dim
        self.hidden_dim = hidden_dim
        self.weights = []
        self.build_initial_weights(name="policy")
        self.build_initial_weights(name="backup")
        self.build_initial_weights(name="target_policy")

    def build_mlp(self, input_data, weights=None):
        weights = weights or self.weights[0]
        if FLAGS.use_ob_encoder:
            raise NotImplementedError
            # skills = tf.reshape(input_data[1], [-1, FLAGS.reset_times])
            # mlp_input = tf.concat([state_code, skills], axis=-1)
            # #########################################################################
            # #                              test RNNSGD                              #
            # #########################################################################
            # # 暂时搁置
            # mlp_input = tf.concat([state_code, tf.tile(self.memory, [tf.shape(state_code)[0], 1])], axis=-1)
            # #########################################################################
        else:
            mlp_input = tf.reshape(input_data, [-1, self.ob_dim + FLAGS.hidden_dim])
        # mean_net
        x = tf.matmul(mlp_input, weights["dense_kernel_0"]) + weights["dense_bias_0"]
        x = tf.nn.tanh(x)
        x = tf.matmul(x, weights["dense_kernel_1"]) + weights["dense_bias_1"]
        x = tf.nn.tanh(x)
        x = tf.matmul(x, weights["dense_kernel_6"]) + weights["dense_bias_6"]
        x = tf.nn.tanh(x)
        # x = tf.matmul(x, weights["dense_kernel_7"]) + weights["dense_bias_7"]
        # x = tf.nn.tanh(x)
        mean = tf.matmul(x, weights["dense_kernel_2"]) + weights["dense_bias_2"]


        # logstd_net
        x = tf.matmul(mlp_input, weights["dense_kernel_3"]) + weights["dense_bias_3"]
        x = tf.nn.tanh(x)
        x = tf.matmul(x, weights["dense_kernel_4"]) + weights["dense_bias_4"]
        x = tf.nn.tanh(x)
        x = tf.matmul(x, weights["dense_kernel_8"]) + weights["dense_bias_8"]
        x = tf.nn.tanh(x)
        # x = tf.matmul(x, weights["dense_kernel_9"]) + weights["dense_bias_9"]
        # x = tf.nn.tanh(x)
        x = tf.matmul(x, weights["dense_kernel_5"]) + weights["dense_bias_5"]
        log_std = tf.maximum(x, tf.log(self.min_std))

        return (mean, log_std)

    def build_initial_weights(self, name=None):
        weights = {}
        # bound = 1. / tf.sqrt(float(self.hidden_dim))
        kernel_init = tf.initializers.random_normal(0, 0.01)
        bias_init = tf.initializers.zeros()

        with tf.variable_scope(name or "policy"):
            # _bound = 1. / tf.sqrt(float(self.ob_dim + FLAGS.hidden_dim))
            weights["dense_kernel_0"] = tf.get_variable("dense_kernel_0", [self.ob_dim + FLAGS.hidden_dim, self.hidden_dim], tf.float32,
                                                        initializer=kernel_init, trainable=True)
            weights["dense_bias_0"] = tf.get_variable("dense_bias_0", [self.hidden_dim], tf.float32,
                                                      initializer=bias_init, trainable=True)
            weights["dense_kernel_1"] = tf.get_variable("dense_kernel_1", [self.hidden_dim, self.hidden_dim], tf.float32,
                                                        initializer=kernel_init, trainable=True)
            weights["dense_bias_1"] = tf.get_variable("dense_bias_1", [self.hidden_dim], tf.float32,
                                                      initializer=bias_init, trainable=True)
            weights["dense_kernel_2"] = tf.get_variable("dense_kernel_2", [self.hidden_dim, self.action_dim],
                                                        tf.float32,
                                                        initializer=kernel_init, trainable=True)
            weights["dense_bias_2"] = tf.get_variable("dense_bias_2", [self.action_dim], tf.float32,
                                                      initializer=bias_init, trainable=True)


            weights["dense_kernel_6"] = tf.get_variable("dense_kernel_6", [self.hidden_dim, self.hidden_dim],
                                                        tf.float32,
                                                        initializer=kernel_init, trainable=True)
            weights["dense_bias_6"] = tf.get_variable("dense_bias_6", [self.hidden_dim], tf.float32,
                                                      initializer=bias_init, trainable=True)
            weights["dense_kernel_7"] = tf.get_variable("dense_kernel_7", [self.hidden_dim, self.hidden_dim],
                                                        tf.float32,
                                                        initializer=kernel_init, trainable=True)
            weights["dense_bias_7"] = tf.get_variable("dense_bias_7", [self.hidden_dim], tf.float32,
                                                      initializer=bias_init, trainable=True)


            # logstd_net weights
            weights["dense_kernel_3"] = tf.get_variable("dense_kernel_3", [self.ob_dim + FLAGS.hidden_dim, self.hidden_dim], tf.float32,
                                                        initializer=kernel_init, trainable=True)
            weights["dense_bias_3"] = tf.get_variable("dense_bias_3", [self.hidden_dim], tf.float32,
                                                      initializer=bias_init, trainable=True)
            weights["dense_kernel_4"] = tf.get_variable("dense_kernel_4", [self.hidden_dim, self.hidden_dim], tf.float32,
                                                        initializer=kernel_init, trainable=True)
            weights["dense_bias_4"] = tf.get_variable("dense_bias_4", [self.hidden_dim], tf.float32,
                                                      initializer=bias_init, trainable=True)
            weights["dense_kernel_5"] = tf.get_variable("dense_kernel_5", [self.hidden_dim, self.action_dim], tf.float32,
                                                        initializer=kernel_init, trainable=True)
            weights["dense_bias_5"] = tf.get_variable("dense_bias_5", [self.action_dim], tf.float32,
                                                      initializer=bias_init, trainable=True)

            weights["log_std"] = tf.get_variable("log_std", [1, self.action_dim], tf.float32,
                                                 initializer=kernel_init, trainable=True)



            weights["dense_kernel_8"] = tf.get_variable("dense_kernel_8", [self.hidden_dim, self.hidden_dim],
                                                        tf.float32,
                                                        initializer=kernel_init, trainable=True)
            weights["dense_bias_8"] = tf.get_variable("dense_bias_8", [self.hidden_dim], tf.float32,
                                                      initializer=bias_init, trainable=True)
            weights["dense_kernel_9"] = tf.get_variable("dense_kernel_9", [self.hidden_dim, self.hidden_dim],
                                                        tf.float32,
                                                        initializer=kernel_init, trainable=True)
            weights["dense_bias_9"] = tf.get_variable("dense_bias_9", [self.hidden_dim], tf.float32,
                                                      initializer=bias_init, trainable=True)

        self.weights.append(weights)


class ContextualStateImportance:
    def __init__(self, hidden_dim=64, embedding_dim=32):
        # (B, T, N)
        self.top_k = 40
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.contextual_encoder = tf.keras.layers.CuDNNLSTM(self.hidden_dim, return_sequences=True)
        self.attention = tf.layers.Dense(self.hidden_dim)
        self.query = tf.get_variable("importance_query", [self.embedding_dim], initializer=tf.initializers.glorot_normal())
        self.transition_encoder = tf.layers.Dense(self.embedding_dim)

    def build_attention(self, x):
        q = tf.expand_dims(tf.tile(tf.expand_dims(self.query, axis=0), [tf.shape(x)[0], 1]), axis=1)
        x = self.transition_encoder(x)
        x = self.contextual_encoder(tf.concat([x, q], axis=1))
        x = tf.einsum("ijk,ik->ij", self.attention(x[:, :-1, :]), x[:, -1, :])
        x = tf.nn.softmax(x, axis=-1)
        col_idx = tf.argsort(x, axis=-1, direction="DESCENDING")
        col_idx = col_idx[:, :self.top_k]
        row_idx = tf.tile(tf.expand_dims(tf.range(tf.shape(x)[0]), axis=1), [1, self.top_k])
        idx = tf.reshape(tf.transpose(tf.stack([row_idx, col_idx]), [1, 2, 0]), [-1, 2])
        x = tf.gather_nd(x, idx)
        # x = tf.reshape(x, [FLAGS.inner_batch_size * FLAGS.inner_time_steps])
        return x, idx


class BaseMLP:
    def __init__(self, layer_sizes, name=None):
        self.sizes = layer_sizes
        self.layers = [tf.layers.Dense(size, kernel_initializer=tf.initializers.random_normal(0, 0.01),
                                       bias_initializer=tf.initializers.zeros()) for size in self.sizes]
        self.name = name or "mlp"

    def build_forward(self, x):
        with tf.variable_scope(self.name):
            for l in self.layers[:-1]:
                x = l(x)
                x = tf.nn.relu(x)
            x = self.layers[-1](x)
        return x

class SkillClassifier(BaseMLP):
    def __init__(self, hidden_dim, num_skills):
        self.hidden_dim = hidden_dim
        self.num_skills = num_skills
        super(SkillClassifier, self).__init__((self.hidden_dim, self.hidden_dim, self.num_skills))


class Qfunction(BaseMLP):
    def __init__(self, hidden_dim, name):
        self.hidden_dim = hidden_dim
        super(Qfunction, self).__init__((self.hidden_dim, self.hidden_dim, self.hidden_dim, 1), name)

    def build_forward(self, x):
        x = super().build_forward(x)
        return tf.squeeze(x)


class Vfunction(BaseMLP):
    def __init__(self, hidden_dim, name):
        self.hidden_dim = hidden_dim
        super(Vfunction, self).__init__((self.hidden_dim, self.hidden_dim, self.hidden_dim, 1), name)

    def build_forward(self, x):
        x = super().build_forward(x)
        return tf.squeeze(x)


class ContextEncoder:
    def __init__(self, hidden_dim, embedding_dim):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = hidden_dim
        self.contextual_encoder = tf.keras.layers.CuDNNLSTM(self.hidden_dim, return_state=True, return_sequences=True)
        self.transition_encoder = tf.layers.Dense(self.embedding_dim)

    def build_forward(self, x, init_state=None):
        with tf.variable_scope("rnn_ctx_enc"):
            x = self.transition_encoder(x)
            x = tf.nn.relu(x)
            x, h_state, c_state = self.contextual_encoder(x, initial_state=init_state)
            context = tf.squeeze(x)
            return context, h_state, c_state


class MLPContextualEncoder:
    def __init__(self, hidden_dim, latent_dim):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.mean_net = BaseMLP((self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim, self.latent_dim), "mlp_ctx_enc_mean")
        self.logstd_net = BaseMLP((self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim, self.latent_dim), "mlp_ctx_enc_logstd")

    def build_forward(self, x):
        return (self.mean_net.build_forward(x), tf.maximum(self.logstd_net.build_forward(x), tf.log(1e-3)))

    def gaussian_product(self, dist_info):
        mean, logstd = dist_info
        sigmas_squared = tf.exp(logstd)**2
        sigma_squared = 1. / tf.reduce_sum(tf.reciprocal(sigmas_squared), axis=0)
        mu = sigma_squared * tf.reduce_sum(mean / sigmas_squared, axis=0)
        return (mu, sigma_squared)

    def sample_from(self, dist_info, num_samples=None):
        mean, logstd = dist_info
        distribution = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(logstd))
        if num_samples is None:
            return distribution
        else:
            samples = tf.squeeze(distribution.sample(num_samples))
            return samples, distribution
# class MetaSGD:
#     def __init__(self, hidden_dim):
#         self.hidden_dim = hidden_dim
#         self.rnn_sgd = tf.keras.layers.CuDNNLSTM(self.hidden_dim, return_sequences=True)
#
#     def build_forward(self, input_data, init_state=None):
#         x, h, c = self.rnn_sgd(input_data, initial_state=init_state)
#         return x, h, c

# if __name__ == "__main__":
#     tf.disable_v2_behavior()
#     loc = tf.constant([1., 1.])
#     std = tf.constant([1.2, 1.2])
#     d = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=std)
#     s = d.sample(1)
#     a = s * 3 + 5
#     g = tf.gradients(a, [loc, std])
#     tf.InteractiveSession()
#     tf.global_variables_initializer().run()
#     for i in g:
#         print(i.eval())