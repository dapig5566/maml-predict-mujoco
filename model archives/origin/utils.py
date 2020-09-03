from dependencies import *

def vpg_estimator(log_pi, adv):
    return -log_pi * adv


def lvc_estimator(pi, pi_prime, adv):
    return -pi * adv / pi_prime


def ppo_estimator(pi, pi_prime, adv):
    epi = 0.3
    r = pi / pi_prime
    return -tf.minimum(r * adv, tf.clip_by_value(r, 1 - epi, 1 + epi) * adv)


def squash_backward(dist, ac, log_prob=False):
    pre_tanh = tf.atanh(ac)
    if log_prob:
        log_pi = dist.log_prob(pre_tanh)
        det = tf.reduce_prod(1 - ac ** 2, axis=-1) + 1e-8
        return log_pi - tf.log(det)
    else:
        pi = dist.prob(pre_tanh)
        det = tf.reduce_prod(1 - ac**2, axis=-1) + 1e-8
        return pi / det


def reverse_action(dist, ac):
    mean, logstd = dist
    pre_tanh = tf.atanh(ac)
    ac = tf.stop_gradient((pre_tanh - mean) / (tf.exp(logstd)) + 1e-8)
    return ac


def reparameterization(dist, ac):
    mean, log_std = dist
    ac = tf.tanh(ac * tf.exp(log_std) + mean)
    return ac
