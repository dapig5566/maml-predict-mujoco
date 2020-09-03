"""
（已完成）验证环境初始化对trajectory的影响，看看是否还有无法稳定抽样的问题。--------------------------环境对trajectory有较大影响。
（已完成）将多次抽样全部encode进LSTM中，使得每次抽样尽可能分布广泛，又解决随机性不稳定问题。-------------实验未能成功。轨迹全部收敛到一起，无明显约束的情况下没有用。
（已完成）使用diayn使得四次连续抽样分化到不同轨迹。-----------------------------------------------可以得到分离的4条轨迹
（已完成）使得post-update轨迹变得正确。-------------------------------------------------------需要大的inner lr，并发现轨迹分布广才容易得到高的post-performance，进一步印证mutual information的理论。
（已完成）查看E-MAML的结果。-----------------------------------------------------------------情况较差，考虑使用PEARL的context加上确定性决策优化以及E-MAML试试。
（进行中）E-MAML/ProMP+sac/dpg+PEARL。
1.确认policy与Q匹配，是否增加sac，调查ACE方法。 2.LSTM编码使用确定性还是所有轨迹的平均编码。 3.是否使用mlp乘积作为z编码。
目前使用MLP编码。q-v作为advantage没有问题。可调点：relu，kl，v_lr, q-v
（待进行）PEARL+inner update，本质上与diayn的实验结果相近，skill在抽样过程中不变，z~p(z|c)在抽样过程中也不变，所以z与skill没有本质区别，
带skill的pre-policy可以优化为不带skill（skill=0）的post-policy，z相当于是多个skill而已，
于是只要Q对pre-policy的advantage estimation是准确的，就可以进行k步inner update。这样既可以有structured exp，又可以进行policy更新
"""

import logging
from tensorflow.python.platform import flags
from trainer import train_2d_exploration, eval_maml_exp

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 500, "number of iterations.")
flags.DEFINE_integer("num_inner_loop", 1, "number of inner steps.")
flags.DEFINE_integer("n_way", 5, "number of classes.")
flags.DEFINE_integer("k_shot", 1, "number of shots.")
flags.DEFINE_integer("meta_batch_size", 20, "the meta batch size.")
flags.DEFINE_integer("inner_time_steps", 400, "the time steps of an sub inner trajectory.")
flags.DEFINE_integer("inner_batch_size", 2, "the number of trajectories sampled from the same task.")
flags.DEFINE_integer("reset_times", 4, "the number of times to reset the same task when performing a trajectory.")
flags.DEFINE_integer("num_ppo_steps", 2, "the number of ppo steps for the meta update.")
flags.DEFINE_integer("show_pre_update_every", 2, "the number of steps between two plots.")

flags.DEFINE_integer("extra_rl_batch_size", 0, "the extra meta batch size.")
flags.DEFINE_integer("train_value_iters", 0, "the iterations training value functions.")
flags.DEFINE_integer("value_meta_batch_size", 20, "the meta batch size of value iteration.")
flags.DEFINE_integer("context_batch_size", 256, "the number of transitions in context encoder.")
flags.DEFINE_integer("value_batch_size", 256, "the number of transitions in context encoder.")
flags.DEFINE_integer("value_iterations", 128, "the number of transitions in context encoder.")
flags.DEFINE_integer("hidden_dim", 32, "the number of transitions in context encoder.")
flags.DEFINE_float("reward_scale", 3., "reward scale.")

flags.DEFINE_boolean("use_ob_encoder", False, "whether to encode trajectory.")
flags.DEFINE_boolean("use_info_bn", True, "whether to use information bottleneck.")
flags.DEFINE_boolean("use_attention", False, "whether to use attention.")
flags.DEFINE_boolean("cut_zero", False, "whether to discard zero reward states.")
flags.DEFINE_boolean("use_min_traj_ent", False, "whether to minimize trajectory entropy.")
flags.DEFINE_boolean("reset_traj", False, "whether to reset trajectory.")

flags.DEFINE_float("meta_lr", 1e-3, "meta learning rate.")
flags.DEFINE_float("inner_lr", 0.05, "inner learning rate.")
flags.DEFINE_float("value_lr", 5e-4, "value functions learning rate.")
flags.DEFINE_float("gamma", 0.99, "the discount factor.")
flags.DEFINE_float("decay", 0.8, "the decay rate in inner loop.")
flags.DEFINE_float("clip_norm", 0, "the max norm of each gradient.")
flags.DEFINE_float("init_kl_coeff", 5e-4, "the initial value of kl constraint coefficient.")
flags.DEFINE_float("target_kl", 0.01, "the target value of kl divergence.")
flags.DEFINE_float("mutual_info_coeff", 1e-3, "the coefficient of mutual information term.")
flags.DEFINE_float("info_bottleneck_coeff", 0.5, "the coefficient of information bottleneck term.")
flags.DEFINE_float("ent_coeff", 0.7, "the coefficient of entropy term.")
flags.DEFINE_float("traj_ent_coeff", 0.1, "the coefficient of entropy term.")
flags.DEFINE_float("min_pre_coeff", 0.1, ".")
if __name__ == "__main__":
    train_2d_exploration()
    # eval_maml_exp()
#####################################
#               policy reg
#####################################
