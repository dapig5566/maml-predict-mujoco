import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator


def plot_pre_trajs(traj, len):
        pre = traj[0, ...]
        post = traj[1, ...]
        set_fig()
        plot_all_traj(len, pre, post)
        plt.show()


def plot_all_traj(i, pre, post):
    for n in range(0, 4):
        plt.scatter(pre[:, n*i:(n+1)*i, 0], pre[:, n*i:(n+1)*i, 1], s=4, label="pre{}".format(n))
    # plt.scatter(pre[:, n * i:(n + 1) * i, 0], pre[:, n * i:(n + 1) * i, 1], s=3, label="pre{}".format(n))
    if post is not None:
        # for n in range(0, 4):
        plt.scatter(post[:, :, 0], post[:, :, 1], s=4, label="post", color="black")
    plt.legend()




def plot_single_traj(end, i, pre, post):

    for n in range(0, 4):
        plt.scatter(pre[end, n*i:(n+1)*i, 0], pre[end, n*i:(n+1)*i, 1], label="pre{}".format(n), s=10)
    plt.scatter(post[end, :i, 0], post[end, :i, 1], color="black", s=10)
    plt.legend()


def set_fig():
    plt.figure()
    ax = plt.gca()

    plt.axes(aspect='equal')
    # plt.axis([-2, 2, -2, 2])
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    # plt.plot([-0.5, 0.0], [0.0, 0.5], color="black")
    # plt.plot([-0.5, 0.0], [0.0, -0.5], color="black")
    # plt.plot([0.0, 0.5], [0.5, 0.0], color="black")
    # plt.plot([0.0, 0.5], [-0.5, 0.0], color="black")

    plt.plot([-3.0, 3.0], [0.0, 0.0], color="black")
    plt.plot([0.0, 0.0], [-3.0, 3.0], color="black")
    alpha = np.linspace(0, 2 * np.pi, 100)
    x, y = 0.5 * np.cos(alpha), 0.5 * np.sin(alpha)
    plt.plot(x, y, 'r', color="green")


if __name__ == "__main__":
    plot = 0

    if plot == 0:
        num_tasks = 2
        avg_return_pre = []
        avg_return_post = []
        task_return_pre = [[] for _ in range(num_tasks)]
        task_return_post = [[] for _ in range(num_tasks)]
        time_step_return = []
        file = open("logs/maml_exp.log")

        try:
            while True:
                str = file.readline()
                if str == "":
                    break
                if str.find("task") != -1 and str.find("id") != -1:
                    values = re.findall(r"\[(.+?)\]", str)
                    tmp = re.search(r"([-]?\d*[.]\d*)\s+([-]?\d*[.]\d*)", values[3])
                    time_step_return.append(float(tmp.group(2)))
                if str.find("QFNCAVG") != -1:
                    values = re.findall(r"\[(.+?)\]", str)
                    tmp = re.search(r"([-]?\d*[.]\d*)\s+([-]?\d*[.]\d*)", values[1])
                    avg_return_pre.append(float(tmp.group(1)))
                    avg_return_post.append(float(tmp.group(2)))
                    for i in range(2, 4):
                        r = re.search(r"([-]?\d*[.]\d*)\s+([-]?\d*[.]\d*)", values[i])
                        if r is None:
                            print(values[i])
                        task_return_pre[i-2].append(float(r.group(1)))
                        task_return_post[i-2].append(float(r.group(2)))
                else:
                    continue
        finally:
            file.close()
        end = 350
        steps = [list(range(len(avg_return_pre)))[:end] for _ in range(num_tasks)]

        fig = plt.figure()
        data = np.array(time_step_return)
        length = (data.shape[0] // num_tasks) * num_tasks
        data = data[:length].reshape([num_tasks, -1], order="F")
        data = {"Iteration": np.tile(range(data.shape[-1]), num_tasks), "AvgReturn": data.reshape([-1])}
        data = pd.DataFrame(data)
        sns.set(style="darkgrid", font_scale=0.8)
        sns.lineplot(data=data, x="Iteration", y="AvgReturn")
        plt.show()

        fig = plt.figure()
        data = {"Iteration": np.concatenate(steps + steps), "AvgReturn": np.concatenate(task_return_post + task_return_pre), "type": ["post policy"]*len(avg_return_post)*num_tasks + ["meta policy"]*len(avg_return_pre)*num_tasks}
        data = pd.DataFrame(data)
        sns.set(style="darkgrid", font_scale=0.8)
        sns.lineplot(data=data, x="Iteration", y="AvgReturn", hue="type")

        plt.show()
    else:

        for task in range(0, 4):
            a = np.load("ob_{}.npy".format(task))
            print(a.shape)
            pre, post = a
            show_all = 0
            start = 0
            length = a.shape[2] // 4
            if show_all:
                set_fig()
                plot_all_traj(length, pre, post)
            else:
                for i in range(start+0, start+2):
                    set_fig()
                    plot_single_traj(i, length, pre, post)

            plt.show()