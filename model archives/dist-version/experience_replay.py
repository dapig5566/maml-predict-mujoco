import numpy as np
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class ReplayBuffer:
    def __init__(self, max_buffer_size, ob_dim, action_dim, context_dim):
        self.start_sample_size = 400
        self.max_buffer_size = max(max_buffer_size, 1000)
        self.ob_dim = ob_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.buffer = None
        self.segment = None
        self.size = None
        self.length = [self.ob_dim, self.action_dim, 1, self.ob_dim, 1, self.context_dim, self.context_dim]

    def initialize(self):
        self.buffer = np.empty([self.max_buffer_size, self.ob_dim + self.action_dim + 1 + self.ob_dim + 1], np.float32)
        self.segment = [0, -1]
        self.size = 0

    def clear_buffer(self):
        self.segment = [0, -1]
        self.size = 0

    def push_back(self, transition):
        assert isinstance(transition, (list, tuple)) and len(transition) >= 5, "invalid transition format. [o, a, r, o', t, ...] is expected."
        assert self.buffer is not None and self.segment is not None and self.size is not None, "invalid buffer and pointer."
        transition = self.preprocess(transition)
        if transition.ndim == 1:
            transition = transition[None, :]

        length = transition.shape[0]
        if self.size + length > self.max_buffer_size:
            move = (self.size + length) - self.max_buffer_size
            self.segment[0] = (self.segment[0] + move) % self.max_buffer_size
            self.size -= move

        self.segment[1] = (self.segment[1] + 1) % self.max_buffer_size
        if self.segment[1] + length > self.max_buffer_size:
            l1 = self.max_buffer_size - 1 - self.segment[1]
            l2 = length - l1
            self.buffer[self.segment[1]:self.segment[1] + l1] = transition[:l1]
            self.buffer[:l2] = transition[l1:]
            self.segment[1] = l2 - 1
        else:
            self.buffer[self.segment[1]:self.segment[1] + length] = transition
            self.segment[1] = (self.segment[1] + length - 1) % self.max_buffer_size
        self.size += length
        return self.size

    def sample(self, batch_size):
        assert self.size >= self.start_sample_size, "cannot sample before minimum size reached."
        assert self.buffer is not None and self.segment is not None and self.size is not None, "invalid buffer and pointer."
        indices = np.random.randint(0, self.size, batch_size)
        return np.array(self.buffer[indices])

    def preprocess(self, transition):
        transition = [np.expand_dims(i, axis=2) if i.ndim == 2 else i for i in transition]
        transition = np.concatenate(transition, axis=-1)
        terminal = transition[..., sum(self.length[:4])]
        transitions = []
        for i, j in zip(transition, terminal):
            end = np.where(j)[0][0] if np.where(j)[0].size > 0 else j.shape[0]
            transitions.append(i[:end + 1])
        return np.concatenate(transitions)

    def __len__(self):
        return self.size


class MultitaskReplayBuffer(ReplayBuffer):
    def __init__(self, num_tasks, **kwargs):
        self.num_tasks = num_tasks
        self.task_buffers = None
        self.sizes = None
        self.segments = None
        super(MultitaskReplayBuffer, self).__init__(**kwargs)
        self.initialize()

    def initialize(self):
        self.task_buffers = [np.empty([self.max_buffer_size, sum(self.length)], np.float32) for _ in range(self.num_tasks)]
        self.sizes = [0 for _ in range(self.num_tasks)]
        self.segments = [[0, -1] for _ in range(self.num_tasks)]

    def clear_buffer(self, task_id):
        self.sizes[task_id] = 0
        self.segments[task_id] = [0, -1]

    def set_task(self, task_id):
        self.buffer = self.task_buffers[task_id]
        self.segment = self.segments[task_id]
        self.size = self.sizes[task_id]

    def push_back(self, task_id, transition):
        self.set_task(task_id)
        size = super().push_back(transition)
        self.sizes[task_id] = size

    def sample(self, task_id, batch_size, num_output=7):
        if isinstance(task_id, int):
            task_id = [task_id]
        assert hasattr(task_id, '__iter__'), "task_id must be iterable."
        batch = []
        for id in task_id:
            self.set_task(id)
            batch.append(super().sample(batch_size))
        batch = np.stack(batch)[..., :sum(self.length[:num_output])]
        return batch


class MultitaskTrajectoryBuffer:
    def __init__(self, num_tasks, max_buffer_size, ob_dim, action_dim, context_dim):
        self.ob_dim = ob_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.max_buffer_size = max_buffer_size
        self.num_tasks = num_tasks
        self.length = [self.ob_dim, self.action_dim, 1, self.ob_dim, 1]
        self.task_buffers = [np.empty([self.max_buffer_size, FLAGS.inner_time_steps, sum(self.length)], np.float32) for _ in range(self.num_tasks)]
        self.task_traj_lengths = [np.empty([self.max_buffer_size], np.int) for _ in range(self.num_tasks)]
        self.task_buf_sizes = [0 for _ in range(self.num_tasks)]
        self.task_buf_ptrs = [0 for _ in range(self.num_tasks)]

    def clear_buffer(self):
        self.task_buffers = [[] for _ in range(self.num_tasks)]

    def sample(self, task_ids, batch_size):
        samples = []
        indexs = []
        for id in task_ids:
            index = np.random.randint(0, self.task_buf_sizes[id], batch_size)
            index2 = [np.random.randint(self.task_traj_lengths[id][i]) for i in index]
            samples.append(self.task_buffers[id][index])
            indexs.append(index2)
        return np.stack(samples), np.stack(indexs)

    def push_back(self, task_id, transition):
        """[o, a, r, no, t]"""
        transition = [np.expand_dims(i, axis=2) if i.ndim == 2 else i for i in transition]
        transition = np.concatenate(transition, axis=-1)
        terminal = transition[..., -1]
        lengths = []
        ptr = self.task_buf_ptrs[task_id]
        buf = self.task_buffers[task_id][ptr:ptr+transition.shape[0]]
        for b, i, j in zip(buf, transition, terminal):
            # tmp = []
            # i = i.reshape([-1, FLAGS.inner_time_steps // 4, sum(self.length)])
            # j = j.reshape([-1, FLAGS.inner_time_steps // 4])
            # for s1, s2 in zip(i, j):
            #     end = np.where(s2)[0][0] if np.where(s2)[0].size > 0 else s2.shape[0]
            #     tmp.append(s1[:end + 1])
            # tmp = np.concatenate(tmp)
            # b[:len(tmp)] = tmp
            # lengths.append(len(tmp))
            # end = np.where(j)[0][0] if np.where(j)[0].size > 0 else j.shape[0]-1
            # end = j.shape[0]-1
            end = np.where(j)[0][3]
            assert end==j.shape[0]-1, np.where(j)
            b[:end+1] = i[:end + 1]
            lengths.append(end+1)
        self.task_traj_lengths[task_id][ptr:ptr+transition.shape[0]] = np.array(lengths)
        self.task_buf_ptrs[task_id] = (self.task_buf_ptrs[task_id] + len(transition)) % self.max_buffer_size
        self.task_buf_sizes[task_id] = min(self.task_buf_sizes[task_id] + len(transition), self.max_buffer_size)
