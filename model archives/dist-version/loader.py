import numpy as np
import PIL.Image as pmg
import os


class DataManager:
    def __init__(self, dir, n_way, k_shot):
        self.dir = dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.size = 28
        self.channels = 1
        language_folders = [os.path.join(dir, lang_folder) for lang_folder in os.listdir(dir)]
        char_folders = [os.path.join(lang_folder, char_folder)
                        for lang_folder in language_folders
                        if os.path.isdir(lang_folder)
                        for char_folder in os.listdir(lang_folder)
                        if os.path.isdir(os.path.join(lang_folder, char_folder))]
        classes = []
        for path in char_folders:
            files = [os.path.join(path, i) for i in os.listdir(path)]
            images = np.stack([np.asarray(pmg.open(file)).reshape(-1) for file in files])
            classes.append(images)
        self.classes = np.stack(classes)
        self.devide_validation()

    def devide_validation(self):
        self.train = self.classes[:-300, ...]
        self.validation = self.classes[-300:, ...]

    def sample_tasks(self, num_tasks, set="train", shuffle=False):
        if set == "train":
            set = self.train
        elif set == "validation":
            set = self.validation
        else:
            raise ValueError("no such set named {}".format(set))
        class_ids = np.array([np.random.choice(set.shape[0], [self.n_way], replace=False) for _ in range(num_tasks)])
        shots = np.array([np.random.choice(set.shape[1], [self.k_shot + 1], replace=False) for _ in range(num_tasks * self.n_way)]).reshape([num_tasks, self.n_way, -1])
        labels = np.eye(self.n_way).repeat(self.k_shot + 1, axis=0).reshape([self.n_way, self.k_shot + 1, self.n_way])
        samples = [np.concatenate([np.stack([set[i][j] for i, j in zip(class_id, shot)]), labels], axis=-1)
                   for class_id, shot in zip(class_ids, shots)]
        samples = np.stack(samples)
        if shuffle:
            for i in samples:
                np.random.shuffle(i[:, :self.k_shot, ...])
                np.random.shuffle(i[:, -1, ...])
        d_train, d_test = np.split(samples, [self.k_shot], axis=2)
        d_train_input, d_train_label = np.split(d_train, [self.size**2], axis=-1)
        d_test_input, d_test_label = np.split(d_test, [self.size**2], axis=-1)
        return [d_train_input, d_train_label, d_test_input, d_test_label]


# if __name__=="__main__":
#     dm = DataManager("datasets/omniglot_resized", 5, 1)
#     t1, t2 = dm.sample_tasks(10)
#     print(t1.shape)
#     print(t2.shape)
#     t1, t2 = dm.sample_tasks(10, "validation")
#     print(t1.shape)
#     print(t2.shape)



