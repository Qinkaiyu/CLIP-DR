import math
import os
import os.path as osp
import random
from collections import defaultdict

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import SequentialSampler

from ordinalclip.utils.logging import get_logger, print_log
from torch.utils.data import Sampler
from collections import defaultdict

from .utils import get_transforms
from torch.utils.data import BatchSampler

logger = get_logger(__name__)
print = lambda x: print_log(x, logger=logger)

class FixedNumSamplesBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, num_samples, replacement=True):
        """
        Args:
            data_source (Dataset): 数据集对象
            batch_size (int): 每个批次的样本数量
            num_samples (int): 每个训练周期中的总样本数量
            replacement (bool): 是否有放回采样
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_batches = num_samples // batch_size
        self.replacement = replacement

        if len(data_source) < num_samples:
            raise ValueError("num_samples should not exceed the length of data_source")

        # 统计每个类别的样本索引
        self.class_indices = defaultdict(list)
        for i, (_, label) in enumerate(data_source):
            self.class_indices[label].append(i)

    def __iter__(self):
        # 将每个类别的样本索引分配给不同的批次
        lengh = len(self.data_source) // self.batch_size
        batch = []

        for i in range(lengh):
            indice= []
            random.seed(i)
            output = self.data_source.split_dataset_by_label_index()
            num_samples_per_label = [[k, len(v)] for k, v in output[1].items()]
            num_samples_per_label.sort(key=lambda x: x[1], reverse=True)
            for idx, label_cnt in enumerate(num_samples_per_label):
                whatweneed = output[1]
                imgs_ls = whatweneed[label_cnt[0]]
                sampled_imgs_ls = random.sample(imgs_ls, 1)
                #print(sampled_imgs_ls)

                indice.extend(sampled_imgs_ls)
                #labels.extend([[label_cnt[0]]] * len(sampled_imgs_ls))
            for idx in indice:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    # 通过yield返回，下一个iter时清空batch继续采集
                    random.shuffle(batch)

                    yield batch
                    batch = []


    def __len__(self):
        return len(self.data_source) // self.batch_size

class RegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_images_root,
        val_images_root,
        test_images_root,
        train_data_file,
        val_data_file,
        test_data_file,
        transforms_cfg=None,
        train_dataloder_cfg=None,
        eval_dataloder_cfg=None,
        few_shot=None,
        label_distributed_shift=None,
        use_long_tail=False
    ):
        super().__init__()
        train_transforms, eval_transforms = get_transforms(**transforms_cfg)

        self.train_set = RegressionDataset(train_images_root, train_data_file, train_transforms)
        #self.batch_train_set = select_batch(self.train_set)
        self.val_set = RegressionDataset(val_images_root, val_data_file, eval_transforms)
        self.test_set = RegressionDataset(test_images_root, test_data_file, eval_transforms)

        self.train_set.generate_fewshot_dataset(**few_shot)
        self.train_set.generate_distribution_shifted_dataset(**label_distributed_shift)
        if use_long_tail:
            self.val_set.generate_long_tail()
            self.test_set.generate_long_tail()

        self.train_dataloder_cfg = train_dataloder_cfg
        self.eval_dataloder_cfg = eval_dataloder_cfg
    def setup(self, stage=None):
        self.batch_sampler = FixedNumSamplesBatchSampler(self.train_set, 5, 1)
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_sampler=self.batch_sampler, num_workers=10)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.eval_dataloder_cfg)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.eval_dataloder_cfg)

    def select_batch(self):
        output = self.train_set.split_dataset_by_label()
        print(output)
        print(self.train_set)

        num_samples_per_label = [[k, len(v)] for k, v in output.items()]
        num_samples_per_label.sort(key=lambda x: x[1], reverse=True)
        images_file = []
        labels = []
        image_indices = []
        for idx, label_cnt in enumerate(num_samples_per_label):
            imgs_ls = output[label_cnt[0]]
            sampled_imgs_ls = random.sample(imgs_ls, 1)
            selected_indices = [imgs_ls.index(img) for img in sampled_imgs_ls]
            image_indices.extend(selected_indices)
            images_file.extend(sampled_imgs_ls)
            labels.extend([[label_cnt[0]]] * len(sampled_imgs_ls))

        # batch_sampler = FixedNumSamplesBatchSampler(output, 20, 4)
        # print(batch_sampler)
        # batch_sampler = BatchSampler(torch.utils.data.RandomSampler(data_indices), batch_size=batch_size,
        #                      drop_last=True)
        print(images_file)
        print(labels)
        return images_file,labels

class RegressionDataset(Dataset):
    def __init__(self, images_root, data_file, transforms=None):
        self.images_root = images_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms

        with open(data_file) as fin:
            for line in fin:
                # image_file, image_label = line.split()
                splits = line.split()
                image_file = splits[0]
                labels = splits[1:]
                self.labels.append([int(label) for label in labels])
                self.images_file.append(image_file)

        self.name = osp.splitext(osp.basename(data_file))[0].lower()
        if "val" in self.name or "test" in self.name:
            print(f"Dataset prepare: val/test data_file: {data_file}")
        elif "train" in self.name:
            print(f"Dataset prepare: train data_file: {data_file}")
        else:
            raise ValueError(f"Invalid data_file: {data_file}")
        print(f"Dataset prepare: len of labels: {len(self.labels[0])}")
        print(f"Dataset prepare: len of dataset: {len(self.labels)}")

    def __getitem__(self, index):
        img_file, target_list = self.images_file[index], self.labels[index]
        if "val" in self.name or "test" in self.name:
            target = target_list[len(target_list) // 2]
        else:
            target = random.choice(target_list)

        full_file = os.path.join(self.images_root, img_file)
        img = Image.open(full_file)

        if img.mode == "L":
            img = img.convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def generate_long_tail(self):
        images_file_new, labels_new = [], []
        len_before = len(self.labels)
        for index in range(len_before):
            img_file, target_list = self.images_file[index], self.labels[index]
            if "val" in self.name or "test" in self.name:
                target = target_list[len(target_list) // 2]
            else:
                target = random.choice(target_list)
            if target >= 50:
                images_file_new.append(img_file)
                labels_new.append(target_list)
        
        self.images_file = images_file_new
        self.labels = labels_new
        len_after = len(self.labels)
        logger.info(f"generate long tail dataset, the change of # of samples: {len_before} -> {len_after}.")

    def get_label_dist(self, target):
        label_dist = [self.normal_sampling(int(target), i, std=self.std) for i in range(self.n_cls)]
        label_dist = [i if i > 1e-15 else 1e-15 for i in label_dist]
        label_dist = torch.Tensor(label_dist)

        return label_dist

    def __len__(self):
        return len(self.labels)

    def split_dataset_by_label(self):
        output = defaultdict(list)
        for img, label in zip(self.images_file, self.labels):
            target = label[len(label) // 2]
            output[target].append(img)
        return output
    def split_dataset_by_label_index(self):
        output = defaultdict(list)
        index_list = defaultdict(list)

        for index, (img, label) in enumerate(zip(self.images_file, self.labels)):
            target = label[len(label) // 2]
            output[target].append(img)
            index_list[target].append(index)

        return output, index_list
    def generate_fewshot_dataset(self, num_shots=-1, repeat=False):
        if num_shots <= 0:
            print("not generate few-shot dataset: num_shots<=0")
            return

        output = self.split_dataset_by_label()

        print("generate few-shot dataset")
        print("clear full dataset: images_file & labels")
        self._images_file = self.images_file
        self._labels = self.labels

        self.images_file = []
        self.labels = []
        print(
            f"build few_shot: num labels: {len(output.keys())}, {list(output.keys())[:5]}, ..., {list(output.keys())[-5:]}"
        )
        for label, imgs_ls in output.items():
            # self.images_file.extend(imgs_ls)
            # self.labels.extend([label] * len(imgs_ls))

            if len(imgs_ls) >= num_shots:
                sampled_imgs_ls = random.sample(imgs_ls, num_shots)
            else:
                print(f"not enough: class-{label}: {len(imgs_ls)}")
                if repeat:
                    sampled_imgs_ls = random.choices(imgs_ls, k=num_shots)
                else:
                    sampled_imgs_ls = imgs_ls

            self.images_file.extend(sampled_imgs_ls)
            self.labels.extend([[label]] * len(sampled_imgs_ls))
        assert len(self.images_file) == len(self.labels), f"{len(self.images_file)} != {len(self.lables)}"
        print(f"len of few shot dataset: {len(self.images_file)}")

    def generate_distribution_shifted_dataset(self, num_topk_scaled_class=-1, scale_factor=0.3):
        if num_topk_scaled_class <= 0:
            print("not generate distribution shifted dataset: num_topk_scaled_class<=1")
            return
        if scale_factor == 1.0:
            print("not generate distribution shifted dataset: scale_factor=1.0")
            return
        assert scale_factor > 0 and scale_factor < 1.0

        output = self.split_dataset_by_label()

        print("generate distribution shifted dataset")
        print("clear full dataset: images_file & labels")
        self._images_file = self.images_file
        self._labels = self.labels

        self.images_file = []
        self.labels = []

        print(
            f"build distribution shifed: num labels: {len(output.keys())}, {list(output.keys())[:5]}, ..., {list(output.keys())[-5:]}"
        )

        num_samples_per_label = [[k, len(v)] for k, v in output.items()]
        num_samples_per_label.sort(key=lambda x: x[1], reverse=True)
        print(num_samples_per_label)

        for idx, label_cnt in enumerate(num_samples_per_label):
            if idx < num_topk_scaled_class:
                imgs_ls = output[label_cnt[0]]
                sampled_imgs_ls = random.sample(imgs_ls, max(int(len(imgs_ls) * scale_factor), 1))
            else:
                sampled_imgs_ls = output[label_cnt[0]]

            self.images_file.extend(sampled_imgs_ls)
            self.labels.extend([[label_cnt[0]]] * len(sampled_imgs_ls))

        assert len(self.images_file) == len(self.labels), f"{len(self.images_file)} != {len(self.lables)}"
        print(f"len of distribution shifed dataset: {len(self.images_file)}")

    @staticmethod
    def normal_sampling(mean, label_k, std=2):
        return math.exp(-((label_k - mean) ** 2) / (2 * std**2)) / (math.sqrt(2 * math.pi) * std)
