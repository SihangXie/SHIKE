import torch
import torchvision
from PIL import Image
import numpy as np

from utils.util import to_categorical


class IMBALANCECIFAR10_CONTRASTIVE(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=True, args=None, is_sample=True):
        super(IMBALANCECIFAR10_CONTRASTIVE, self).__init__(root, train, transform, target_transform, download)
        self.idx_targets = []  # 类别ID格式的标签
        # print(rand_number)
        np.random.seed(0)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

        self.pos_k = args.pos_k  # 正样本数量
        self.neg_k = args.neg_k  # 负样本数量
        self.args = args
        self.is_sample = is_sample

        print('====> IMBALANCE CIFAR DATASET initialization stage 1 finished!')

        if self.is_sample:
            num_classes = self.cls_num
            self.num_samples = len(self.idx_targets)
            # 数据集所有样本的类别ID列表是self.idx_targets

            # 构建正样本，每个类的正样本就是同一个类中的样本
            self.cls_positive = [[] for i in range(self.cls_num)]
            for i in range(self.num_samples):
                self.cls_positive[self.idx_targets[i]].append(i)

            # 构造负样本，每个类的负样本就是除了同一个类中的样本以外的所有样本
            self.cls_negative = [[] for i in range(self.cls_num)]
            for i in range(self.cls_num):
                for j in range(self.cls_num):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            # 把正负样本索引转化成numpy对象列表
            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(self.cls_num)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(self.cls_num)]

        print('====> IMBALANCE CIFAR DATASET initialize successfully!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # sample contrastive examples
        neg_idx = np.random.choice(self.cls_negative[np.argmax(target)], self.neg_k, replace=True)  # 随机从Memory Bank中采集8192个负样本
        pos_idx = np.random.choice(self.cls_positive[np.argmax(target)], self.pos_k, replace=False)  # 随即从1300个同类样本中采集1个正样本
        pos_idx = np.hstack((index, pos_idx))  # 将anchor样本与正样本水平堆叠成1个ndarray
        return img, target, pos_idx, neg_idx

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        print(len(self.data))
        self.targets = to_categorical(new_targets)  # TODO 标签在此处被转化为one-hot格式
        self.idx_targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100_CONTRASTIVE(IMBALANCECIFAR10_CONTRASTIVE):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    trainset = IMBALANCECIFAR100_CONTRASTIVE(root='./data', train=True,
                                             download=True, transform=None, rand_number=0)
