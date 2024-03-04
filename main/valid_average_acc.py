import _init_paths

# from config import cfg, update_config
from eccl_train import args
from dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
from eccl_train import data_transforms
from dataset.imbalanced_cifar import *
from dataset.imbalanced_cifar_contrastive import *
from network.networks import resnet32

import copy


def parse_args():
    parser = argparse.ArgumentParser(description="evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="/home/lijun/papers/Long_Tailed/configs/cifar100_im100_NCL.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def my_shot_acc(predict, label, many_shot_thr, low_shot_thr, train_class_dict):
    predict = predict.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    class_num = len(train_class_dict)
    class_correct = np.zeros(class_num)
    train_class_sum = np.zeros(class_num)
    test_class_sum = np.zeros(class_num)
    for i in range(class_num):
        class_correct[i] = (predict[label == i] == label[label == i]).sum()
        train_class_sum[i] = train_class_dict[i]
        test_class_sum[i] = len(label[label == i])

    many_shot_correct = 0
    many_shot_all = 0

    median_shot_correct = 0
    median_shot_all = 0

    few_shot_correct = 0
    few_shot_all = 0

    for i in range(class_num):
        if train_class_sum[i] >= many_shot_thr:
            many_shot_correct += class_correct[i]
            many_shot_all += test_class_sum[i]
        elif train_class_sum[i] <= low_shot_thr:
            few_shot_correct += class_correct[i]
            few_shot_all += test_class_sum[i]
        else:
            median_shot_correct += class_correct[i]
            median_shot_all += test_class_sum[i]

    print('{:>5.2f}\t{:>5.2f}\t{:>5.2f}\t{:>5.2f}'.format(many_shot_correct / many_shot_all * 100, \
                                                          median_shot_correct / median_shot_all * 100, \
                                                          few_shot_correct / few_shot_all * 100, \
                                                          (many_shot_correct + median_shot_correct + few_shot_correct) / (
                                                                  many_shot_all + median_shot_all + few_shot_all) * 100))


def valid_model(dataLoader, model, cfg, device, train_class_dict):
    model.eval()
    network_num = args.num_exps
    every_network_predict = [[] for _ in range(network_num)]
    every_network_logits = [[] for _ in range(network_num)]
    every_network_feature = [[] for _ in range(network_num)]

    average_predict = []

    all_label = []
    with torch.no_grad():

        for i, (image, label) in tqdm(enumerate(dataLoader)):

            image, label = image.to(device), label.to(device)
            all_label.append(label.cpu())

            outputs, embeddings = model(image, True)  # 计算模型输出

            sum_result = copy.deepcopy(outputs[0])  # 对模型输出进行深拷贝
            for k in range(network_num):
                if k > 0:
                    sum_result += outputs[k]
            average_predict.append(sum_result.argmax(dim=1).cpu())

            for j, logit in enumerate(outputs):
                every_network_logits[j].append(logit)
                every_network_predict[j].append(torch.argmax(logit, dim=1).cpu())

    all_label = torch.cat(all_label)

    average_predict = torch.cat(average_predict)

    average_acc = torch.sum(average_predict == all_label) / all_label.shape[0]
    print('average_acc: {}'.format(average_acc))

    print('average')
    my_shot_acc(average_predict, all_label, 100, 20, train_class_dict)

    for i in range(network_num):
        every_network_predict[i] = torch.cat(every_network_predict[i])
        print('network {}'.format(i))
        my_shot_acc(every_network_predict[i], all_label, 100, 20, train_class_dict)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    test_set = datasets.CIFAR100(
        root='/home/og/XieSH/dataset/long-tailed/public', train=False, download=True, transform=data_transforms['test'])  # 生成测试集
    # 构建训练集，控制是否构建memory bank的对比学习数据集
    if args.contrastive_dataset:
        train_set = IMBALANCECIFAR100_CONTRASTIVE(root='/home/og/XieSH/dataset/long-tailed/public', imb_factor=0.02,
                                                  rand_number=0, train=True, transform=data_transforms['advanced_train'],
                                                  args=args, is_sample=True)
    else:
        train_set = IMBALANCECIFAR100(root='/home/og/XieSH/dataset/long-tailed/public', imb_factor=0.02,
                                      rand_number=0, train=True, transform=data_transforms['advanced_train'])  # 生成训练集
    num_classes = len(test_set.class_to_idx)  # 测试集类别总数
    device = torch.device("cuda")  # GPU设备
    model = resnet32(num_classes=num_classes, use_norm=True,
                     num_exps=args.num_exps).to(device)  # 生成模型

    model_dir = '/home/og/XieSH/models/SHIKE/output/CIFAR100-LT-IF100/models/2024-01-20-15-05(57.72)/best_stage2.pth'  # 模型参数保存路径

    model.load_model(model_dir)  # 加载模型参数

    # model = torch.nn.DataParallel(model).cuda()  # DP多卡并行

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=16)  # 生成测试集加载器

    valid_model(test_loader, model, args, device, train_set.num_per_cls_dict)  # 开始测试模型准确率
