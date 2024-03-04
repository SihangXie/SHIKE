import argparse
import os
from datetime import datetime
import _init_paths
import matplotlib.pyplot as plt

from utils.utils import get_logger
import warnings
import numpy as np

import torch.optim as optim
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.cuda.amp import autocast as autocast, GradScaler
from dataset.autoaugment import CIFAR10Policy, Cutout
from dataset.imbalanced_cifar import *
from dataset.imbalanced_cifar_contrastive import *
from network.networks import resnet32
from utils.util import *
from dataset.class_sampler import MPerClassSampler
from loss.imagenet_sup_mcl_loss import Sup_MCL_Loss

# data transform settings
normalize = transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
data_transforms = {
    'base_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    # augmentation adopted in balanced meta softmax & NCL
    'advanced_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
}

parser = argparse.ArgumentParser(description='ECCL CIFAR-LT Training')
parser.add_argument('--cfg_name', default='ECCL CIFAR100-LT-IF100', help='name of this configuration')
parser.add_argument('--output_dir', default='/home/og/XieSH/models/SHIKE/output', help='folder to output images and model checkpoints')
parser.add_argument('--pre_epoch', default=0, help='epoch for pre-training')
parser.add_argument('--epochs', default=400, help='epoch for augmented training')
parser.add_argument('--batch_size', default=128)
parser.add_argument('--learning_rate', default=0.05)
parser.add_argument('--seed', default=3407, help='keep all seeds fixed')
parser.add_argument('--re_train', default=True, help='implement cRT')
parser.add_argument('--cornerstone', default=380)
parser.add_argument('--num_exps', default=3, help='number of experts')
parser.add_argument('--distributed', default=False, help='use distributed data parallel training or not')  # 是否启用分布式训练
parser.add_argument('--local_rank', default=0, type=int, help='local_rank for distributed training')  # DDP训练的进程ID
parser.add_argument('--world_size', default=4, type=int, help='number of processes for distributed training')  # DDP训练的进程总数
parser.add_argument('--save_step', default=50, type=int, help='number of processes for distributed training')  # DDP训练的进程总数
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')  # 对比学习时embedding维度
parser.add_argument('--contrastive_sampler', default=False, type=bool, help='m per class sampler for mcl contrastive learning')  # 对比学习正负样本采样器
parser.add_argument('--tensorboard_enable', default=True, type=bool, help='enable tensorboard or not')  # 是否启用TensorBoard记录关键数据

parser.add_argument('--kd_T', default=3., type=float, help='temperature of KL-divergence')  # 对比学习时embedding维度
parser.add_argument('--tau', default=0.1, type=float, help='temperature for contrastive distribution')
parser.add_argument('--alpha', default=0.1, type=float, help='weight balance for VCL')
parser.add_argument('--gamma', default=1., type=float, help='weight balance for Soft VCL')
parser.add_argument('--beta', default=0.1, type=float, help='weight balance for ICL')
parser.add_argument('--lam', default=1., type=float, help='weight balance for Soft ICL')
parser.add_argument('--rep_dim', default=64, type=int, help='penultimate dimension')
parser.add_argument('--pos_k', default=1, type=int, help='number of positive samples for NCE')
parser.add_argument('--neg_k', default=8192, type=int, help='number of negative samples for NCE')
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
parser.add_argument('--contrastive_dataset', default=True, type=bool, help='whether apply memory bank dataset or not')
parser.add_argument('--dis_align', default=False, type=bool, help='whether apply DisAlign in classifier retrain stage or not')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    default=False,
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p',
                    '--print-freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
args = parser.parse_args()


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    rank = args.local_rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(args=args, rank=rank)
    model_dir = os.path.join(args.output_dir, args.cfg_name, 'models',
                             str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
    if not os.path.exists(model_dir) and rank == 0:
        os.makedirs(model_dir)
    if rank == 0:
        print('====> output model will be saved in {}'.format(model_dir))

    tensorboard_dir = (  # 定义TensorBoard的存放路径
        os.path.join(args.output_dir, args.cfg_name, "tensorboard",
                     str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
        if args.tensorboard_enable  # 如果配置文件中打开了tensorboard，就设置tensorboard的存储路径
        else None
    )
    if tensorboard_dir is not None and (not os.path.exists(tensorboard_dir)) and rank == 0:
        os.makedirs(tensorboard_dir)  # 不存在就创建文件夹
        writer = SummaryWriter(log_dir=tensorboard_dir)  # 创建TensorBoard记录器对象
    else:
        writer = None
    if rank == 0:
        print('====> tensorboard logs will be saved in {}'.format(tensorboard_dir))

    # ----- BEGIN INITIALIZE RANDOM SEED -----
    if args.distributed:
        set_seed(args.seed + rank)
    else:
        # pass
        set_seed(args.seed)
    # ----- END INITIALIZE RANDOM SEED -----

    if args.distributed:
        if rank == 0:
            print("Init the process group for distributed training")
        torch.cuda.set_device(rank)
        ddp_setup(rank=rank, world_size=args.world_size)

        if rank == 0:
            print("DDP progress group initialized successfully")

    # imbalance distribution
    # img_max * (imb_factor ** (cls_idx / (cls_num - 1.0))) 不平衡度为100
    num = np.array([int(np.floor(500 * (0.01 ** (i / (100 - 1.0)))))  # `**`是幂运算
                    for i in range(100)])
    args.label_dis = num  # 每类的样本数

    # 构建训练集，控制是否构建memory bank的对比学习数据集
    if args.contrastive_dataset:
        train_set = IMBALANCECIFAR100_CONTRASTIVE(root='/home/og/XieSH/dataset/long-tailed/public', imb_factor=0.01,
                                                  rand_number=0, train=True, transform=data_transforms['advanced_train'],
                                                  args=args, is_sample=True)
    else:
        train_set = IMBALANCECIFAR100(root='/home/og/XieSH/dataset/long-tailed/public', imb_factor=0.01,
                                      rand_number=0, train=True, transform=data_transforms['advanced_train'])

    # 定义训练集采样器
    if args.distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=rank)
    elif args.contrastive_sampler:
        train_sampler = MPerClassSampler(labels=train_set.idx_targets, m=2,
                                         batch_size=args.batch_size,
                                         length_before_new_iter=len(train_set.targets))
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               pin_memory=True,
                                               num_workers=16,
                                               sampler=train_sampler)

    test_set = datasets.CIFAR100(
        root='/home/og/XieSH/dataset/long-tailed/public', train=False, download=True, transform=data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=16)
    logger.info('size of train set data:{}'.format(train_set.__len__()))
    logger.info('size of test set data:{}'.format(test_set.__len__()))
    args.n_data = len(train_set)

    best_epoch, start_epoch = 0, 1
    best_acc1 = .0  # 最佳精度1

    # ----- BEGIN MODEL BUILDER -----
    num_classes = train_set.cls_num
    model = resnet32(num_classes=num_classes, use_norm=True,
                     num_exps=args.num_exps).to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # ----- BEGIN LOSS BUILDER -----
    criterion_mcl = Sup_MCL_Loss(args).to(device)  # 初始化MCL损失函数

    # optimizers and schedulers for decoupled training
    trainable_list = nn.ModuleList([])  # 可学习的网络参数列表
    trainable_list.append(model)
    trainable_list.append(criterion_mcl.embed_list)
    optimizer_feat = optim.SGD(  # 骨干特征提取网络的优化器
        trainable_list.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer_crt = optim.SGD(  # 分类器的优化器
        trainable_list.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler_feat = CosineAnnealingLRWarmup(  # 骨干特征提取网络的学习率调度器
        optimizer=optimizer_feat,
        T_max=args.epochs - 20,
        eta_min=0.0,
        warmup_epochs=5,
        base_lr=args.learning_rate,
        warmup_lr=0.20
    )
    scheduler_crt = CosineAnnealingLRWarmup(  # 分类器的学习率调度器
        optimizer=optimizer_crt,
        T_max=20,
        eta_min=0.0,
        warmup_epochs=5,
        base_lr=args.learning_rate,
        warmup_lr=0.1
    )

    criterion = nn.CrossEntropyLoss().cuda()  # 损失函数经典交叉熵损失

    if args.evaluate:
        validate(test_loader, model, criterion, 180, args, logger, writer)
        return

    # proceeding with torch apex
    scaler = GradScaler()  # 创建GradScaler对象，自动混合精度，提速

    for epoch in range(0, args.epochs):  # args.start_epoch
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # freezing shared parameters
        if epoch >= args.cornerstone:  # 从第180个epoch开始冻结所有共享参数s
            if args.distributed:
                for name, param in trainable_list.named_parameters():
                    if "rt_classifiers" not in name:  # DDP NAME module.classifiers
                        param.requires_grad = False  # 除了分类器，其他层的参数全部冻结
            else:
                for name, param in trainable_list.named_parameters():
                    if name[2:16] != "rt_classifiers":  # DDP NAME module.classifiers
                        param.requires_grad = False  # 除了分类器，其他层的参数全部冻结
        if epoch == args.cornerstone:
            if args.distributed:
                logger.info('====> rt_classifier parameter begin update: {}'.format(trainable_list[0].module.rt_classifiers[0].weight.requires_grad))
            else:
                logger.info('====> rt_classifier parameter begin update: {}'.format(trainable_list[0].rt_classifiers[0].weight.requires_grad))

            # if rank == 0:
            #     logger.info('====> trainable_list parameters: {}'.format(trainable_list.named_parameters()))

        # train for one epoch
        train(train_loader if epoch >= args.cornerstone else train_loader, model, scaler,
              optimizer_crt if epoch >= args.cornerstone else optimizer_feat, criterion_mcl,
              epoch, args, logger, writer, rank)

        # evaluate on validation set
        acc1 = validate(test_loader, model, criterion, epoch, args, logger, writer, rank)

        # adjust learning rate
        if epoch >= args.cornerstone:
            scheduler_crt.step()
            lr_dict = {'lr_rate': optimizer_crt.state_dict()['param_groups'][0]['lr']}
            writer.add_scalars('scalar/lr', lr_dict, epoch)
        else:
            scheduler_feat.step()
            lr_dict = {'lr_rate': optimizer_feat.state_dict()['param_groups'][0]['lr']}
            writer.add_scalars('scalar/lr', lr_dict, epoch)

        # record best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best and rank == 0:
            logger.info("Epoch {}, best acc1 = {}".format(epoch, best_acc1))

        # ----- SAVE MODEL -----
        if rank == 0:
            if args.distributed:
                state_dict = trainable_list.state_dict()
            else:
                state_dict = trainable_list.state_dict()
            save_checkpoint(
                {
                    'epoch': epoch,
                    'architecture': "resnet32",
                    'state_dict': state_dict,
                    'best_acc1': best_acc1,
                }, model_dir, is_best, feat=(epoch < args.cornerstone), epoch=epoch, args=args, rank=rank)
    if args.tensorboard_enable and rank == 0:
        writer.close()  # 关闭Tensorboard日志记录

    logger.info("Training Finished, TotalEPOCH=%d, Best Acc=%f" % (args.epochs, best_acc1))
    # plt.plot(range(0, 400), lr_list)
    # try:
    #     plt.savefig('learning_rate.pdf')
    #     print('Successfully saved figure')
    # except Exception as e:
    #     print(str(e))
    # plt.show()  # 显示学习率图片

    if args.distributed:
        destroy_process_group()


def mix_outputs(logits, embeddings, criterion_mcl, labels, pos_idx, neg_idx, balance=False, label_dis=None):
    logits_rank = logits[0].unsqueeze(1)  # 在第二维上插入1个新的维度作为logits排名
    for i in range(len(logits) - 1):  # i取值(0, 1)
        logits_rank = torch.cat(  # 水平方向拼接logits排名(bs, 3, 100)
            (logits_rank, logits[i + 1].unsqueeze(1)), dim=1)  # 其实就是把3个专家的分类头输出拼接到一个(bs, 3, 100)的张量

    max_tea, max_idx = torch.max(logits_rank, dim=1)  # 获取同一个类中，3个专家logits值最高的值与专家的索引作为教师模型teacher
    # min_tea, min_idx = torch.min(logits_rank, dim=1)

    non_target_labels = torch.ones_like(labels) - labels  # 获取non target labels，target logit值会变0，相当于掩码

    avg_logits = torch.sum(logits_rank, dim=1) / len(logits)  # 计算3个专家输出的logits的均值(沿专家维度算平均值)
    non_target_logits = (-30 * labels) + avg_logits * non_target_labels  # 【重要】×-30后target logit值会变-30.基于平均logits，计算non target logits

    _hardest_nt, hn_idx = torch.max(non_target_logits, dim=1)  # 【重要】计算`consensus hardest negative class`及其索引

    hardest_idx = torch.zeros_like(labels)  # 创建形状与真实标签相同的0值张量作为`consensus hardest negative class`的索引张量
    hardest_idx.scatter_(1, hn_idx.data.view(-1, 1), 1)  # 把`consensus hardest negative class`的索引位置置1，其余为0
    hardest_logit = non_target_logits * hardest_idx  # `consensus hardest negative class`的值

    rest_nt_logits = max_tea * (1 - hardest_idx) * (1 - labels)  # 排除target logit和consensus hardest negative class的剩余最大non target logits
    reformed_nt = rest_nt_logits + hardest_logit  # 【重要】得到全部non target logits的值作为教师模型

    preds = [F.softmax(logit) for logit in logits]  # 计算三个专家原始输出的prediction

    reformed_non_targets = []  # 初始化空列别
    for i in range(len(preds)):  # 循环3次
        target_preds = preds[i] * labels  # 获取target的prediction值

        target_preds = torch.sum(target_preds, dim=-1, keepdim=True)  # 将其他99个0值清除
        target_min = -30 * labels  # 还是没搞清楚为什么要乘-30？
        target_excluded_preds = F.softmax(  # 计算排除了target logit的prediction
            logits[i] * (1 - labels) + target_min)
        reformed_non_targets.append(target_excluded_preds)  # 把每个专家的non target prediction加到列表中

    label_dis = torch.tensor(  # 把每类的样本数转换成Tensor
        label_dis, dtype=torch.float, requires_grad=False).cuda()
    label_dis = label_dis.unsqueeze(0).expand(labels.shape[0], -1)  # 把label_dis形状从(100,)修改成(bs, 100)

    loss = 0.0

    targets = torch.argmax(labels, dim=1)
    if balance == True:  # epoch超过180进入该分支，表示训练分类器
        for i in range(len(logits)):  # 循环3次计算分类器损失，最后损失之求和
            loss += soft_entropy(logits[i] + label_dis.log(), labels)  # BSCE平衡Softmax CE
    else:  # epoch小于180进入该分支，表示训练backbone
        for i in range(len(logits)):  # 循环3次
            # base ce
            loss += soft_entropy(logits[i], labels)  # 计算每个专家的交叉熵损失函数
            # hardest negative suppression
            loss += 10.0 * \
                    F.kl_div(  # KL散度
                        torch.log(reformed_non_targets[i]), F.softmax(reformed_nt))  # 计算每个专家的non target logits与教师模型的KL散度
            # mutual distillation loss
            for j in range(len(logits)):  # DKF两两专家之间进行知识蒸馏
                if i != j:
                    loss += F.kl_div(F.log_softmax(logits[i]),
                                     F.softmax(logits[j]))
        # 计算MCL损失
        loss_vcl, loss_soft_vcl, loss_icl, loss_soft_icl = criterion_mcl(embeddings, pos_idx, neg_idx)
        loss_mcl = args.alpha * loss_vcl + args.gamma * loss_soft_vcl \
                   + args.beta * loss_icl + args.lam * loss_soft_icl
        loss += loss_mcl

    avg_output = sum(logits) / len(logits)  # 计算多专家原始平均输出
    return loss, avg_output


def train(train_loader, model, scaler, optimizer, criterion_mcl, epoch, args, logger, writer, rank=0):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # worst_case_per_round = None
    for i, (images, target, pos_idx, neg_idx) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        pos_idx = pos_idx.cuda().long()
        neg_idx = neg_idx.cuda().long()

        optimizer.zero_grad()
        # compute output
        with autocast():  # amp混合精度库前向传播
            logits, embeddings = model(images, (epoch >= args.cornerstone))  # 前向传播使用torch.float16(2字节)精度
            loss, output = mix_outputs(logits=logits, embeddings=embeddings,
                                       criterion_mcl=criterion_mcl,
                                       labels=target, pos_idx=pos_idx, neg_idx=neg_idx,
                                       balance=(epoch >= args.cornerstone),
                                       label_dis=args.label_dis)  # 计算损失值和3个专家的平均原始输出
        _, target = torch.max(target.data, 1)  # 获取正确标签的索引

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # 计算本批次的top1和top5准确率

        losses.update(loss.item(), images.size(0))  # 更新loss值计数器
        top1.update(acc1.item(), images.size(0))  # 更新top1准确率计数器
        top5.update(acc5.item(), images.size(0))  # 更新top5准确率计数器

        scaler.scale(loss).backward()  # 使用float32进行反向传播，避免使用float16精度梯度下溢
        scaler.step(optimizer)
        scaler.update()  # 更新网络参数

        # measure elapsed time
        batch_time.update(time.time() - end)  # 计算本batch训练耗时
        end = time.time()

        if i % args.print_freq == 0:  # 每100个batch打印一次信息
            progress.display(i, logger)

    # 添加TensorBoard日志信息
    loss_dict, acc_dict = {'train_loss': losses.avg}, {'train_acc': top1.avg}
    if args.tensorboard_enable and rank == 0:
        writer.add_scalars('scalar/acc', acc_dict, epoch)
        writer.add_scalars('scalar/loss', loss_dict, epoch)


def validate(val_loader, model, criterion, epoch, args, logger, writer, rank=0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=False)
            target = target.cuda(non_blocking=False)

            # compute output
            outputs, embeddings = model(images, (epoch >= args.cornerstone))
            output = sum(outputs) / len(outputs)  # 每一个类的logit等于3个专家的平均值

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, logger)

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    # 添加TensorBoard日志信息
    loss_dict, acc_dict = {'valid_loss': losses.avg}, {'valid_acc': top1.avg}
    if args.tensorboard_enable and rank == 0:
        writer.add_scalars('scalar/acc', acc_dict, epoch)
        writer.add_scalars('scalar/loss', loss_dict, epoch)
    return top1.avg


def save_checkpoint(state, model_dir, is_best, feat, epoch, args, rank=0):
    if epoch % args.save_step == 0 or epoch == args.epochs - 1:
        torch.save(state, os.path.join(model_dir, f'ckp_epoch_{epoch}.pth'))
    if is_best and feat:
        torch.save(state, os.path.join(model_dir, f'best_stage1.pth'))
    elif is_best:
        torch.save(state, os.path.join(model_dir, f'best_stage2.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)  # 最大top k数
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # 获取logit前5大的logit值的索引
        pred = pred.t()  # 矩阵转置
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 把target的形状扩展成pred再计算相等bool矩阵

        res = []
        for k in topk:  # 循环2次，分别计算top1和top5准确率
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)  # 切片操作后不连续，调用contiguous保证张量在内存里的连续
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # 指定可用的GPU
    warnings.filterwarnings("ignore")
    clock_start = datetime.now()
    main()
    clock_end = datetime.now()
    print(clock_end - clock_start)
