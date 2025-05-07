import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from datasets.dataset_synapse import Synapse_dataset
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F
best_dice = 0.0

#
### test ###
def trainer_synapse(args, model, snapshot_path):
    # 日志配置
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # 数据集
    x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    y_transforms = transforms.ToTensor()
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               img_size=args.img_size, norm_x_transform=x_transforms, norm_y_transform=y_transforms)

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)

    # 模型配置
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.to(device="cuda:0")
    model.train()

    # 损失函数和优化器
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)  # 加入平滑项
    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num=0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    best_inference = 0.0
    best_epoch =0
    loss_list = []
    dice_list = []
    for epoch_num in range(max_epoch):
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch_num}/{max_epoch}', unit='img', leave=True) as pbar:
            for sampled_batch in trainloader:
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()

                # 检查输入是否有效
                if torch.isnan(image_batch).any() or torch.isinf(image_batch).any():
                    logging.warning("输入数据包含 NaN 或 Inf，跳过当前迭代")
                    continue

                # 前向传播

                outputs = model(image_batch)

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice

                # 检查损失是否有效

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()


                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                # Adjust learning rate
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num += 1

            # 验证
            if epoch_num % 1== 0:
                val = inference(model, test_save_path=None,epoch = epoch_num)
                loss_list.append(loss.item())
                dice_list.append(val.item())
                logging.info(f"epoch:{epoch_num}, Validation result: {val}, Best: {best_inference} epoch{best_epoch}")
                if val > best_inference:
                    best_epoch = epoch_num
                    best_inference = val

    df = pd.DataFrame({
        'epoch': list(range(max_epoch)),
        'loss': loss_list,
        'dice': dice_list
    })
    df.to_csv(snapshot_path + '/uc_syn.csv', index=False)

    writer.close()
    return "Training Finished!"
#
#     logging.info('Overall Performance: Mean Dice = %.4f, Mean HD95 = %.4f' % (performance, mean_hd95))
#
#     # 找出最优类别
#     best_class_dice = np.argmax(average_dice) + 1
#     best_class_hd95 = np.argmin(average_hd95) + 1
#     logging.info(f'Best Class for Dice: {best_class_dice} with Dice = {average_dice[best_class_dice - 1]:.4f}')
#     logging.info(f'Best Class for HD95: {best_class_hd95} with HD95 = {average_hd95[best_class_hd95 - 1]:.4f}')
#
#     return performance
def inference(model, test_save_path=None,epoch=None):
    db_test = Synapse_dataset(base_dir='F:/yuyifenge/MISSFormer-main1/data/Synapse/test_vol_h5', split="test_vol", img_size=224,list_dir='./lists/lists_Synapse')

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0,drop_last=True)
    #logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    num_val_batches=len(testloader)
    with tqdm(total=num_val_batches, desc='Validation round', unit='person', position=0, leave=True) as pbar:
        # for i_tch, sampled_batch in tqdm(enumerate(testloader)):
        for i,sampled_batch in enumerate(testloader):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"].squeeze(1).cuda(), sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=9, patch_size=[224, 224],
                                          test_save_path=test_save_path, case=case_name, z_spacing=1,path=i,epoch=epoch)

            metric_list += np.array(metric_i)
            pbar.update(1)
            pbar.set_postfix(**{'dice': np.mean(metric_i, axis=0)})

            # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list = metric_list / len(db_test)
        # for i in range(1, 9):
        #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))

    for i in range(0,8 ):

        print("class %d  = %f  (best =)" % (i+1, metric_list[i]))
    performance = np.mean(metric_list, axis=0)
    #mean_hd95 = np.mean(metric_list, axis=0)[1]
    #mean_hd95 = np.mean(metric_list, axis=0)

    logging.info('Testing performance in  val model : mean_dice : %f ' % (performance))
    return performance