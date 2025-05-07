import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random

from datasets.dataset_ACDC import ACDCdataset, RandomGenerator
from networks.CAAFNet import CAAFNet
import numpy as np
from tqdm import tqdm
from medpy.metric import dc, hd95

from utils import DiceLoss

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=12, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=150)
parser.add_argument("--img_size", default=224)
parser.add_argument('--base_lr', type=float,  default=0.05,
                    help='segmentation network learning rate')
parser.add_argument("--save_path", default="./model_pth/ACDC")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="F:/yuyifenge/Pytorch-UNet-master/data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="F:/yuyifenge/Pytorch-UNet-master/data/ACDC/")
parser.add_argument("--volume_path", default="DF:/yuyifenge/Pytorch-UNet-master/data/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if not args.deterministic:
#     cudnn.benchmark = True
#     cudnn.deterministic = False
# else:
#     cudnn.benchmark = False
#     cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
def inference(net):
    logging.info("Validation ===>")
    dc_sum = 0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(
            torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.to(device).unsqueeze(1), val_label_batch.to(
            device).unsqueeze(1)
        val_outputs = net(val_image_batch)

        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)

        dc_sum += dc(val_outputs.cpu().data.numpy(), val_label_batch[:].cpu().data.numpy())
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f, ' % (performance))

    # print("val avg_dsc: %f" % (performance))
    return performance

net = CAAFNet(num_classes=args.num_classes) # model initialization for TransCASCADE

# net = PVT_CASCADE(n_class=config_vit.n_classes).cuda() # model initialization for PVT-CASCADE. comment above two lines if use PVT-CASCADE


if args.checkpoint:
    net.load_state_dict(torch.load(args.checkpoint))

train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
transforms.Compose(
    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
print("The length of train set is: {}".format(len(train_dataset)))
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val = ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader = DataLoader(db_val, batch_size=1, shuffle=False)
db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)
base_lr = args.lr
if args.n_gpu > 1:
    net = nn.DataParallel(net)

net = net.to(device)
net.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
# optimizer = optim.SGD(
#         net.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001
#     )
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)
iterator = tqdm(range(0, args.max_epochs), ncols=70)
iter_num = 0
max_epoch = args.max_epochs
max_iterations = args.max_epochs * len(Train_loader)
val=0
n_train = len(Train_loader)
for epoch_num in range(max_epoch):
    with tqdm(total=n_train, desc=f'Epoch {epoch_num}/{max_epoch}', unit='img', leave=True) as pbar:
        for i_batch, sampled_batch in enumerate(Train_loader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            outputs = net(image_batch)  # forward
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice  # current setting is for additive aggregation.

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            pbar.update(1)
            pbar.set_postfix(**{'loss': loss})
            # writer.add_scalar('info/lr', lr_, iter_num)
            # writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))


        if iter_num % 2 == 0:
            # val=evaluate(model,val_loader,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),amp=False)
            val = inference(net)
            logging.info(
                'best_val: {}'.format(best_inference))

    if val > best_inference:
        best_inference = val





