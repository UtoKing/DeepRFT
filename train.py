import argparse
from cmath import isnan
from torch.utils.tensorboard import SummaryWriter
import kornia
from dataset_RGB import DataLoaderTrain, DataLoaderVal
from get_parameter_number import get_parameter_number
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
import losses
from DeepFFTAttention import DeepFFTAttention as myNet
from data_RGB import get_training_data, get_validation_data
import utils
import numpy as np
import time
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import os
from sklearn.model_selection import train_test_split
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'

torch.backends.cudnn.benchmark = True


######### Set Seeds ###########
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

start_epoch = 1

parser = argparse.ArgumentParser(description='Image Deblurring')

parser.add_argument('--train_dir', default='./Datasets/GoPro/train',
                    type=str, help='Directory of train images')
parser.add_argument('--val_dir', default='./Datasets/GoPro/val',
                    type=str, help='Directory of validation images')
parser.add_argument('--model_save_dir', default='./checkpoints',
                    type=str, help='Path to save weights')
parser.add_argument('--pretrain_weights', default='./checkpoints/model_best.pth',
                    type=str, help='Path to pretrain-weights')
parser.add_argument('--mode', default='Deblurring', type=str)
parser.add_argument('--session', default='DeepRFT_gopro',
                    type=str, help='session')
parser.add_argument('--patch_size', default=256, type=int,
                    help='patch size, for paper: [GoPro, HIDE, RealBlur]=256, [DPDD]=512')
parser.add_argument('--num_epochs', default=3000, type=int, help='num_epochs')
parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
parser.add_argument('--val_epochs', default=20, type=int, help='val_epochs')
args = parser.parse_args()

mode = args.mode
session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, mode+"_" + 'models', session)
utils.mkdir(model_dir)

train_dir = args.train_dir
val_dir = args.val_dir

num_epochs = args.num_epochs
batch_size = args.batch_size
val_epochs = args.val_epochs

start_lr = 1e-4
end_lr = 5e-7

######### Model ###########
model_restoration = myNet(num_res=4)

# print number of model
get_parameter_number(model_restoration)

model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

optimizer = optim.Adam(model_restoration.parameters(),
                       lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, num_epochs-warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

RESUME = False
Pretrain = False
model_pre_dir = ''
######### Pretrain ###########
if Pretrain:
    utils.load_checkpoint(model_restoration, model_pre_dir)

    print('------------------------------------------------------------------------------')
    print("==> Retrain Training with: " + model_pre_dir)
    print('------------------------------------------------------------------------------')

######### Resume ###########
if RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(
        model_restoration, device_ids=device_ids)

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_fft = losses.fftLoss()
######### DataLoaders ###########

train_test_list = os.listdir(os.path.join(train_dir, "blur"))
train_list, test_list = train_test_split(
    train_test_list, shuffle=True, train_size=0.8)

train_dataset = DataLoaderTrain(
    train_dir, train_list, {'patch_size': patch_size})

# train_dataset = get_training_data(train_dir, {'patch_size': patch_size})
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=8, drop_last=False, pin_memory=True)

val_dataset = DataLoaderVal(train_dir, test_list, {'patch_size': patch_size})

# val_dataset = get_validation_data(val_dir, {'patch_size': patch_size})
val_loader = DataLoader(dataset=val_dataset, batch_size=16,
                        shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

print("===> Training Size "+str(len(train_dataset)) +
      " Test Size "+str(len(val_dataset)))

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(os.path.join(
    args.model_save_dir, mode+"_"+'models', session, current_time))
iter = 0

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target_ = data[0].cuda()
        input_ = data[1].cuda()
        target = kornia.geometry.transform.build_pyramid(target_, 3)
        restored = model_restoration(input_)

        loss_fft = criterion_fft(restored[0], target[0]) + criterion_fft(restored[1], target[1]) + criterion_fft(
            restored[2], target[2])
        loss_char = criterion_char(restored[0], target[0]) + criterion_char(
            restored[1], target[1]) + criterion_char(restored[2], target[2])
        loss_edge = criterion_edge(restored[0], target[0]) + criterion_edge(
            restored[1], target[1]) + criterion_edge(restored[2], target[2])
        loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        iter += 1
        writer.add_scalar('loss/fft_loss', loss_fft, iter)
        writer.add_scalar('loss/char_loss', loss_char, iter)
        writer.add_scalar('loss/edge_loss', loss_edge, iter)
        writer.add_scalar('loss/iter_loss', loss, iter)
    writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)
    #### Evaluation ####
    if epoch % val_epochs == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)

            for res, tar, file_name in zip(restored[0], target, data_val[2]):
                if torch.isnan(res).any():
                    print(file_name)
                psnr_mean = utils.torchPSNR(res, tar)
                psnr_val_rgb.append(psnr_mean)

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        writer.add_scalar('val/psnr', psnr_val_rgb, epoch)
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
              (epoch, psnr_val_rgb, best_epoch, best_psnr))

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
        epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

writer.close()
