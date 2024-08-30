import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.HybridSN import HybridSN

from utils.dataloader import *
from utils.metrics import output_metric
from utils.utils import *
from train import train_epoch,valid_epoch
from test import test_epoch

#-------------------------------------------------------------------------------
# setting the parameters
# model mode
mode = "train" # train or test
pretrained = False # pretrained or not
model_path = r"" # model path

# model settings
model_type = "HybridSN" 
patch_size = 33 

# training settings
gpu = "0"
epoch = 10
test_freq = 5
batch_size = 4
learning_rate = 5e-4
weight_decay = 0
gamma = 0.9

# data settings
train_ratio = 0.5 # percentage => percentage of samples(0-1) 
val_ratio = 0.1 # percentage => percentage of samples(0-1) 
data_mat_path = r"data\data_patches.mat" # data
label_mat_path = r"data\label_patches.mat" # label
#-------------------------------------------------------------------------------

# make the run folder in logs
#-------------------------------------------------------------------------------
time_now = time.localtime()
if model_type == "HybridSN":
    time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type
    logs_file = os.path.join(time_folder, "logs.txt")
    os.makedirs(time_folder)
else:
    raise ValueError("invalid model type!")
#-------------------------------------------------------------------------------

# time setting
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
cudnn.deterministic = True
cudnn.benchmark = False

# 定义一系列变换操作
transform = Compose([
    Normalize(min_val=0.0, max_val=1.0),
    RandomHorizontalFlip(p=0.5),
])

data_train, data_val, data_test, label_train, label_val, label_test, bands, num_classes = load_and_split_data(data_mat_path, label_mat_path, train_ratio, val_ratio, patch_size)

if model_type == "HybridSN":
    model = HybridSN(
        in_channels = bands, 
        patch_size = patch_size, 
        num_classes = num_classes,
    )
else:
    raise ValueError("invalid model type!")

# dataset and dataloder
#-------------------------------------------------------------------------------
train_dataset = HSI_dataset(data_train, label_train, transform)
val_dataset = HSI_dataset(data_val, label_val)
test_dataset = HSI_dataset(data_test, label_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#-------------------------------------------------------------------------------

# model settings
#-------------------------------------------------------------------------------
model = model.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//10, gamma=gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch//10, eta_min=5e-4)

#-------------------------------------------------------------------------------
if mode == "train":
    # if pretrained
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("load model path : " + model_path)
    # train
    print("===============================================================================")
    print("start training")
    tic = time.time()
    epoch_result = np.zeros([3, epoch])
    for e in range(epoch): 
        model.train()
        train_acc, train_loss, label_t, prediction_t = train_epoch(model, train_loader, criterion, optimizer, e, epoch)
        scheduler.step()
        OA_train, AA_train, Kappa_train, CA_train, CM_train = output_metric(label_t, prediction_t) 
        print("Epoch: {:03d} | train_loss: {:.4f} | train_acc: {:.4f}%".format(e+1, train_loss, train_acc))
        epoch_result[0][e], epoch_result[1][e], epoch_result[2][e] = e+1, train_loss, train_acc
        log_training_results(logs_file, mode='train', epoch_num=e+1, train_loss=train_loss, train_acc=train_acc)

        if ((e+1) % test_freq == 0) | (e == epoch - 1):
            print("===============================================================================")
            print("start validating")      
            model.eval()
            label_v, prediction_v = valid_epoch(model, val_loader, criterion)
            OA_val, AA_val, Kappa_val, CA_val, CM_val = output_metric(label_v, prediction_v)
            log_training_results(logs_file, mode='train', epoch_num=e+1, train_loss=train_loss, train_acc=train_acc,
                     OA_val=OA_val, AA_val=AA_val, Kappa_val=Kappa_val, CA_val=CA_val, CM_val=CM_val)
            if (e != epoch -1):
                print("Epoch: {:03d}  =>  OA: {:.4f}% | AA: {:.4f}% | Kappa: {:.4f}".format(e+1, OA_val, AA_val, Kappa_val))
            print("===============================================================================")

    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("end training")
    print("===============================================================================")
    print("Val result:")
    print("OA: {:.4f}% | AA: {:.4f}% | Kappa: {:.4f}".format(OA_val, AA_val, Kappa_val))
    print("CA:", end="")
    print(CA_val)
    print("Val Confusion Matrix:")
    print(CM_val)
    print("===============================================================================")

elif mode == "test":
    model.load_state_dict(torch.load(model_path))
    print("load model path : " + model_path)
    print("===============================================================================")

if mode == "train":
    # save model and its parameters 
    torch.save(model, time_folder + r"\\model.pt")
    torch.save(model.state_dict(), time_folder + r"\\model_state_dict.pth")

print("start testing")
model.eval()

label, prediction = test_epoch(model, test_loader, criterion)
OA_test, AA_test, Kappa_test, CA_test, CM_test = output_metric(label, prediction)
log_training_results(logs_file, mode='test', OA_test=OA_test, AA_test=AA_test, Kappa_test=Kappa_test, 
                     CA_test=CA_test, CM_test=CM_test)
print("Test result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA_test, AA_test, Kappa_test))
print("CA:", end="")
print(CA_test)
print("Test Confusion Matrix:")
print(CM_test)
print("===============================================================================")

print("end testing")
print("===============================================================================")
