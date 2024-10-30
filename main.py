import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.HybridSN import HybridSN
from models.CNN_3D1D import CNN_3D_Classifer_1D
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils.dataloader import *
from utils.metrics import output_metric
from utils.utils import *
from train import train_epoch,valid_epoch
from test import test_epoch

#-------------------------------------------------------------------------------
# setting the parameters
# model mode
mode = "train" # train or test
DP = False
pretrained = True # pretrained or not
model_path = r"logs/2024-09-24-16-20-26-3D_1D_CNN-train-5/model_state_dictl_e004_loss1.8600.pth" # model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model settings
model_type = "3D_1D_CNN" 
patch_size = 5 

# training settings
gpu = "0,1"
epoch = 50
test_freq = 2
batch_size = 64
learning_rate = 5e-4
weight_decay = 0
gamma = 0.9
ignore_index = 0
focal_loss = False

# data settings
train_ratio = 0.9 # percentage => percentage of samples(0-1) 
val_ratio = 0.05 # percentage => percentage of samples(0-1) 
data_type = "folder" # folder/file
data_mat_path = r"/home/ljs/HybridSN-pytorch/data/data" # data
label_mat_path = r"/home/ljs/HybridSN-pytorch/data/label" # label
#-------------------------------------------------------------------------------

# make the run folder in logs
#-------------------------------------------------------------------------------
time_now = time.localtime()
if model_type == "HybridSN":
    time_folder = os.path.join("logs" ,time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + mode + "-" + str(patch_size))
    logs_file = os.path.join(time_folder, "logs.txt")
    os.makedirs(time_folder)
elif model_type == "3D_1D_CNN":
    time_folder = os.path.join("logs" ,time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + mode + "-" + str(patch_size))
    logs_file = os.path.join(time_folder, "logs.txt")
    os.makedirs(time_folder)
else:
    raise ValueError("invalid model type!")
#-------------------------------------------------------------------------------

# time setting
# np.random.seed(3407)
# torch.manual_seed(3407)
# torch.cuda.manual_seed(3407)

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
# cudnn.deterministic = True
# cudnn.benchmark = False

if data_type == "file":
    data_train, data_val, data_test, label_train, label_val, label_test, bands, num_classes, classes = load_and_split_data_by_file(data_mat_path, label_mat_path, train_ratio, val_ratio, patch_size=patch_size)
elif data_type == "folder":
    data_train, data_val, data_test, label_train, label_val, label_test, bands, num_classes, classes= load_and_split_data_by_folder(data_mat_path, label_mat_path, train_ratio, val_ratio, patch_size=patch_size)
else:
    raise ValueError("invalid data type!")

# 定义一系列变换操作
transform = Compose([
    Normalize(mean=0, std=1),
    # RandomHorizontalFlip(p=0.5),
])

if model_type == "HybridSN":
    model = HybridSN(
        in_channels = bands, 
        patch_size = patch_size, 
        num_classes = int(np.max(classes).item()) + 1,
    )
elif model_type == "3D_1D_CNN":
    model = CNN_3D_Classifer_1D(
            input_channels = bands,
            num_classes = int(np.max(classes).item()) + 1,
            patch_size = patch_size,
            dilation =  1
        )
else:
    raise ValueError("invalid model type!")

# dataset and dataloder
#-------------------------------------------------------------------------------
train_dataset = HSI_dataset(data_train, label_train, transform)
val_dataset = HSI_dataset(data_val, label_val, transform)
test_dataset = HSI_dataset(data_test, label_test, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#-------------------------------------------------------------------------------

# model settings
#-------------------------------------------------------------------------------
if DP:
    model = nn.DataParallel(model).to(device) # 包装为 DataParallel
else:
    model = model.to(device) 
    
# criterion
if focal_loss:
    criterion = FocalLoss(ignore_index=ignore_index).to(device) 
else:
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device) 
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//2, gamma=gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch//5, eta_min=5e-7)

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
        train_acc, train_loss, label_t, prediction_t = train_epoch(model, train_loader, criterion, optimizer, e, epoch, device)
        scheduler.step()
        OA_train, AA_train, Kappa_train, CA_train, CM_train = output_metric(label_t, prediction_t, classes) 
        print("Epoch: {:03d} | train_loss: {:.4f} | train_acc: {:.4f}%".format(e+1, train_loss, train_acc))
        epoch_result[0][e], epoch_result[1][e], epoch_result[2][e] = e+1, train_loss, train_acc
        log_training_results(logs_file, mode='train', epoch_num=e+1, train_loss=train_loss, train_acc=train_acc)

        if ((e+1) % test_freq == 0) | (e == epoch - 1):
            print("===============================================================================")
            print("start validating")      
            model.eval()
            loss_v, label_v, prediction_v = valid_epoch(model, val_loader, criterion, device)
            OA_val, AA_val, Kappa_val, CA_val, CM_val = output_metric(label_v, prediction_v, classes)
            log_training_results(logs_file, mode='train', epoch_num=e+1, train_loss=train_loss, train_acc=train_acc,
                     OA_val=OA_val, AA_val=AA_val, Kappa_val=Kappa_val, CA_val=CA_val, CM_val=CM_val)
            
            # save model and its parameters 
            torch.save(model, os.path.join(time_folder, "model_e{:03d}_loss{:.4f}.pt".format(e+1, loss_v)))
            torch.save(model.state_dict(), os.path.join(time_folder, "model_state_dictl_e{:03d}_loss{:.4f}.pth".format(e+1, loss_v)))

            if (e != epoch -1):
                print("Epoch: {:03d}  =>  OA: {:.4f}% | AA: {:.4f}% | Kappa: {:.4f}".format(e+1, OA_val*100, AA_val*100, Kappa_val))
            print("===============================================================================")

    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("end training")
    print("===============================================================================")
    print("Val result:")
    print("OA: {:.4f}% | AA: {:.4f}% | Kappa: {:.4f}".format(OA_val*100, AA_val*100, Kappa_val))
    print("CA:", end="")
    print(CA_val)
    print("Val Confusion Matrix:")
    print(CM_val)
    print("===============================================================================")

elif mode == "test":
    model.load_state_dict(torch.load(model_path))
    print("load model path : " + model_path)
    print("===============================================================================")

print("start testing")
model.eval()

label, prediction = test_epoch(model, test_loader, criterion, device)
OA_test, AA_test, Kappa_test, CA_test, CM_test = output_metric(label, prediction, classes)
log_training_results(logs_file, mode='test', OA_test=OA_test, AA_test=AA_test, Kappa_test=Kappa_test, 
                     CA_test=CA_test, CM_test=CM_test)
print("Test result:")
print("OA: {:.4f}% | AA: {:.4f}% | Kappa: {:.4f}".format(OA_test*100, AA_test*100, Kappa_test))
print("CA:", end="")
print(CA_test)
print("Test Confusion Matrix:")
print(CM_test)
print("===============================================================================")

print("end testing")
print("===============================================================================")
