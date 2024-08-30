import torch

# a class for calculating the average of the accuracy and the loss
#-------------------------------------------------------------------------------
class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.average = 0 
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.count += n
    self.average = self.sum / self.count
#-------------------------------------------------------------------------------

# transform
#-------------------------------------------------------------------------------
# 自定义归一化 transform
class Normalize:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample):
        return (sample - self.min_val) / (self.max_val - self.min_val)

# 自定义水平翻转 transform
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand(1).item() < self.p:
            sample = torch.flip(sample, dims=[2])
        return sample

# 组合多个变换的 Compose 类
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
#-------------------------------------------------------------------------------
    

#-------------------------------------------------------------------------------
def log_training_results(file_path, mode, epoch=None, train_loss=None, train_acc=None,
                         OA_val=None, AA_val=None, Kappa_val=None, CA_val=None, CM_val=None,
                         OA_test=None, AA_test=None, Kappa_test=None, CA_test=None, CM_test=None,
                         epoch_num=None):
    with open(file_path, 'a') as log_file:
        if mode == "train":
            if epoch_num is not None:
                log_file.write(f"Epoch: {epoch_num:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n")
            
            if OA_val is not None and AA_val is not None and Kappa_val is not None and CA_val is not None and CM_val is not None:
                log_file.write("===============================================================================\n")
                log_file.write(f"Epoch: {epoch_num:03d} => Validation Results:\n")
                log_file.write(f"OA: {OA_val:.4f} | AA: {AA_val:.4f} | Kappa: {Kappa_val:.4f}\n")
                log_file.write("CA: ")
                log_file.write(f"{CA_val}\n")
                log_file.write("Validation Confusion Matrix:\n")
                log_file.write(f"{CM_val}\n")
                log_file.write("===============================================================================\n")
        
        elif mode == "test":
            if OA_test is not None and AA_test is not None and Kappa_test is not None and CA_test is not None and CM_test is not None:
                log_file.write("===============================================================================\n")
                log_file.write("Test Results:\n")
                log_file.write(f"OA: {OA_test:.4f} | AA: {AA_test:.4f} | Kappa: {Kappa_test:.4f}\n")
                log_file.write("CA: ")
                log_file.write(f"{CA_test}\n")
                log_file.write("Test Confusion Matrix:\n")
                log_file.write(f"{CM_test}\n")
                log_file.write("===============================================================================\n")
        
        if epoch_num is None:
            log_file.write("End of process\n")
            log_file.write("===============================================================================\n")
#-------------------------------------------------------------------------------