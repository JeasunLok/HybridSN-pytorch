import numpy as np
from tqdm import tqdm
from utils.utils import AverageMeter
from utils.metrics import accuracy

#-------------------------------------------------------------------------------
def test_epoch(model, test_loader, criterion, device):
    loss_show = AverageMeter()
    acc = AverageMeter()
    label = np.array([])
    prediction = np.array([])
    loop = tqdm(enumerate(test_loader), total = len(test_loader))
    for batch_idx, (batch_data, batch_label) in loop:
        batch_data = batch_data.float().to(device)
        batch_label = batch_label.long().to(device) 

        batch_prediction = model(batch_data)
        loss = criterion(batch_prediction, batch_label)

        # calculate the accuracy
        acc_batch, l, p = accuracy(batch_prediction, batch_label, topk=(1,))
        n = batch_data.shape[0]

        # update the loss and the accuracy 
        loss_show.update(loss.data, n)
        acc.update(acc_batch[0].data, n)
        label = np.append(label, l.data.cpu().numpy())
        prediction = np.append(prediction, p.data.cpu().numpy())

        # Format accuracy as percentage
        accuracy_percentage = acc.average.item() * 100

        loop.set_description(f'Test Epoch')
        loop.set_postfix({
            "test_loss": loss_show.average.item(),
            "test_accuracy": f"{acc.average.item():.2f}%"
        })
        
    return label, prediction
#----------------------------------------------------------------------------