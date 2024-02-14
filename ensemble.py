## import libraries for training
import sys
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
import torch.nn as nn
from utils import *
warnings.filterwarnings('ignore')

## Writing the loss and results
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
log.open("logs/%s_log_train.txt")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |----- Train -----|----- Valid----|---------|\n')
log.write('mode     iter     epoch    |       loss      |        mAP    | time    |\n')
log.write('-------------------------------------------------------------------------------------------\n')

## Training the model
def train(train_loader,model,criterion,optimizer,epoch,valid_accuracy,start):
    losses = AverageMeter()
    model.train()
    model.training=True
    for i,(images,target,fnames) in enumerate(train_loader):
        img = images.cuda(non_blocking=True)
        label = target.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast():
            logits = model(img)
        loss = criterion(logits, label)
        losses.update(loss.item(),images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        optimizer.zero_grad()
        scheduler.step()

        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f        |      %0.3f     |      %0.3f     | %s' % (\
                "train", i, epoch,losses.avg,valid_accuracy[0],time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    log.write(message)

    return [losses.avg]

# Validating the model
def evaluate(val_loader,model,criterion,epoch,train_loss,start):
    model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter()
    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            top_5.append(valid_acc5)
            top_1.append(valid_acc1)
            map.update(valid_map5,img.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % (\
                    "val", i, epoch, train_loss[0], map.avg,time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")  
        log.write(message)
    return [map.avg]

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5

class EnsembleModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EnsembleModel, self).__init__()
        # Load EfficientNetV2-L
        self.effnet = timm.create_model('tf_efficientnetv2_l', pretrained=pretrained, num_classes=0)
        # Load ResNet-152
        self.resnet = timm.create_model('resnet152', pretrained=pretrained, num_classes=0)

        # Example: Concatenate the features from both models
        self.classifier = nn.Linear(self.effnet.num_features + self.resnet.num_features, num_classes)

    def forward(self, x):
        # Get features from both models
        effnet_feat = self.effnet(x)
        resnet_feat = self.resnet(x)

        # Combine features
        combined_feat = torch.cat((effnet_feat, resnet_feat), dim=1)

        # Final classification layer
        out = self.classifier(combined_feat)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

#criterion = FocalLoss().cuda()

######################## load file and get splits #############################
train_imlist = pd.read_csv("train.csv")
train_gen = knifeDataset(train_imlist,mode="train")
train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=8)
val_imlist = pd.read_csv("test.csv")
val_gen = knifeDataset(val_imlist,mode="val")
val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=8)

## Loading the model to run
#model = EnsembleModel(num_classes=config.n_classes)
model = timm.create_model('tf_efficientnetv2_l', pretrained=True, num_classes=config.n_classes)
#model = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=config.n_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Parameters #################################
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,last_epoch=-1)
criterion = nn.CrossEntropyLoss().cuda()
#criterion = FocalLoss().cuda()
############################# Training #################################
start_epoch = 0
val_metrics = [0]
scaler = torch.cuda.amp.GradScaler()
start = timer()
print(f"Starting training with Batch Size: {config.batch_size}")
print(f"Initial Learning Rate: {config.learning_rate}")
top_5 = []
top_1 = []
#train
for epoch in range(0,config.epochs):
    lr = get_learning_rate(optimizer)
    train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,start)
    val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,start)
    ## Saving the model
    filename = "Knife-Effb0-E" + str(epoch + 1)+  ".pt"
    torch.save(model.state_dict(), filename)

## Plot
plt.plot(epoch, top_5, 'b--', label='Top 5 Accuracy')  # Dotted line for top_5
plt.plot(epoch, top_1, 'r-', label='Top 1 Accuracy')   # Solid line for top_1

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Top 1 and Top 5 Accuracy Over Epochs')
plt.legend()

plt.show()
    

   
