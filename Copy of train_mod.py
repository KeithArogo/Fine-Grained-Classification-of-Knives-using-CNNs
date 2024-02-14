import sys
import warnings
import numpy as np
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
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
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
from sklearn.metrics import precision_recall_fscore_support
import torch

def evaluate(val_loader, model, criterion, epoch, train_loss, start):
    model.cuda()
    model.eval()
    model.training = False
    map = AverageMeter()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images, target, fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)

            logits = model(img)
            preds = logits.softmax(1)
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5, img.size(0))
            
            # Store predictions and targets for calculating class-wise metrics
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(label.cpu().numpy())
            
    print('\r',end='',flush=True)
    message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % (\
            "val", i, epoch, train_loss[0], map.avg,time_to_str((timer() - start),'min'))
    print(message, end='',flush=True)
    log.write("\n")  
    log.write(message)

    # Calculate class-wise precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)

    # Assuming your model is trained for a fixed number of classes (e.g., 192)
    num_classes = 191
    all_classes = np.arange(num_classes)

    # Read test.csv to get class labels
    test_df = pd.read_csv('test.csv')
    test_classes = test_df['Label'].unique()

    # Filter out class indices that are not in test.csv
    valid_classes = np.intersect1d(all_classes, test_classes)
    
    # Filter valid_classes to include only those present in precision, recall, and f1
    filtered_valid_classes = [cls for cls in valid_classes if cls < len(precision)]

    # Proceed with filtering if valid classes are within range
    #try:
    filtered_metrics = {
        'precision': precision[filtered_valid_classes],
        'recall': recall[filtered_valid_classes],
        'f1': f1[filtered_valid_classes]
    }
 
    val_mAPs.append(map.avg.item()) 
 
    # Instead of just returning metrics, also return the labels they correspond to
    # Get unique labels present in predictions and limit them to the model's range
    actual_labels = np.unique(all_targets)
    actual_labels = [label for label in actual_labels if label < len(precision)]
    return [map.avg, {'precision': precision, 'recall': recall, 'f1': f1}, actual_labels]

# Modify the main training loop to handle class-wise metrics
# ...

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

class EarlyStopping:
  def __init__(self, patience=5, min_delta=0.02):  # Adjust these parameters as needed
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

  def __call__(self, val_mAP, model):
    score = val_mAP

    if self.best_score is None:
        self.best_score = score
        self.save_checkpoint(val_mAP, model)
    elif score > self.best_score + self.min_delta:
        self.best_score = score
        self.save_checkpoint(val_mAP, model)
        self.counter = 0
    else:
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


  def save_checkpoint(self, val_mAP, model):
        '''Saves model when validation score improves.'''
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.best_score = val_mAP
######################## seed #################################################
# Set a seed value
seed = 42 
# Set the seed for PyTorch's random number generator for all devices (both CPU and CUDA)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
######################## load file and get splits #############################
train_imlist = pd.read_csv("train.csv")
train_gen = knifeDataset(train_imlist,mode="train")
train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=8)

#####################_____________##############################################
val_imlist = pd.read_csv("test.csv")
val_gen = knifeDataset(val_imlist,mode="val")
val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=8)

## Loading the model to run
model = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=config.n_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Parameters #################################
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,last_epoch=-1)
criterion = nn.CrossEntropyLoss().cuda()

############################# Training #################################
start_epoch = 0
val_metrics = [0]
scaler = torch.cuda.amp.GradScaler()
start = timer()

# Initialize early stopping object .
# patience is the number of epochs to wait for improvement.
# min_delta is the minimum change in the validation metric to qualify as an improvement.
early_stopping = EarlyStopping(patience=5, min_delta=0.02)
print(f"Starting training with Batch Size: {config.batch_size}")
print(f"Initial Learning Rate: {config.learning_rate}")

# learning curve values
train_losses, val_losses = [], []
val_mAPs = []  
all_class_wise_metrics = {}

########
num_classes = 192
all_classes = np.arange(num_classes)

# Read test.csv to get class labels
test_df = pd.read_csv('test.csv')

# training
for epoch in range(0,config.epochs):
  
    lr = get_learning_rate(optimizer)    
    train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, start)
    train_loss = train_metrics[0]
    
    if torch.is_tensor(train_loss):
        train_loss = train_loss.cpu().item()  # Convert to Python scalar
    train_losses.append(train_loss)

    #val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, start)
    #val_loss, class_wise_metrics = val_metrics

    val_loss, class_wise_metrics, actual_labels = evaluate(val_loader, model, criterion, epoch, train_metrics, start)
    #actual_labels = [label for label in actual_labels if label != 131]

    # Use actual_labels instead of filtering based on the length
    filtered_metrics = {
        metric: [values[label] for label in actual_labels[0:len(class_wise_metrics['precision'])]]
        for metric, values in class_wise_metrics.items()
    }

    # Create DataFrame
    tabulated_data = pd.DataFrame(filtered_metrics, index=actual_labels)

    # Save as CSV
    csv_filename = f"classwise_metrics_epoch_{epoch}.csv"

    # Create a DataFrame using the test class labels
    tabulated_data.to_csv(csv_filename)
    
    if torch.is_tensor(val_loss):
        val_loss = val_loss.cpu().item()
    
    val_losses.append(val_loss)
    val_mAP = val_mAPs[epoch]

    # Get the unique classes in the test set
    #test_classes = test_df['Label'].unique()

    # Ensure test classes are within the range of calculated metrics
    #test_classes = [cls for cls in test_classes if cls < len(class_wise_metrics['precision'])]

    # Filter metrics for classes in the test set
    #filtered_metrics = {
       # metric: [values[cls] for cls in test_classes]
        #for metric, values in class_wise_metrics.items()
    #}

    # Create a DataFrame using the test class labels
    #tabulated_data = pd.DataFrame(filtered_metrics, index=test_classes)
    
    if torch.is_tensor(val_mAP):
            val_mAP = val_mAP.cpu().item()
    
    # Early Stopping check and save the best model
    if val_mAP is not None:
        early_stopping(val_mAP, model)
        #print("Early stop check....")

        if early_stopping.early_stop:
            print("Early stopping triggered....")
            break
    ## Saving the model
    filename = "Knife-Effb0-E" + str(epoch + 1)+  ".pt"
    torch.save(model.state_dict(), filename)
    

# Plotting training and validation losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting validation mAP (if it's being calculated)
if val_mAPs:
    #val_mAPs = val_mAPs.cpu().item()
    plt.subplot(1, 2, 2)
    plt.plot(val_mAPs, label='Validation mAP')
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()

plt.show()   