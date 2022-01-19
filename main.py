import os, sys, torch, time
from network import VGG16
from emotionsdataset import EmotionsDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics
from torchsummary  import summary
from transforms import Gaussian, Rescale, ToTensor

from torch import nn, optim

# TODO procurar o modo debug p

def train(train_loader, net, epoch, criterion, optimizer, device, logfile):

    # Training mode
    net.train()

    metric = torchmetrics.Accuracy().to(device)
    
    start = time.time()
    epoch_loss  = []
    for batch in train_loader:
        
        dado, rotulo = batch['image'].to(device), batch['label'].to(device)
                
        # Forward
        ypred = net(X=dado)
        loss = criterion(ypred, rotulo)
        epoch_loss.append(loss.cpu().data)

        pred = ypred.softmax(dim=-1)
        acc = metric(pred, rotulo)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epoch_loss = np.asarray(epoch_loss)
    
    acc = metric.compute()
    
    end = time.time()
    fprint(logfile, '#################### Train ####################')
    fprint(logfile, 'Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))
    
    return epoch_loss.mean()

def validate(train_loader, net, epoch, criterion, device, logfile):

    # Training mode
    net.eval()

    metric = torchmetrics.Accuracy().to(device)
    
    start = time.time()
    epoch_loss  = []

    with torch.no_grad():
        for batch in train_loader:
            
            dado, rotulo = batch['image'].to(device), batch['label'].to(device)
                    
            # Forward
            ypred = net(X=dado)
            loss = criterion(ypred, rotulo)
            epoch_loss.append(loss.cpu().data)

            pred = ypred.softmax(dim=-1)
            acc = metric(pred, rotulo)
    
    epoch_loss = np.asarray(epoch_loss)
    
    acc = metric.compute()
    
    end = time.time()
    fprint(logfile, '#################### Validation ####################')
    fprint(logfile, 'Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))
    
    return epoch_loss.mean()

def plot(in_trains, in_vals, epochs, title, figname):
    plt.plot(epochs, in_trains, label="Train")
    plt.plot(epochs, in_vals, label="Validation")
    plt.plot()

    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title(str(title))
    plt.legend()
    plt.savefig(figname)

def generateLogFile(in_filename, in_start_line):
    log_path = 'logs'
    if(os.path.isdir(log_path) == False):
        os.mkdir(str(log_path))
    
    f = open(os.path.join(log_path, in_filename),"w+")
    f.write(str(in_start_line)+"\n")
    f.close()

def fprint(in_filename, in_output):
    print(in_output)
    with open(os.path.join('./logs', in_filename), "a") as f:
        f.write("{}\n".format(in_output))
class Main():
    param = sys.argv[1:]

    args = {
        'train_path': param[0].replace('\\', '/').replace('\"',''),
        'val_path': param[1].replace('\\', '/').replace('\"',''),
        'epoch_num': 150,
        'lr': 1e-3,             # [1e-2 até 1e-5] The learning rate was initially set to 10−2, and then decreased by a factor of 10 when the validation set accuracy stopped improving.
        'weight_decay': 5 * 1e-4,   # The training was regularised by weight decay (the L2 penalty multiplier set to 5.10−4) and dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5).
        'momentum': 0.9,
        'batch_size': 50,
    }

    if(torch.cuda.is_available()):
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')
    
    timestamp = time.time()
    log_filename = str(timestamp)+'.txt'
    title = "VGG16 + GUFED (lr: {}, decay: {})".format(args['lr'], args['weight_decay'])
    generateLogFile(log_filename, str(args))

    net = VGG16(in_size=3, out_size=6).to(args['device'])
    net = net.to(args['device'])
    # summary(net, (3, 224, 224))
    
    train_set = EmotionsDataset(csv_file=os.path.join(args['train_path'], 'data.csv'), 
                                    root_dir=args['train_path'],
                                    transform=transforms.Compose([Gaussian(), Rescale(224), ToTensor()]))
    
    val_set = EmotionsDataset(csv_file=os.path.join(args['val_path'], 'data.csv'), 
                                    root_dir=args['val_path'],
                                    transform=transforms.Compose([Gaussian(), Rescale(224), ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=args['batch_size'], shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=args['batch_size'], shuffle=True, num_workers=0)
    
    criterion = nn.CrossEntropyLoss().to(args['device'])
    # optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    # criterion = nn.MSELoss().to(args['device']) # No artigo é utilizado L2
    optimizer = optim.SGD(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'])

    train_losses, val_losses = [], []
    for epoch in range(args['epoch_num']):
        fprint(log_filename, f"EPOCH {epoch}/{args['epoch_num']}")
        # Train
        train_losses.append(train(train_loader, net, epoch, criterion, optimizer, args['device'], log_filename))
        
        #Validation
        val_losses.append(validate(val_loader, net, epoch, criterion, args['device'], log_filename))
    
    plot(train_losses, val_losses, [i for i in range(len(val_losses))], title, os.path.join('./logs',str(timestamp)+'.png'))