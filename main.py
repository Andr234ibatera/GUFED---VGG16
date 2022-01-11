import os, sys, torch
from network import VGG16

class Main():
    param = sys.argv[1:]

    args = {
        'target_path': param[0].replace('\\', '/').replace('\"',''),
        'epoch_num': 150,
        'lr': 1e-2,             # The learning rate was initially set to 10−2, and then decreased by a factor of 10 when the validation set accuracy stopped improving.
        'weight_decay': 5 * 1e-4,   # The training was regularised by weight decay (the L2 penalty multiplier set to 5.10−4) and dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5).
        'batch_size': 50,       
    }

    if(torch.cuda.is_available()):
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')
    print(args['device'])

    net = VGG16(in_size=1, out_size=6).to(args['device'])
    print(net)