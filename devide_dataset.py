import sys, os
from typing import Tuple
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from constants import emotions

def generateDirs(in_target_dir, in_train_dir, in_val_dir, limit):
    subjects_count = len(os.listdir(in_target_dir))
    count = 0

    print("Making Directories")
    for dir, subdir, files in os.walk(in_target_dir):
        if(len(files) > 0):
            mkdir_target = str(dir).replace(in_target_dir, (in_train_dir if count <= (limit+1) else in_val_dir)+'/')
            mkdir_target = str(mkdir_target).replace('\\', '/').split('/')
            mkdir_target = [os.path.join(mkdir_target[0], mkdir_target[1])] + mkdir_target[2:]

            pth = ''
            for path in mkdir_target:
                pth = os.path.join(str(pth), str(path))

                if(os.path.isdir(pth) == False):
                    os.mkdir(str(pth))
        
        if(len(subdir) > 0):
            print("{x}/{y}".format(x = count, y = subjects_count))
            count += 1

def getDatasetLimitation(in_target_dir, in_train, in_val):
    print("Getiing statistics")
    file_count = []
    file_means = []
    max_subdir = 0

    for _, subdir, files in os.walk(in_target_dir):
        if(len(subdir) > max_subdir):
            max_subdir = len(subdir)

        if(len(files) > 0):
            file_count.append(len(files))

        if(len(subdir) > 0):
            file_means.append(np.mean(file_count))
    
    train = (max_subdir * (int(in_train)/100))//1
    val = (max_subdir - train)//1
    return train, val

class Main():
    param = sys.argv[1:]

    args = {
        'emotions': emotions,
        'target_path': param[0].replace('\\', '/').replace('\"',''),
        'trainset': './trainset',
        'validationset': './validation',
        'train_percent': param[1],
        'validation_percent': param[2],
    }
    
    train_c, val_c = getDatasetLimitation(args['target_path'], args['train_percent'], args['validation_percent'])
    generateDirs(args['target_path'], args['trainset'], args['validationset'], train_c)

    subjects_count = len(os.listdir(args['target_path']))
    count = 0
    for dir, subdir, files in os.walk(args['target_path']):
        if(len(files) > 0):
            imgs = [cv.imread(f"{path}\{file}", 0) for path, file in zip([dir] * (len(files) - 1), files)]
            imgs = [cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in imgs]
            
            splited_dir = ('/').join(str(dir).replace('\\', '/').split('/')[2:])
            path = ('/').join([(args['trainset'] if count <= (train_c+1) else args['validationset']), splited_dir])
            paths = [f"{path}/{file}" for path, file in zip([path] * (len(files) - 1), files)]
            
            [cv.imwrite(path, img) for path, img in zip(paths, imgs)]
            
        if(len(subdir) > 0):
            print("{x}/{y}".format(x = count, y = subjects_count))
            count += 1

