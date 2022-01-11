import sys, os
from typing import Tuple
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from constants import emotions

def getDimList(target):
    min_dim = []
    max_dim = []
    subjects_count = len(os.listdir(target))
    count = 0

    for dir, subdir, files in os.walk(target):

        if(len(files) > 0):
            imgs = [cv.imread(f"{path}\{file}") for path, file in zip([dir] * (len(files) - 1), files)]
            widths = np.array([x.shape[0] for x in imgs])
            
            if(widths.size > 0):
                    min_dim.append(widths.min())
                    max_dim.append(widths.max())
        
        if(len(subdir) > 0):
            print("{x}/{y}".format(x = count, y = subjects_count))
            count += 1
    
    return min_dim, max_dim

def showMinMax(mins, maxs):
    x = np.linspace(0, len(mins), len(mins))

    mins = np.sort(mins)
    maxs = np.sort(maxs)

    means = [(x+y)/2 for x, y in zip(mins, maxs)]

    plt.figure(num = 3, figsize=(8, 5))
    plt.plot(x, mins, color='blue', linewidth=1)
    plt.plot(x, maxs, color='blue', linewidth=1)
    plt.plot(x, means, color='red', linewidth=0.5, linestyle='--')

    plt.show()

def generateDirs(in_target_dir, in_new_dir):
    subjects_count = len(os.listdir(in_target_dir))
    count = 0

    print("Making Directories")
    for dir, subdir, files in os.walk(in_target_dir):
        if(len(files) > 0):
            mkdir_target = str(in_new_dir+'\\'+'\\'.join(str(dir).split("\\")[1:])).split("\\")
            maked_dir = mkdir_target[0]
            mkdir_target = mkdir_target[1:]
            
            for path in mkdir_target+['/']:
                if(os.path.isdir(maked_dir) == False):
                    os.mkdir(str(maked_dir))
                
                maked_dir = os.path.join(str(maked_dir), str(path))
        
        if(len(subdir) > 0):
            print("{x}/{y}".format(x = count, y = subjects_count))
            count += 1

def resizeImage(in_target_dir, in_new_dir, in_dim):
    subjects_count = len(os.listdir(in_target_dir))
    count = 0

    generateDirs(in_target_dir, in_new_dir)

    print("Resizing Images")
    for dir, subdir, files in os.walk(in_target_dir):

        if(len(files) > 0):
            imgs = [cv.imread(f"{path}\{file}") for path, file in zip([dir] * (len(files) - 1), files)]
            imgs = [cv.resize(img, (dim, dim)) for img, dim in zip(imgs, [in_dim] * (len(imgs) - 1))]
            
            paths = [f"{path}\{file}" for path, file in zip([in_new_dir+'\\'+'\\'.join(str(dir).split("\\")[1:])] * (len(files) - 1), files)]

            [cv.imwrite(os.path.join('\\'.join(path.split('\\')[:-1]), path.split('\\')[-1:][0]).replace('\\','/'), img) for path, img in zip(paths, imgs)]
        
        if(len(subdir) > 0):
            print("{x}/{y}".format(x = count, y = subjects_count))
            count += 1

def applyGaussianBlur(in_target_dir, in_new_dir):
    subjects_count = len(os.listdir(in_target_dir))
    count = 0
    
    generateDirs(in_target_dir, in_new_dir)

    print("Applying Gaussian Blur")
    for dir, subdir, files in os.walk(in_target_dir):

        if(len(files) > 0):
            imgs = [cv.imread(f"{path}\{file}") for path, file in zip([dir] * (len(files) - 1), files)]
            imgs = [cv.GaussianBlur(img, (3, 3), 0) for img in imgs]
            imgs = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imgs]
            
            paths = [f"{path}\{file}" for path, file in zip([in_new_dir+'\\'+'\\'.join(str(dir).split("\\")[1:])] * (len(files) - 1), files)]

            [cv.imwrite(os.path.join('\\'.join(path.split('\\')[:-1]), path.split('\\')[-1:][0]).replace('\\','/'), img) for path, img in zip(paths, imgs)]
        
        if(len(subdir) > 0):
            print("{x}/{y}".format(x = count, y = subjects_count))
            count += 1


class Main():
    param = sys.argv[1:]

    args = {
        'emotions': emotions,
        'target_path': param[0].replace('\\', '/').replace('\"',''),
        'normalized_path': './Normalized size',
        'gaussian_path': './Gaussian Blured',
    }

    # min_dim, max_dim = getDimList(args['target_path'])
    # showMinMax(min_dim, max_dim)

    applyGaussianBlur(args['target_path'], args['gaussian_path'])

    resizeImage(args['gaussian_path'], args['normalized_path'], 224)
