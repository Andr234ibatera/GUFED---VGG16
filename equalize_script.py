import cv2 as cv
import os, sys
import numpy as np
from constants import emotions

def makeDir(old_dir, new_dir):
    for dir, _, files in os.walk(old_dir):
        if(len(files) > 0):
            mkdir_target = str(dir).replace(old_dir, new_dir+'/')
            mkdir_target = str(mkdir_target).replace('\\', '/').split('/')
            mkdir_target = [os.path.join(mkdir_target[0], mkdir_target[1])] + mkdir_target[2:]

            pth = ''
            for path in mkdir_target:
                pth = os.path.join(str(pth), str(path))

                if(os.path.isdir(pth) == False):
                    os.mkdir(str(pth))

class Main():
    param = sys.argv[1:]

    args = {
        'emotions': emotions,
        'target_path': param[0].replace('\\', '/').replace('\"',''),
        'new_path': './Equalized',
    }

    subjects_count = len(os.listdir(args['target_path']))
    count = 0

    makeDir(args['target_path'], args['new_path'])

    for dir, subdir, files in os.walk(args['target_path']):
        if(len(files) > 0):
            imgs = [cv.imread(f"{path}\{file}", 0) for path, file in zip([dir] * (len(files) - 1), files)]
            imgs = [cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in imgs]

            t_imgs = [img.copy() for img in imgs]
            t_imgs = [cv.normalize(img, temp, 0, 255, cv.NORM_MINMAX) for img, temp in zip(imgs, t_imgs)]
            
            splited_dir = ('/').join(str(dir).replace('\\', '/').split('/')[2:])
            path = ('/').join([args['new_path'], splited_dir])
            paths = [f"{path}/{file}" for path, file in zip([path] * (len(files) - 1), files)]

            # print(paths)
            # break
            [cv.imwrite(path, img) for path, img in zip(paths, t_imgs)]
            # [cv.imwrite(os.path.join('/'.join(path.split('/')[:-1]), path.split('/')[-1:][0]), img) for path, img in zip(paths, t_imgs)]
        
            
        if(len(subdir) > 0):
            print("{x}/{y}".format(x = count, y = subjects_count))
            count += 1