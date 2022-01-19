import os, sys
import numpy as np
import pandas as pd
from constants import emotions

def getLabel(in_str):
    splited = str(in_str).split('\\')
    return splited[len(splited) - 1]

def getSubject(in_str):
    splited = str(in_str).split('\\')[0]
    splited = str(splited).split('/')
    return splited[len(splited) - 1]

def getImagePath(in_dir, in_file_name):
    path = os.path.join(in_dir, in_file_name)
    return str(path).replace('\\', '/')


class GenerateCVS():
    param = sys.argv[1:]

    args = {
        'target_path': param[0].replace('\\', '/').replace('\"',''),
        'new_path': './Equalized',
        'headers': ['subject', 'label', 'image_name', 'path'],
    }
    data = []

    for dir, _, files in os.walk(args['target_path']):

        if(len(files) > 0):
            label = emotions[getLabel(dir)]
            subject = getSubject(dir)
            paths = [getImagePath(dir, file) for dir, file in zip([dir]*len(files), files)]
            
            for path, file in zip(paths, files):
                data.append([subject, label, file, path])
    
    new_file = os.path.join(args['target_path'],'data.csv')
    if os.path.isfile(new_file):
        os.remove(new_file)

    df = pd.DataFrame(np.array(data), columns=args['headers'])
    df.to_csv(new_file, index=False)
    
    