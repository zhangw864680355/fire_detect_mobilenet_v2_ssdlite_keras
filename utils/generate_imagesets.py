#! -*- coding:utf-8 -*-

import os
import random

def file_is_image(img_path):
    suffix = img_path.split('.')[-1].lower()
    if suffix in ['jpg', 'jpeg', 'png', 'bmp']:
        return True
    else:
        return False

def get_imgset_list(dirs):
    imgset_list = []
    if not os.path.exists(dirs):
        return None
    for root, dirs, files in os.walk(dirs):
        for img in files:
            if file_is_image(img):
                imgset_list.append(img.split('.')[0])
    return imgset_list

if __name__ == '__main__':
    img_dir = './JPEGImages'
    imgset_list = get_imgset_list(img_dir)
    random.shuffle(imgset_list)

    train_imgsets_path = './ImageSets/train.txt'
    val_imgsets_path = './ImageSets/val.txt'
    ftrain = open(train_imgsets_path, 'w')
    fval = open(val_imgsets_path, 'w')
    
    for index , imgset in enumerate(imgset_list):
        if index > 0 and index % 5 == 0:
            fval.write(imgset+'\n')
        else:
            ftrain.write(imgset+'\n')

    ftrain.close()
    fval.close()
