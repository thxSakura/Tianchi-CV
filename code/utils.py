import collections
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from const import (DATA_FOLDER, DATA_NAME, DATA_PATH, LABEL_NAME,
                   OUTPUT_FOLDER, SUBMIT_NAME)
from data_holder import TESTDataset, TRAINDataset

THREAD = 8

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True    

def get_data_loader(data, batch_size, sample=None):
    data_path = glob.glob(DATA_FOLDER + DATA_PATH.format(data) + DATA_NAME)

    data_path.sort()
    if data == 'test_a':        
        data_loader = torch.utils.data.DataLoader(
                TESTDataset(data_path,
                           transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.ColorJitter(0.3, 0.3, 0.2),
                               transforms.RandomRotation(5),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])), 
            batch_size=batch_size, # 每批样本个数
            shuffle=False, # 是否打乱顺序
            num_workers=THREAD, # 读取的线程个数
        )
    else:
        data_json = json.load(open(DATA_FOLDER + LABEL_NAME.format(data)))
        data_label = [data_json[x]['label'] for x in data_json]

        data_loader = torch.utils.data.DataLoader(
                TRAINDataset(data_path, data_label,
                           transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.ColorJitter(0.3, 0.3, 0.2),
                               transforms.RandomRotation(5),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])), 
            batch_size=batch_size, # 每批样本个数
            shuffle=False, # 是否打乱顺序
            num_workers=THREAD, # 读取的线程个数
        )

    return data_loader

def cal_num(c1, c2, c3, c4, c5, c6, label=False, result=None):
    if result == None:
        result = []
    for i in range(len(c1)):
        num = []
        for c in [c1, c2, c3, c4, c5, c6]:
            if label:
                num.append(int(c[i]))
            else:
                _, indice = c[i].max(0)
                num.append(int(indice))
        
        number = 0
        invalid = collections.Counter(num)[10]
        i = 1
        for n in num:
            if n == 10:
                break
            number += n * (10 ** (6 - invalid - i))
            i += 1
        result.append(number)
    return result

def cal_auc(predict_nums, label_nums):
    score = 0
    for i in range(len(predict_nums)):
        if predict_nums[i] == label_nums[i]:
            score += 1
    return score / len(predict_nums)

def make_submit(result, data):
    sub = pd.DataFrame(columns=('file_name','file_code'))
    names = os.listdir(DATA_FOLDER + DATA_PATH.format(data))
    names.sort()
    sub['file_name'] = names
    sub['file_code'] = result
    
    sub.to_csv(OUTPUT_FOLDER + SUBMIT_NAME, index=False)
    print('生成提交文件。')
    
def bar(iterator, epoch):
    
    return tqdm(iterator, 
#                 desc='EPOCH:{}'.format(epoch), #进度栏的前缀。
                ncols=75, #整个输出消息的宽度。
                maxinterval=1, #最大进度显示更新间隔[默认：10]秒。
                colour='green',
            )
