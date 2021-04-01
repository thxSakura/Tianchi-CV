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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True    

def get_data_loader(data, batch_size):
    data_path = glob.glob(DATA_FOLDER + DATA_PATH.format(data) + DATA_NAME)

    data_path.sort()
    if data == 'test_a':        
        data_loader = torch.utils.data.DataLoader(
                TESTDataset(data_path,
                           transforms.Compose([
                               transforms.Resize((64, 128)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])), 
            batch_size=batch_size, # 每批样本个数
            shuffle=False, # 是否打乱顺序
            num_workers=10, # 读取的线程个数
        )
    else:
        data_json = json.load(open(DATA_FOLDER + LABEL_NAME.format(data)))
        data_label = [data_json[x]['label'] for x in data_json]

        data_loader = torch.utils.data.DataLoader(
                TRAINDataset(data_path, data_label,
                           transforms.Compose([
                               transforms.Resize((64, 128)),
                               transforms.ColorJitter(0.3, 0.3, 0.2),
                               transforms.RandomRotation(5),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])), 
            batch_size=batch_size, # 每批样本个数
            shuffle=False, # 是否打乱顺序
            num_workers=10, # 读取的线程个数
        )

    return data_loader

def make_submit(result, data):
    submit = []
    for num in tqdm(result):
        number = 0
        invalid = collections.Counter(num)[10]
        i = 1
        for n in num:
            if n == 10:
                break
            number += n * (10 ** (6 - invalid - i))
            i += 1
        submit.append(number)
    sub = pd.DataFrame(columns=('file_name','file_code'))
    names = os.listdir(DATA_FOLDER + DATA_PATH.format(data))
    names.sort()
    sub['file_name'] = names
    sub['file_code'] = submit
    
    sub.to_csv(OUTPUT_FOLDER + SUBMIT_NAME, index=False)
    print('生成提交文件。')
