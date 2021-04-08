import sys

import torch
import torch.nn as nn
from apex import amp
from tqdm import tqdm

from const import MODEL_NAME, TMP_FOLDER
from model import Module
from utils import (bar, cal_auc, cal_num, get_data_loader, make_submit,
                   setup_seed)

GPU = torch.device('cuda:0')

PATIENCE = 5

BATCH_SIZE = 100

EPOCH = 2000

LR = 0.001

def run_train():
    train_loader = get_data_loader('train', BATCH_SIZE)
    val_loader = get_data_loader('val', BATCH_SIZE)
    
    model = Module().to(GPU)
    criterion = nn.CrossEntropyLoss()
#     optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
#     model, optim = amp.initialize(model, optim, opt_level="O1") # 这里要添加这句代码

    min_loss = 10000
    earlystopping = 0
    
    model.train()
    print('-'*75)
    for epoch in range(EPOCH):
        #train
        print('Epoch {}/{}'.format(epoch+1, EPOCH))
        for data in bar(train_loader, epoch):
            c1, c2, c3, c4, c5, c6 = model(data[0].to(GPU))
            loss = criterion(c1, data[1][:, 0].to(GPU)) + \
                criterion(c2, data[1][:, 1].to(GPU)) + \
                criterion(c3, data[1][:, 2].to(GPU)) + \
                criterion(c4, data[1][:, 3].to(GPU)) + \
                criterion(c5, data[1][:, 4].to(GPU)) + \
                criterion(c6, data[1][:, 5].to(GPU))

#             loss /= 5
            optim.zero_grad()
#             with amp.scale_loss(loss, optim) as scaled_loss:
#                 scaled_loss.backward()   # loss要这么用
            loss.backward()    
            optim.step()
            
        predict_nums = cal_num(c1, c2, c3, c4, c5, c6)
        label_nums = cal_num(data[1][:, 0], data[1][:, 1], data[1][:, 2], data[1][:, 3], data[1][:, 4], data[1][:, 5], label=True)
        train_score = cal_auc(predict_nums, label_nums)

        #validation
        for data in val_loader:
            c1, c2, c3, c4, c5, c6 = model(data[0].to(GPU))
            val_loss = criterion(c1, data[1][:, 0].to(GPU)) + \
                criterion(c2, data[1][:, 1].to(GPU)) + \
                criterion(c3, data[1][:, 2].to(GPU)) + \
                criterion(c4, data[1][:, 3].to(GPU)) + \
                criterion(c5, data[1][:, 4].to(GPU)) + \
                criterion(c6, data[1][:, 5].to(GPU))
            break
#             val_loss /= 5
        
        predict_nums = cal_num(c1, c2, c3, c4, c5, c6)
        label_nums = cal_num(data[1][:, 0], data[1][:, 1], data[1][:, 2], data[1][:, 3], data[1][:, 4], data[1][:, 5], label=True)
        val_score = cal_auc(predict_nums, label_nums)
        
        print('loss:{} / auc:{} / val_loss:{} / val_auc:{}'.format(loss, train_score, val_loss, val_score))  
        print('-'*75)
        
        if val_loss < min_loss:
            earlystopping = 0
            min_loss = val_loss
            torch.save(model.state_dict(), TMP_FOLDER + MODEL_NAME)
        else:
            earlystopping += 1
            if earlystopping == PATIENCE:
                break

def run_test():
    test_loader = get_data_loader('test_a', BATCH_SIZE)

    model = Module().to(GPU)
    model.load_state_dict(torch.load(TMP_FOLDER + MODEL_NAME))           
    model.eval()
    
    result = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            c1, c2, c3, c4, c5, c6 = model(data.to(GPU))
            result = cal_num(c1, c2, c3, c4, c5, c6, False, result)
    make_submit(result, 'test_a')
    
if __name__ == "__main__":
    setup_seed(0)
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'test':
            run_test()
        else:
            print('错误的命令参数`{}`'.format(cmd))
    else:
        run_train()
