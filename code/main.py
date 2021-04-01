import sys

import torch
import torch.nn as nn
from tqdm import tqdm

from model import Module
from utils import make_submit, get_data_loader, setup_seed
from const import MODEL_NAME, TMP_FOLDER

GPU = 'cuda:0'

CPU = 'cpu'

MIN_LOSS = 1000

PATIENCE = 5

BATCH_SIZE = 20

def run_train():
    train_loader = get_data_loader('train', BATCH_SIZE)
    model = Module().to(GPU)
    criterion = nn.CrossEntropyLoss(size_average=False)
    optim = torch.optim.Adam(model.parameters(), 0.005)
    
    min_loss = MIN_LOSS
    
    model.train()
    
    for epoch in range(20):
        print('EPOCH:{}'.format(epoch))
        for data in tqdm(train_loader):
            c1, c2, c3, c4, c5, c6 = model(data[0].to(GPU))
            loss = criterion(c1, data[1][:, 0].to(GPU)) + \
                criterion(c2, data[1][:, 1].to(GPU)) + \
                criterion(c3, data[1][:, 2].to(GPU)) + \
                criterion(c4, data[1][:, 3].to(GPU)) + \
                criterion(c5, data[1][:, 4].to(GPU)) + \
                criterion(c6, data[1][:, 5].to(GPU))

            loss /= 6
            optim.zero_grad()
            loss.backward()
            optim.step()
        if loss < min_loss:
            earlystopping = 0
            min_loss = loss
            torch.save(model.state_dict(), TMP_FOLDER + MODEL_NAME)
        else:
            earlystopping+=1
            if earlystopping == PATIENCE:
                break
        print('CrossEntropyLoss:{}'.format(loss))
        
def run_validation():
    val_loader = get_data_loader('val', BATCH_SIZE)

    model = Module().to(GPU)
    model.load_state_dict(torch.load(TMP_FOLDER + MODEL_NAME))           
    model.eval()
    
    result1 = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            c1, c2, c3, c4, c5, c6 = model(data[0].to(GPU))
            for i in range(len(c1)):
                result = []
                for c in [c1, c2, c3, c4, c5, c6]:
                    _, indice = c[i].max(0)
                    result.append(int(indice))
                result1.append(result)
    make_submit(result1, 'val')

def run_test():
    test_loader = get_data_loader('test_a', BATCH_SIZE)

    model = Module().to(GPU)
    model.load_state_dict(torch.load(TMP_FOLDER + MODEL_NAME))           
    model.eval()
    
    result1 = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            c1, c2, c3, c4, c5, c6 = model(data.to(GPU))
            for i in range(len(c1)):
                result = []
                for c in [c1, c2, c3, c4, c5, c6]:
                    _, indice = c[i].max(0)
                    result.append(int(indice))
                result1.append(result)
    make_submit(result1, 'test_a')
    
if __name__ == "__main__":
    setup_seed(0)
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'val':
            run_validation()
        elif cmd == 'test':
            run_test()
        else:
            print('错误的命令参数`{}`'.format(cmd))
    else:
        run_train()
