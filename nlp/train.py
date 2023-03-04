import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm

from model import YelpModel, createMask
from yelp_dataset import YelpDataset, collate_fn

def main():

    batchSize = 16
    trainset = YelpDataset('train')
    validset = YelpDataset('valid')

    nWorkers = 4
    trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=nWorkers, collate_fn=collate_fn)
    validloader = DataLoader(validset, batch_size=batchSize, shuffle=False, num_workers=nWorkers, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocabSize = len(trainset.tokToIdx.keys())
    seqLen = trainset.maxLen
    nLayers = 4
    embSize = 256
    padIdx = trainset.tokToIdx['<pad>']

    model = YelpModel(nLayers, embSize, vocabSize, device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)

    nEpochs = 100
    #default
    #NOTE, lrscheduler is in epoch loop, NOT training loop
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    #factor is by how much you reduce lr rate
    #patience is the number of epochs with no improvement until lr is reduced
    factor = 0.5
    patience = 5
    lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=True)
    #'''

    #from gpt-2 paper
    #NOTE, lrscheduler is in training loop, NOT epoch loop
    #'''
    lrRate = 0.00025
    optimizer = torch.optim.Adam(model.parameters(), lr=lrRate)
    #max lr: 2.5e-4
    #linearly increase from 0 for first 2000 updates
    #they use batchSize of 64, so we have more updates
    #last_epoch is index of last batch.  Used when resuming a training job. default is -1 since you haven't started
    lastEpoch = -1
    stepsPerEpoch = len(trainloader)
    pctStart = (2000 * (64/batchSize)) / (stepsPerEpoch * nEpochs) #percent of cycle spent increasing learning rate, usually 0.3
    lrScheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lrRate, 
                    steps_per_epoch=stepsPerEpoch, pct_start=pctStart, epochs=nEpochs, last_epoch=lastEpoch, anneal_strategy='cos')
    #'''

    #load previous state
    #TODO

    lossFunction = nn.CrossEntropyLoss(ignore_index=padIdx)
    bestLoss = float('inf')
    for epoch in range(nEpochs):

        print('Epoch:', epoch)
        
        model.train()
        trainLoss = 0
        cnt = 0
        dataIter = tqdm(trainloader)
        for item in dataIter:
            optimizer.zero_grad()

            tokTen = item['tokTen'].to(device)
            tokInput = tokTen[:,:-1]

            mask = createMask(tokInput, padIdx).to(device)
            
            pred = model(tokInput, mask)
            pred = pred.contiguous().view(-1, pred.shape[-1])
            
            tokTrue = tokTen[:,1:].contiguous().view(-1)
            
            loss = lossFunction(pred, tokTrue)

            loss.backward()            
            optimizer.step()
            lrRate = optimizer.param_groups[0]['lr']

            dataIter.set_description("Epoch: %d Loss: %5.3f lr: %7.5f"%(epoch, loss.item(), lrRate))
            trainLoss += loss.item()
            cnt += 1

            #for OneCycleLR
            lrScheduler.step()


        trainLoss = trainLoss / cnt
        print('Training Loss:',  trainLoss)

        model.eval()
        validLoss = 0
        cnt = 0
        with torch.no_grad():
            dataIter = tqdm(validloader)
            for item in dataIter:
                tokTen = item['tokTen'].to(device)
                tokInput = tokTen[:,:-1]

                mask = createMask(tokInput, padIdx).to(device)

                pred = model(tokInput, mask)
                pred = pred.contiguous().view(-1, pred.shape[-1])
                
                tokTrue = tokTen[:,1:].contiguous().view(-1)
                
                loss = lossFunction(pred, tokTrue)
                dataIter.set_description("Epoch: %d Loss: %5.3f"%(epoch, loss.item()))
                validLoss += loss.item()
                cnt += 1


        validLoss = validLoss / cnt
        print('Validation Loss:', validLoss)

        if validLoss < bestLoss:
            print('Saving state')
            bestLoss = validLoss
            torch.save(model.state_dict(), 'model.pt')
            torch.save(optimizer.state_dict(), 'optimizer.pt')
            torch.save(lrScheduler.state_dict(), 'lrScheduler.pt')

        #for reduceLROnPlateau
        #lrScheduler.step(valLoss)

if __name__ == '__main__':
    main()