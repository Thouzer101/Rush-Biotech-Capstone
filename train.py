import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import MimicModel, createMask
from dataset import MimicDataset, collate_fn

def main():

    batchSize = 16
    trainset = MimicDataset('train.txt')
    validset = MimicDataset('valid.txt')

    trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
    validloader = DataLoader(validset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    nTokens = len(trainset.tokToIdx.keys())
    nLayers = 4
    embSize = 256
    peSize = 64
    seqLen = trainset.seqLen
    maxHrs = trainset.maxHrs

    padIdx = trainset.tokToIdx['<pad>']
    clsIdx = trainset.tokToIdx['<cls>']
    
    model = MimicModel(nLayers, embSize, nTokens, peSize, device, maxHrs=maxHrs)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)

    nEpochs = 100
    #basic LR scheduler
    #NOTE, lrscheduler is in epoch loop, NOT training loop
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    #factor is by how much you reduce lr rate
    #patience is the number of epochs with no improvement until lr is reduced
    factor = 0.5
    patience = 5
    lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=True)
    #'''

    #LR scheduler from gpt-2 paper
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


    model.load_state_dict(torch.load('model.pt'))
    optimizer.load_state_dict(torch.load('optimizer.pt'))
    lrScheduler.load_state_dict(torch.load('lrScheduler.pt'))

    tokLossFunction = nn.CrossEntropyLoss(ignore_index=padIdx)
    timesLossFunction = nn.CrossEntropyLoss()
    valsLossFunction = nn.MSELoss()

    bestLoss = float('inf')
    trainLosses = []
    validLosses = []
    for epoch in range(nEpochs):

        model.train()
        trainLoss = 0
        cnt = 0
        dataIter = tqdm(trainloader)
        #dataIter = trainloader
        for item in dataIter:
            optimizer.zero_grad()

            died = item['died']

            tokTen = item['tokTen'].to(device)
            valsTen = item['valsTen'].to(device)
            timesTen = item['timesTen'].to(device)

            inTokTen = tokTen[:,:-1]
            inValsTen = valsTen[:,:-1]
            inTimesTen = timesTen[:,:-1]

            mask = createMask(inTokTen, padIdx).to(device)

            out = model(inTokTen, inValsTen, inTimesTen, mask)
            tokPred, valsPred, timesPred = model.unsupervised(out)

            tokPred = tokPred.contiguous().view(-1, tokPred.shape[-1])
            tokTrue = tokTen[:,1:].contiguous().view(-1)
            tokLoss = tokLossFunction(tokPred, tokTrue)

            valsPred = valsPred.squeeze(-1)
            valsTrue = valsTen[:,1:]
            valsLoss = valsLossFunction(valsPred, valsTrue)

            timesPred = timesPred.contiguous().view(-1, timesPred.shape[-1])
            timesTrue = timesTen[:,1:].contiguous().view(-1)
            timesLoss = timesLossFunction(timesPred, timesTrue)

            totalLoss = tokLoss + timesLoss + valsLoss 

            totalLoss.backward()
            optimizer.step()

            lrRate = optimizer.param_groups[0]['lr']
            dataIter.set_description('Epoch: %d Loss: %5.3f lr: %7.5f'%
                                     (epoch, totalLoss.item(), lrRate))
            
            trainLoss += totalLoss.item()
            cnt += 1

            #for OneCycleLR
            lrScheduler.step()

        trainLoss = trainLoss / cnt
        trainLosses.append(trainLoss)
        print('Training Loss:', trainLoss)

        model.eval()
        validLoss = 0
        cnt = 0
        with torch.no_grad():
            dataIter = tqdm(validloader)
            for item in dataIter:

                died = item['died']

                tokTen = item['tokTen'].to(device)
                valsTen = item['valsTen'].to(device)
                timesTen = item['timesTen'].to(device)

                inTokTen = tokTen[:,:-1]
                inValsTen = valsTen[:,:-1]
                inTimesTen = timesTen[:,:-1]

                mask = createMask(inTokTen, padIdx).to(device)

                out = model(inTokTen, inValsTen, inTimesTen, mask)
                tokPred, valsPred, timesPred = model.unsupervised(out)

                tokPred = tokPred.contiguous().view(-1, tokPred.shape[-1])
                tokTrue = tokTen[:,1:].contiguous().view(-1)
                tokLoss = tokLossFunction(tokPred, tokTrue)

                valsPred = valsPred.squeeze(-1)
                valsTrue = valsTen[:,1:]
                valsLoss = valsLossFunction(valsPred, valsTrue)

                timesPred = timesPred.contiguous().view(-1, timesPred.shape[-1])
                timesTrue = timesTen[:,1:].contiguous().view(-1)
                timesLoss = timesLossFunction(timesPred, timesTrue)

                totalLoss = tokLoss + timesLoss + valsLoss

                dataIter.set_description('Epoch: %d Loss: %5.3f lr: %7.5f'%
                                        (epoch, totalLoss.item(), lrRate))
                
                validLoss += totalLoss.item()
                cnt += 1


        validLoss = validLoss / cnt
        validLosses.append(validLoss)
        print('validation Loss:', validLoss)

        if validLoss < bestLoss:
            print('Saving State')
            bestLoss = validLoss
            torch.save(model.state_dict(), 'model.pt')
            torch.save(optimizer.state_dict(), 'optimizer.pt')
            torch.save(lrScheduler.state_dict(), 'lrScheduler.pt')

        #for reduceLROnPlateau
        #lrScheduler.step(validLoss)

    with open('losses.txt', 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['trainLoss', 'validLoss'])
        for i, j in zip(trainLosses, validLosses):
            csvwriter.writerow([i, j])



if __name__ == '__main__':
    main()