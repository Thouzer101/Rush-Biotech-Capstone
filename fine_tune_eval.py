import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import csv
from tqdm import tqdm

from model import MimicModel, createMask
from dataset import MimicDataset, collate_fn

def main():

    batchSize = 16
    validset = MimicDataset('subvalid.txt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    nTokens = len(validset.tokToIdx.keys())
    nLayers = 4
    embSize = 256
    peSize = 64
    maxHrs = validset.maxHrs
    
    model = MimicModel(nLayers, embSize, nTokens, peSize, device, maxHrs=maxHrs)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)
    model.load_state_dict(torch.load('fine_tune_model_first_24.pt'))
    model.eval()

    lossFunction = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()


    csvFile = open('no_pretrain_results_24.csv', 'w', newline='')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['hadm_id', 'Label', 'Prediction'])

    for item in tqdm(validset):
        hadmId = item['hadmId']
        died = item['died'].to(device)

        tokTen = item['tokTen'].unsqueeze(0).to(device)
        valsTen = item['valsTen'].unsqueeze(0).to(device)
        timesTen = item['timesTen'].unsqueeze(0).to(device)


        out = model(tokTen, valsTen, timesTen)
        pred = model.supervised(out)
        pred = pred.squeeze(-1)
        pred = sigmoid(pred)

        csvWriter.writerow([hadmId, died.item(), pred.item()])




if __name__ == '__main__':
    main()