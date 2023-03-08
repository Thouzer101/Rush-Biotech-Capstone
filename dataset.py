import os
import sqlite3
import pickle
import csv
import json

import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

epsilon = 0.0000001

def timeStampToSecs(timeStamp):
    return datetime.strptime(timeStamp, '%Y-%m-%d %H:%M:%S').timestamp()

def timeDiff(startStr, stopStr):
    start = timeStampToSecs(startStr)
    stop = timeStampToSecs(stopStr)
    return stop - start

def secToHr(secs):
    return secs / 3600

def readableSecs(seconds):
    seconds = int(seconds)
    return timedelta(seconds=seconds)

def getColNames(cursor, tableName):
    cursor.execute("PRAGMA table_info(%s)"%tableName)
    colNames = cursor.fetchall()
    return [{'colIdx':item[0], 'colName':item[1], 'colType':item[2]} for item in colNames]

def isFloat(inStr):
    try:
        a = float(inStr)
    except (TypeError, ValueError):
        return False
    else:
        return True

def isListNumeric(arr):
    keys = set()
    for item in arr:
        if isFloat(item):
            keys.add(item)

    if len(keys) > 9:
        return True
    else:
        return False

    '''
    nItems = 0
    nNumeric = 0
    for item in arr:

        if len(item):
            nItems += 1

        if isFloat(item):
            nNumeric += 1

    if nItems == 0:
        return 0

    return float(nNumeric) / nItems
    '''

class MimicDataset(Dataset):
    def __init__(self, hadmIdsFile, seqLen=1024, maxHrs=24*60,
                 dbFile='dataset.db', tokenFile='tokens.txt'):
        
        if not('train' in hadmIdsFile or 'valid' in hadmIdsFile):
            print('Invalid file:', hadmIdsFile)
            return

        hadmIds = []
        with open(hadmIdsFile, 'r', encoding='utf-8') as txtFile:
            line = txtFile.readline()
            while line:
                hadmIds.append(line.strip())
                line = txtFile.readline()   

        self.hadmIds = hadmIds

        self.seqLen = seqLen
        self.maxHrs = maxHrs

        conn = sqlite3.connect(dbFile) 
        cursor = conn.cursor()
        self.cursor = cursor

        tokToIdx = {}
        tokToIdx['<pad>'] = 0
        tokToIdx['<cls>'] = 1
        tokToIdx['<end>'] = 2
        with open(tokenFile, 'r', encoding='utf-8') as txtFile:
            line = txtFile.readline()
            cnt = 3
            while line:
                jsonData = json.loads(line)
                key = list(jsonData.keys())[0]
                tokToIdx[key] = cnt
                line = txtFile.readline()
                cnt += 1

        idxToTok = {} 
        for item in tokToIdx.items():
            idxToTok[item[1]] = item[0]

        self.tokToIdx = tokToIdx
        self.idxToTok = idxToTok
    
    def __len__(self):
        return len(self.hadmIds)

    def __getitem__(self, idx):
        hadmId = self.hadmIds[idx]

        self.cursor.execute("SELECT * FROM data WHERE hadmId=?", (hadmId,))
        item = self.cursor.fetchall()[0]
        item = json.loads(item[1])
        
        #label for classification
        died = eval(item['died'])
        if died:
            died = torch.tensor([1]).float()
        else:
            died = torch.tensor([0]).float()

        #build token tensor, vals tensor, and time tensor
        tokTen = [self.tokToIdx['<cls>']] + [self.tokToIdx[demo] for demo in item['demographics']]
        valsTen = len(tokTen) * [0.0]
        timesTen = len(tokTen) * [0]

        events = item['events']
        maxEvents = self.seqLen - len(tokTen) - 1
        if len(events) > maxEvents:
            events = np.random.choice(events, maxEvents, replace=False).tolist()

        #sort just in case
        events.sort(key= lambda x: x['eventTime'])
        
        for event in events:
            tokTen.append(self.tokToIdx[event['eventId']])
            valsTen.append(event['eventVal'])
            timesTen.append(event['eventTime'])
            lastTime = event['eventTime']

        tokTen.append(self.tokToIdx['<end>'])
        valsTen.append(0)
        timesTen.append(lastTime)

        tokTen = torch.tensor(tokTen).long()
        valsTen = torch.tensor(valsTen).float()
        timesTen = torch.tensor(timesTen).long()

        timesTen = torch.clip(timesTen, min=0, max=self.maxHrs - 1)

        return {'died':died, 'tokTen':tokTen, 'valsTen':valsTen, 'timesTen':timesTen}


def collate_fn(batchItem):
    batchSize = len(batchItem)

    maxLen = 0
    for item in batchItem:
        itemLen = len(item['tokTen'])
        maxLen = max(itemLen, maxLen)

    batchDied = torch.zeros(batchSize).float()
    batchTok = torch.zeros(batchSize, maxLen).long()
    batchVals = torch.zeros(batchSize, maxLen).float()
    batchTimes = torch.zeros(batchSize, maxLen).long()

    for i in range(batchSize):
        item = batchItem

        batchDied[i] = item[i]['died']

        tokTen = item[i]['tokTen']
        batchTok[i,:tokTen.shape[0]] = tokTen

        valsTen = item[i]['valsTen']
        batchVals[i,:valsTen.shape[0]] = valsTen

        timesTen = item[i]['timesTen']
        batchTimes[i,:timesTen.shape[0]] = timesTen

    return {'died':batchDied, 'tokTen':batchTok, 'valsTen':batchVals, 'timesTen':batchTimes}


def main():

    #trainset has 52946 entries with 5312 deaths
    #validset has 13237 entries with 1270 deaths
    #subtrainset has 6787 entries with 1925 deaths
    #subvalidset has 1885 entries with 487

    trainset = MimicDataset('train.txt') 
    validset = MimicDataset('valid.txt') 
    print(len(trainset), len(validset))

    testset = trainset

    #nItems = range(len(testset))
    nItems = tqdm(range(len(testset)))
    #nItems = range(0)
    items = []
    for i in nItems:
        item = testset[i]
        items.append(len(item['tokTen']))

    items = np.array(items)
    print(items.min(), items.max(), items.mean(), items.std())
    return

    nWorkers = 0 #HAS to be 0 since you cannot fork sqlite cursor into multiple workers
    batchSize = 32
    trainloader = DataLoader(trainset, shuffle=False, batch_size=batchSize, num_workers=nWorkers, collate_fn=collate_fn)
    validloader = DataLoader(validset, shuffle=False, batch_size=batchSize, num_workers=nWorkers, collate_fn=collate_fn)

    for item in tqdm(trainloader):
        item


if __name__ == '__main__':
    main()