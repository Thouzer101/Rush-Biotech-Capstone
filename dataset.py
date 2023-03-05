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
    def __init__(self, hadmIdsFile, seqLen=1024, groupSize=8,
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
        self.groupSize = groupSize

        conn = sqlite3.connect(dbFile) 
        cursor = conn.cursor()
        self.cursor = cursor

        tokToIdx = {}
        tokToIdx['<pad>'] = -1
        tokToIdx['<cls>'] = 0
        with open(tokenFile, 'r', encoding='utf-8') as txtFile:
            line = txtFile.readline()
            cnt = 1
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

        died = eval(item['died'])
        if died:
            died = torch.tensor([1]).float()
        else:
            died = torch.tensor([0]).float()

        demoIdx = torch.tensor(item['demoIdx']).long()

        events = item['events']
        allEvents = []
        allEventTimes = []
        group = []
        i = 0

        #for end token
        increment = np.ceil(float(len(demoIdx) + len(died)) / self.groupSize) * self.groupSize
        while i < len(events) - 1 and len(allEventTimes) < self.seqLen - increment:
            #make a group of events for a given hr 
            curHr = events[i]['eventTime']
            nextHr = events[i+1]['eventTime']

            if curHr == nextHr:
                group.append(events[i])
                i += 1
            else:
                group.append(events[i])

                for groupSize in range(self.groupSize):
                    allEventTimes.append(curHr)

                if len(group) < self.groupSize + 1:
                    while len(group) < self.groupSize:
                        group.append({'eventIdx':-1, 'eventVal':0.0})
                    allEvents.append(group)
                else:
                    group = np.random.choice(group, self.groupSize, replace=False).tolist()
                    allEvents.append(group)

                group = []
                i += 1

        eventTimes = torch.tensor(allEventTimes).long()
        #idx, val, and time
        eventIndices = []
        eventVals = []
        for group in allEvents:
            for event in group:
                eventIdx = event['eventIdx']
                eventVal = event['eventVal']
                eventIndices.append(eventIdx)
                eventVals.append(eventVal)

        eventIndices = torch.tensor(eventIndices).long()
        eventVals = torch.tensor(eventVals).float()

        events = {'eventIndices':eventIndices, 'eventVals':eventVals, 'eventTimes':eventTimes}

        return {'died':died, 'demoIdx':demoIdx, 'events':events}



def collate_fn(batchItem):
    padIdx = -1
    #find longest event
    batchSize = len(batchItem)
    maxLen = 0
    for item in batchItem:
        itemLen = len(item['events']['eventVals'])
        maxLen = max(itemLen, maxLen)

    batchDied = torch.zeros(batchSize).fill_(padIdx).long()

    demoLen = batchItem[0]['demoIdx'].shape[0]
    batchDemoIdx = torch.zeros(batchSize, demoLen).fill_(padIdx).long()

    batchEventIndices = torch.zeros(batchSize, maxLen).fill_(padIdx).long()
    batchEventVals = torch.zeros(batchSize, maxLen).float()
    batchEventTimes = torch.zeros(batchSize, maxLen).fill_(padIdx).long()

    for i in range(batchSize):
        batchDied[i] = batchItem[i]['died']

        batchDemoIdx[i,:] = batchItem[i]['demoIdx']

        eventIndices = batchItem[i]['events']['eventIndices']
        batchEventIndices[i,:eventIndices.shape[0]] = eventIndices

        eventVals = batchItem[i]['events']['eventVals']
        batchEventVals[i,:eventVals.shape[0]] = eventVals

        eventTimes = batchItem[i]['events']['eventTimes']
        batchEventTimes[i,:eventTimes.shape[0]] = eventTimes

    events = {'eventIndices':batchEventIndices, 'eventVals':batchEventVals, 'eventTimes':batchEventTimes}
    return {'died':batchDied, 'demoIdx':batchDemoIdx, 'events':events}


def main():

    #trainset has 52946 entries with 5312 deaths
    #validset has 13237 entries with 1270 deaths
    #subtrainset has 6787 entries with 1925 deaths
    #subvalidset has 1885 entries with 487

    trainset = MimicDataset('train.txt') 
    validset = MimicDataset('valid.txt') 
    print(len(trainset), len(validset))

    testset = trainset
    #nItems = tqdm(range(len(testset)))
    nItems = range(0)
    items = []
    for i in nItems:
        item = testset[i]
        items.append(len(item['events']['eventVals']))

    items = np.array(items)
    #print(items.min(), items.max(), items.mean(), items.std())


    nWorkers = 0 #HAS to be 0 since you cannot fork sqlite cursor into multiple workers
    batchSize = 32
    trainloader = DataLoader(trainset, shuffle=False, batch_size=batchSize, num_workers=nWorkers, collate_fn=collate_fn)
    validloader = DataLoader(validset, shuffle=False, batch_size=batchSize, num_workers=nWorkers, collate_fn=collate_fn)

    for item in tqdm(trainloader):
        item


if __name__ == '__main__':
    main()