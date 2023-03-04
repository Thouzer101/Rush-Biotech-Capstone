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
    def __init__(self, hadmIdsFile, dbFile='dataset.db', tokenFile='tokens.txt'):

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

        conn = sqlite3.connect(dbFile) 
        cursor = conn.cursor()
        self.cursor = cursor

        tokToIdx = {}
        tokToIdx['<pad>'] = 0
        tokToIdx['<end>'] = 1
        tokToIdx['<cls>'] = 2
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



def main():

    trainset = MimicDataset('train.txt') 
    validset = MimicDataset('valid.txt') 
    print(len(trainset), len(validset))


    

if __name__ == '__main__':
    main()