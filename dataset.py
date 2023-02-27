import os
import sqlite3
import pickle
import csv

import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

def timestamp_to_secs(timeStamp):
    return datetime.strptime(timeStamp, '%Y-%m-%d %H:%M:%S').timestamp()

def time_diff(startStr, stopStr):
    start = timestamp_to_secs(startStr)
    stop = timestamp_to_secs(stopStr)
    #start = datetime.strptime(startStr, '%Y-%m-%d %H:%M:%S').timestamp()
    #stop = datetime.strptime(stopStr, '%Y-%m-%d %H:%M:%S').timestamp()
    return stop - start

def is_valid_time(curTimeStr, startStr, stopStr):
    cur = datetime.strptime(curTimeStr, '%Y-%m-%d %H:%M:%S').timestamp()
    start = datetime.strptime(startStr, '%Y-%m-%d %H:%M:%S').timestamp()
    stop = datetime.strptime(stopStr, '%Y-%m-%d %H:%M:%S').timestamp()
    if cur < start or cur > stop:
        return False
    return True

def sec_to_hr(secs):
    return secs / 3600

def readable_secs(seconds):
    seconds = int(seconds)
    return timedelta(seconds=seconds)

def get_col_idx(colInfo, colName):
    for col in colInfo:
        curColName = col[1]
        colIdx = col[0]
        if curColName == colName:
            return colIdx

    print(colName, 'not found')
    return None

class MimicDataset(Dataset):
    def __init__(self, split, hadmIdFile='valid_hadmid.txt', dbFile='mimic_iv.db', 
                 categoriesFile='categories.csv', chartEventItemStatsFile='chartevents_items.bin'):
        #create dict to convert category to idx
        categories = {}
        categories['<pad>'] = 0
        categories['<end>'] = 1
        with open(categoriesFile) as csvFile:
            csvReader = csv.reader(csvFile)
            cnt = 2
            for row in csvReader:
                categories[row[0]] = cnt
                cnt += 1
        self.categories = categories

        categoryIds = []
        for item in categories.items():
            categoryIds.append(item[0])
        self.categoryIds = categoryIds

        self.chartEventItemStats = pickle.load(open(chartEventItemStatsFile, 'rb'))

        hadmIds = []
        with open(hadmIdFile, 'r') as txtFile:
            line = txtFile.readline().strip()
            while line:
                hadmIds.append(line)
                line = txtFile.readline().strip()   
        nItems = int(len(hadmIds)  * 0.8)

        if split == 'train':     
            self.hadmIds = hadmIds[:nItems]
        elif split == 'valid':
            self.hadmIds = hadmIds[nItems:]
        else:
            print('Invalid split')
            return

        self.conn = sqlite3.connect(dbFile)
        self.cursor = self.conn.cursor()
    
    def __len__(self):
        return len(self.hadmIds)

    def __getitem__(self, idx):
        hadmId = self.hadmIds[idx]

        #get demographic data
        self.cursor.execute("""
                    SELECT * FROM admissions WHERE hadm_id = ?
                    """, (hadmId,))
        #headers = [i[0] for i in self.cursor.description]
        row = self.cursor.fetchall()[0]

        subjectId = row[1]
        admTime = row[2]
        disTime = row[3]
        maritalStatus = row[10]
        if len(row[10]) == 0:
            maritalStatus = 'MARITAL_STATUS_UNKNOWN'
        race = row[11]

        self.cursor.execute("""
                    SELECT * FROM patients WHERE subject_id = ?
                    """, (subjectId,))
        row = self.cursor.fetchall()[0]
        
        gender = row[1]
        age = 'age_' + row[2]

        #get events
        self.cursor.execute("SELECT * FROM chartevents WHERE hadm_id = ?", (hadmId,))
        rows = self.cursor.fetchall()
        headers = [i[0] for i in self.cursor.description]

        events = []
        for row in rows:
            event = {'chartTime': row[4], 'itemId': row[6], 
                     'itemVal': row[7], 'itemUnit': row[9]}
            events.append(event)

        events.sort(key=lambda x:timestamp_to_secs(x['chartTime']))

        #TODO include procedures_icd
        #TODO include omr
        #TODO microbiologyevents
        #TODO prescriptions??
        #TODO emar??
        #TODO poe??

        #convert demographics and events into indices for embedding
        demographics = [age, race, maritalStatus, gender]
        demoIdx = [self.categories[demo] for demo in demographics]
        
        eventIdx = []
        eventVals = []
        eventTimes = []
        startTime = timestamp_to_secs(admTime)
        for event in events:
            #print(event)
            itemId = event['itemId']
            eventIdx.append(self.categories[itemId])
            paramType = self.chartEventItemStats[itemId]['paramType']
            if 'Numeric' in paramType:
                itemVal = float(event['itemVal'])
                mean = self.chartEventItemStats[itemId]['mean']
                std = self.chartEventItemStats[itemId]['std']
                eventVals.append((itemVal - mean) / std)
            else:
                eventVals.append(0)

            eventTime = timestamp_to_secs(event['chartTime'])
            normTime = sec_to_hr(eventTime - startTime)
            normTime = max(normTime, 0)
            normTime = np.round(normTime).astype(int)
            eventTimes.append(normTime)
        print(eventIdx)
        print(eventVals)
        print(eventTimes)

        exit(0)
            


def main():

    trainset = MimicDataset('train') 
    validset = MimicDataset('valid') 
    print(len(trainset), len(validset))

    trainset[0]

    

if __name__ == '__main__':
    main()