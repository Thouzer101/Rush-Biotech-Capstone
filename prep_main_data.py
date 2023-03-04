import os
import sqlite3
import pickle

import numpy as np
from tqdm import tqdm

from dataset import timeDiff, readableSecs, timeStampToSecs, getColNames


def main():

    dbPath = 'mimic_iv.db'
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()

    for col in getColNames(cursor, 'admissions'):
        print(col)

    '''
    #get all hadmids
    cursor.execute("SELECT * FROM admissions")
    hadmIds = []
    total = 0
    for row in tqdm(cursor):
        total += 1
        hadmId = row[0]

        hospAdmTime = row[2]
        hospDisTime = row[3]
        edAdmTime = row[12]
        edDisTime = row[13]

        if len(hospAdmTime) != 0 and len(edAdmTime) != 0:
            startTime = min(timeStampToSecs(hospAdmTime), timeStampToSecs(edAdmTime))
        elif len(hospAdmTime) != 0:
            startTime = timeStampToSecs(hospAdmTime) 
        else:
            startTime = timeStampToSecs(edAdmTime) 

        if len(hospDisTime) != 0 and len(edDisTime) != 0:
            stopTime = max(timeStampToSecs(hospDisTime), timeStampToSecs(edDisTime))
        elif len(hospDisTime) != 0:
            stopTime = timeStampToSecs(hospDisTime)
        else:
            stopTime = timeStampToSecs(edDisTime)

        stayTime = stopTime - startTime

        #don't use hadmid if the times are weird or its longer than 30 days
        #60 secs, 60 mins, 24 hrs, 30 days
        if stayTime < 0 or stayTime > 60 * 60 * 24 * 30:
            continue
        
        hadmIds.append(hadmId)
    
    print('number of hadmIds:', len(hadmIds))
    pickle.dump(hadmIds, open('hadmdIds.bin', 'wb')) 
    #'''

    '''
    hadmIds = pickle.load(open('hadmdIds.bin', 'rb'))

    #check to see if hadmIds has events since many do not
    validHadmIds = []
    totalChartEvents = []
    totalLabEvents = []
    for hadmId in tqdm(hadmIds):
        #cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM chartevents WHERE hadm_id=?", (hadmId,))
        cursor.execute("SELECT * FROM chartevents WHERE hadm_id=?", (hadmId,))

        rows = cursor.fetchall()
        nChartEvents = len(rows)
        if nChartEvents != 0: 
            totalChartEvents.append(nChartEvents)

        #cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM labevents WHERE hadm_id=?", (hadmId,))
        cursor.execute("SELECT * FROM labevents WHERE hadm_id=?", (hadmId,))

        rows = cursor.fetchall()
        nLabEvents = len(rows)    
        if nLabEvents != 0:
            totalLabEvents.append(nLabEvents)

        if nLabEvents != 0 and nChartEvents != 0:
           validHadmIds.append(hadmId)
        

    totalChartEvents = np.array(totalChartEvents)
    totalLabEvents = np.array(totalLabEvents)

    pickle.dump(validHadmIds, open('validHadmids.bin', 'wb'))
    pickle.dump(totalChartEvents, open('totalChartEvents.bin', 'wb'))
    pickle.dump(totalLabEvents, open('totalLabEvents.bin', 'wb'))

    return
    #'''

    validHadmIds = pickle.load(open('validHadmids.bin', 'rb'))
    totalChartEvents = pickle.load(open('totalChartEvents.bin', 'rb'))
    totalLabEvents = pickle.load(open('totalLabEvents.bin', 'rb'))

    print('number of valid HadmIds:', len(validHadmIds))
    print('Chart Event Stats: mean: %5.3f std: %5.3f min: %5.3f max: %5.3f'
            %(totalChartEvents.mean(),totalChartEvents.std(), totalChartEvents.min(), totalChartEvents.max()))
    print('Lab Event Stats: mean: %5.3f std: %5.3f min: %5.3f max: %5.3f'
            %(totalLabEvents.mean(),totalLabEvents.std(), totalLabEvents.min(), totalLabEvents.max()))

    np.random.shuffle(validHadmIds)

    with open('hadm_ids.txt', 'w') as txtFile:
        for hadmId in validHadmIds:
            txtFile.write(hadmId + '\n')

if __name__ == '__main__':
    main()