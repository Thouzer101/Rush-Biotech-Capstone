import sqlite3
import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from dataset import timeStampToSecs

def screenByDiagnosis(hadmIds, diagnosis, dbFile):
    
    conn = sqlite3.connect(dbFile)
    cursor = conn.cursor()

    retHadmIds = []
    nDeaths = []
    for hadmId in hadmIds:
        #check if patient has sepsis
        cursor.execute("SELECT * FROM data where hadmId=?", (hadmId,))
        row = cursor.fetchall()[0]
        entry = row[1]

        jsonData = json.loads(entry)
        diagnoses = jsonData['diagnoses']
        for item in diagnoses.items():
            title = item[1]['title']
            if diagnosis in title.lower():
                retHadmIds.append(hadmId)

    conn.close()
    return retHadmIds

def screenByFirstHadmId(hadmIds, dbFile):
    conn = sqlite3.connect(dbFile)
    cursor = conn.cursor()

    subjectIds = defaultdict(list)
    for hadmId in hadmIds:
        cursor.execute("SELECT * FROM admissions WHERE hadm_id=?", (hadmId,))
        row = cursor.fetchall()[0]
        subjectId = row[1]

        subjectIds[subjectId].append(hadmId)

    #get the first hadmId
    retHadmIds = []
    for item in subjectIds.items():
        key = item[0]
        hadmIds = item[1]
        if len(hadmIds) == 1:
            hadmId = hadmIds[0]
            retHadmIds.append(hadmId)
            continue
        
        firstTime = float('inf')
        firstHadmId = 0
        for hadmId in hadmIds:
            cursor.execute("SELECT * FROM admissions WHERE hadm_id=?", (hadmId,))
            row = cursor.fetchall()[0]
            hospAdmTime = row[2]
            edAdmTime = row[12]

            if len(hospAdmTime) != 0 and len(edAdmTime) != 0:
                startTime = min(timeStampToSecs(hospAdmTime), timeStampToSecs(edAdmTime))
            elif len(hospAdmTime) != 0:
                startTime = timeStampToSecs(hospAdmTime) 
            else:
                startTime = timeStampToSecs(edAdmTime)

            if startTime < firstTime:
                firstTime = startTime
                firstHadmId = hadmId

        retHadmIds.append(firstHadmId) 

    conn.close()
    return retHadmIds

def main():

    hadmIds = []
    with open('hadm_ids.txt', 'r', encoding='utf-8') as txtFile:
        line = txtFile.readline()
        while line:
            hadmIds.append(line.strip())
            line = txtFile.readline()

    print('Total hadmIds:', len(hadmIds))
    #make train_hadm_ids.txt and test_hadm_ids.txt
    splitLen = int(0.8 * len(hadmIds))
    trainset = hadmIds[:splitLen]
    validset = hadmIds[splitLen:]

    with open('train.txt', 'w', encoding='utf-8') as txtFile:
        for hadmId in trainset:
            txtFile.write(hadmId + '\n')

    with open('valid.txt', 'w', encoding='utf-8') as txtFile:
        for hadmId in validset:
            txtFile.write(hadmId + '\n')

    print('Trainset:', len(trainset), 'Validset:', len(validset)) 
    #check to see if there an acceptable number of sepsis and mortality in
    #also follow what was in the paper, like remove later hadmid from same patient
    #hadmId split at 80%
    dbFile = 'dataset.db'
    subtrainset = screenByDiagnosis(trainset, 'sepsis', dbFile)
    subvalidset = screenByDiagnosis(validset, 'sepsis', dbFile)

    print('Screened by diagnosis - Trainset:', len(subtrainset), 'Validset:', len(subvalidset)) 
    #get the subject_id for each hadmid
    # and only use the earlier event
    dbFile = 'mimic_iv.db'
    subtrainset = screenByFirstHadmId(subtrainset, dbFile)
    subvalidset = screenByFirstHadmId(subvalidset, dbFile)

    print('Screened by first hadmId - Trainset:', len(subtrainset), 'Validset:', len(subvalidset)) 

    with open('subtrain.txt', 'w', encoding='utf-8') as txtFile:
        for hadmId in subtrainset:
            txtFile.write(hadmId + '\n')

    with open('subvalid.txt', 'w', encoding='utf-8') as txtFile:
        for hadmId in subvalidset:
            txtFile.write(hadmId + '\n')

if __name__ == '__main__':
    main()