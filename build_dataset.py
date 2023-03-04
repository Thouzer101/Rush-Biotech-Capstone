import os
import sqlite3
import json
import csv
import pickle

from tqdm import tqdm
import numpy as np
from collections import defaultdict

from dataset import getColNames, timeStampToSecs, readableSecs, secToHr, isFloat, isListNumeric, epsilon

def main():

    #precalculate the data and store in sqlite3 database
    dbFile = 'dataset.db'
    if os.path.isfile(dbFile):
        os.remove(dbFile)

    conn = sqlite3.connect(dbFile)
    cursor = conn.cursor()

    cmdStart = "CREATE TABLE IF NOT EXISTS data ("
    cols = ', '.join(['hadmId text PRIMARY KEY', 'entry text'])
    cmdEnd = ");"

    cmd = cmdStart + cols + cmdEnd
    cursor.execute(cmd)

    #make it fast insert
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = normal;")

    hadmIds = []
    with open('hadm_ids.txt', 'r', encoding='utf-8') as txtFile:
        line = txtFile.readline()
        while line:
            hadmIds.append(line.strip())
            line = txtFile.readline()

    #icd codes
    icdCodes = {}
    with open('../mimiciv/2.0/hosp/d_icd_diagnoses.csv', 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        row = next(csvReader)
        for row in csvReader:
            code = row[0]
            version = row[1]
            title = row[2]
            icdCodes[code] = {'version':version, 'title':title}

    icdTitles = {}
    for item in icdCodes.items():
        key = item[1]['title']
        version = item[1]['version']
        code = item[0]
        if key not in icdTitles.keys():
            icdTitles[key] = {'version':version, 'code':[code]}
        else:
            icdTitles[key]['code'].append(code)

    #convert to idx
    tokToIdx = {}
    tokToIdx['<pad>'] = 0
    tokToIdx['<end>'] = 1
    tokToIdx['<cls>'] = 2
    with open('tokens.txt', 'r', encoding='utf-8') as txtFile:
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

    #get event paramType
    chartEventTypes = {}
    with open('../mimiciv/2.0/icu/d_items.csv', 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        header = next(csvReader)
        for row in csvReader:
            itemId = row[0]
            paramType = row[6]
            chartEventTypes[itemId] = paramType


    ''' 
    chartEventStats = defaultdict(dict)
    with open('chartEvents.txt', 'r', encoding='utf-8') as txtFile:
        line = txtFile.readline()
        while line:
            jsonData = json.loads(line)
            itemId = list(jsonData.keys())[0]

            paramType = chartEventTypes[itemId]
            chartEventStats[itemId]['paramType'] = paramType
            
            if 'Numeric' in paramType:
                vals = np.array(jsonData[itemId], dtype=float)
                chartEventStats[itemId]['mean'] = vals.mean()
                std = vals.std()
                if std != 0:
                    chartEventStats[itemId]['std'] = std
                else:
                    chartEventStats[itemId]['std'] = 1.0
            else:
                vals = jsonData[itemId]
                valsDict = {}
                cnt = 0
                for val in vals:
                    if len(val) == 0:
                        continue
                    if val not in valsDict.keys():
                        valsDict[val] = cnt
                        cnt += 1
                chartEventStats[itemId]['factors'] = valsDict
            
            line = txtFile.readline()

    pickle.dump(chartEventStats, open('chartEventStats.bin', 'wb'))
    return
    #'''

    '''
    labEventStats = defaultdict(dict)
    with open('labEvents.txt', 'r', encoding='utf-8') as txtFile:
        line = txtFile.readline()
        while line:
            jsonData = json.loads(line)
            itemId = list(jsonData.keys())[0] 
            #check to see if vals are numeric
            vals = jsonData[itemId]
            if isListNumeric(vals):
                floatVals = []
                for val in vals:
                    if isFloat(val):
                        floatVals.append(val)

                vals = np.array(floatVals, dtype=float)
                labEventStats[itemId]['mean'] = vals.mean()
                std = vals.std()
                if std == 0:
                    labEventStats[itemId]['std'] = 1.0
                else:
                    labEventStats[itemId]['std'] = std
            else:
                valsDict = {}
                cnt = 0
                for val in vals:
                    if len(val) == 0:
                        continue
                    if val not in valsDict.keys():
                        valsDict[val] = cnt
                        cnt += 1
                labEventStats[itemId]['factors'] = valsDict

            line = txtFile.readline()

    pickle.dump(labEventStats, open('labEventStats.bin', 'wb'))
    return
    #'''

    chartEventStats = pickle.load(open('chartEventStats.bin', 'rb'))
    labEventStats = pickle.load(open('labEventStats.bin', 'rb'))

    #combine event stats
    eventStats = {}
    for item in chartEventStats.items():
        eventStats[item[0]] = item[1]

    for item in labEventStats.items():
        eventStats[item[0]] = item[1]

    srcDbFile = 'mimic_iv.db'
    srcConn = sqlite3.connect(srcDbFile)
    srcCursor = srcConn.cursor()

    #'''
    colNames = getColNames(srcCursor, 'labevents')
    for col in colNames:
        print(col)
    #'''

    for hadmId in tqdm(hadmIds):
        srcCursor.execute("SELECT * FROM admissions WHERE hadm_id=?", (hadmId,))

        row = srcCursor.fetchall()[0]
        subjectId = row[1]
        hospAdmTime = row[2]
        hospDisTime = row[3]
        deathTime = row[4]
        edAdmTime = row[12]
        edDisTime = row[13]
        
        if len(hospAdmTime) != 0 and len(edAdmTime) != 0:
            startTime = min(timeStampToSecs(hospAdmTime), timeStampToSecs(edAdmTime))
        elif len(hospAdmTime) != 0:
            startTime = timeStampToSecs(hospAdmTime) 
        else:
            startTime = timeStampToSecs(edAdmTime) 

        #if died within 30 day
        if len(deathTime) != 0:
            deathTime = timeStampToSecs(deathTime)
            relDeathTime = deathTime - startTime
            if relDeathTime < 60 * 60 * 24 * 30:
                died = 'True'
            else:
                died = 'False'
        else:
            died = 'False'

        #get demographics
        maritalStatus = row[10]
        if len(maritalStatus) == 0:
            maritalStatus = 'MARITAL_STATUS_UNKNOWN'
        race = row[11]        
        
        srcCursor.execute("SELECT * FROM patients WHERE subject_id=?", (subjectId,))
        row = srcCursor.fetchall()[0]

        gender = row[1]
        age = 'age_' + row[2]

        demographics = [age, race, maritalStatus, gender]
        demoIdx = [tokToIdx[demo] for demo in demographics]

        #get chartEvents
        srcCursor.execute("SELECT * FROM chartevents WHERE hadm_id=?", (hadmId,))
        rows = srcCursor.fetchall()

        #each event is composed of an index, value, and a time
        chartEvents = []
        for row in rows:
            itemId = row[6]
            itemVal = row[7]
            chartTime = timeStampToSecs(row[4])

            eventIdx = tokToIdx[itemId]

            paramType = chartEventTypes[itemId]
            if 'Numeric' in paramType:
                itemVal = float(itemVal)
                mean = eventStats[itemId]['mean']
                std = eventStats[itemId]['std']
                eventVal = (itemVal - mean) / std
                eventVal = np.clip(eventVal, a_min=-3, a_max=3)
            else:
                factors = eventStats[itemId]['factors']
                factorsLen = len(factors)
                if factorsLen == 0 or len(itemVal) == 0:
                    eventVal = -1.0
                else:
                    idx = factors[itemVal]
                    eventVal = float(idx) / (factorsLen - 1 + epsilon)

            eventTime = secToHr(chartTime - startTime)
            eventTime = max(eventTime, 0)
            eventTime = int(np.round(eventTime))
            
            event = {'eventIdx':eventIdx, 'eventVal':eventVal, 'eventTime':eventTime}
            chartEvents.append(event)


        #get lab_events        
        srcCursor.execute("SELECT * FROM labevents WHERE hadm_id=?", (hadmId,))
        rows = srcCursor.fetchall()

        labEvents = []
        for row in rows:
            itemId = row[5]
            itemVal = row[8]
            chartTime = timeStampToSecs(row[6])

            eventIdx = tokToIdx[itemId]

            if 'mean' in eventStats[itemId].keys():
                if isFloat(itemVal):
                    itemVal = float(itemVal)
                    mean = eventStats[itemId]['mean']
                    std = eventStats[itemId]['std']
                    eventVal = (itemVal - mean) / std
                    eventVal = np.clip(eventVal, a_min=-3, a_max=3)
                else:
                    eventVal = 0.0
            else:
                factors = eventStats[itemId]['factors']
                factorsLen = len(factors)
                if factorsLen == 0 or len(itemVal) == 0:
                    eventVal = -1.0
                else:
                    idx = factors[itemVal]
                    eventVal = float(idx) / (factorsLen - 1 + epsilon)

            eventTime = secToHr(chartTime - startTime)
            eventTime = max(eventTime, 0)
            eventTime = int(np.round(eventTime))

            event = {'eventIdx':eventIdx, 'eventVal':eventVal, 'eventTime':eventTime}
            labEvents.append(event)

        #combine chart and lab events
        events = []
        for event in chartEvents:
            events.append(event)
        for event in labEvents:
            events.append(event)

        events.sort(key=lambda x: x['eventTime'])

        #srcCursor.execute("EXPLAIN QUERY PLAN SELECT * FROM diagnoses_icd WHERE hadm_id=?", (hadmId,))
        srcCursor.execute("SELECT * FROM diagnoses_icd WHERE hadm_id=?", (hadmId,))
        rows = srcCursor.fetchall()

        diagnoses = {}
        for row in rows:
            code = row[4]
            version = row[5]
            diagnoses[code] = icdCodes[code]
 
        #TODO include input_events
        #TODO include ingredient_events
        #TODO include procedure_events
        #TODO prescriptions??
        #TODO include procedures_icd??
        #TODO microbiologyevents??
        #TODO emar??
        #TODO poe??

        entry = {'died': died, 'diagnoses':diagnoses, 'demoIdx':demoIdx, 'events':events}

        entryStr = json.dumps(entry)

        cursor.execute("INSERT INTO data(hadmId, entry) VALUES(?,?)", (hadmId, entryStr))
        conn.commit()








if __name__ == '__main__':
    main()