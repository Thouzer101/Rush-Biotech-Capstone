import os
import csv
import sqlite3
import pickle
import json

from tqdm import tqdm
import numpy as np
from collections import defaultdict

from dataset import getColNames

def main():

    hadmIds = []
    with open('hadm_ids.txt', 'r') as txtFile:
        hadmId = txtFile.readline()
        while hadmId:
            hadmIds.append(hadmId.strip())
            hadmId = txtFile.readline() 

    dbPath = 'mimic_iv.db'
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()

    colNames = getColNames(cursor, 'labevents')
    for col in colNames:
        print(col)

    #get the tokens
    #tokens will be used to build index for embeddings
    tokens = defaultdict(int)
    chartEvents = defaultdict(list)
    labEvents = defaultdict(list)
    for hadmId in tqdm(hadmIds):
        #demographic data
        cursor.execute("SELECT * FROM admissions WHERE hadm_id=?", (hadmId,))
        rows = cursor.fetchall()
        if len(rows) != 1:
            print('Something is wrong hadmId', hadmId)
        row = rows[0]

        subjectId = row[1]
        maritalStatus = row[10] 
        if len(maritalStatus) == 0:
            maritalStatus = 'MARITAL_STATUS_UNKNOWN'
        
        race = row[11]

        tokens[maritalStatus] += 1
        tokens[race] += 1

        cursor.execute("SELECT * FROM patients WHERE subject_id=?", (subjectId,))
        rows = cursor.fetchall()
        if len(rows) != 1:
            print('something wrong with subjectId', subjectId)
        row = rows[0]

        gender = row[1]
        age = 'age_' + row[2]

        tokens[gender] += 1
        tokens[age] += 1

        #chart events
        cursor.execute("SELECT * FROM chartevents WHERE hadm_id=?", (hadmId,))
        rows = cursor.fetchall()

        for row in rows:
            itemId = row[6]
            itemVal = row[7]
            tokens[itemId] += 1
            chartEvents[itemId].append(itemVal)

        #lab events
        cursor.execute("SELECT * FROM labevents WHERE hadm_id=?", (hadmId,))
        rows = cursor.fetchall()

        for row in rows:
            itemId = row[5]
            itemVal = row[8]
            tokens[itemId] += 1
            labEvents[itemId].append(itemVal)

    
    tokens = [item for item in tokens.items()]
    tokens.sort(key=lambda x:x[0])
    with open('tokens.txt', 'w', encoding='utf-8') as txtFile:
        for item in tokens:
            line = json.dumps({item[0]:item[1]})
            txtFile.write(line + '\n')

    with open('chartEvents.txt', 'w', encoding='utf-8') as txtFile:
        for item in chartEvents.items():
            line = json.dumps({item[0]:item[1]})
            txtFile.write(line + '\n')

    with open('labEvents.txt', 'w', encoding='utf-8') as txtFile:
        for item in labEvents.items():
            line = json.dumps({item[0]:item[1]})
            txtFile.write(line + '\n')

if __name__ == '__main__':
    main()