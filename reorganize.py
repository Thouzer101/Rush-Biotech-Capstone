import numpy as np
import os
import time
import csv
from datetime import datetime, timedelta
import pandas as pd
import pickle
from tqdm import tqdm
import sqlite3

def time_diff(startStr, stopStr):
    start = datetime.strptime(startStr, '%Y-%m-%d %H:%M:%S').timestamp()
    stop = datetime.strptime(stopStr, '%Y-%m-%d %H:%M:%S').timestamp()
    return stop - start

def is_valid_time(curTimeStr, startStr, stopStr):
    cur = datetime.strptime(curTimeStr, '%Y-%m-%d %H:%M:%S').timestamp()
    start = datetime.strptime(startStr, '%Y-%m-%d %H:%M:%S').timestamp()
    stop = datetime.strptime(stopStr, '%Y-%m-%d %H:%M:%S').timestamp()
    if cur < start or cur > stop:
        return False
    return True

def readable_secs(seconds):
    seconds = int(seconds)
    return timedelta(seconds=seconds)


def main():

    #create database
    conn = sqlite3.connect(r'E:\OneDrive - rush.edu\Research Capstone\Rush-Biotech-Capstone\patient.db')

    #create a table in the database
    cursor = conn.cursor()
    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS patientData(
                        hadmId integer PRIMARY KEY,
                        race text,
                        language text,
                        martialStatus text, 
                        admitTime text,
                        dischargeTime text, 
                        admType text,
                        admLocation text,
                        insurance text
                    );""")


    #put it into a sqlite database
    dataDir = r'E:\OneDrive - rush.edu\Research Capstone\mimiciv\2.0' 
    csvPath = os.path.join(dataDir, 'hosp', 'admissions.csv')
    with open(csvPath, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        row = next(csvReader)
        print(row)


    conn.close()



    return

    #reorganize MIMIC-IV to be based on hospital admissions
    #and convert into a json
    dataDir = r'E:\OneDrive - rush.edu\Research Capstone\mimiciv\2.0'    

    #first get all admission IDs
    csvPath = os.path.join(dataDir, 'hosp', 'admissions.csv')
    df = pd.read_csv(csvPath, engine='pyarrow', keep_default_na=False)
    print(df.columns)

    '''
    admitData = {}
    #times = []
    for idx, row in df.iterrows():
        item = {}

        startStr = str(row['admittime'])
        stopStr = str(row['dischtime'])
        stayTime = time_diff(startStr, stopStr)

        #times.append(stayTime)
        #don't include stay times that are negative
        #and longer than a month
        if stayTime <= 0 or stayTime > 604800 * 4:
            #print(hadmId, admitTime, dischargeTime)
            continue
        
        item['race'] = row['race']
        item['language'] = row['language']
        item['maritalStatus'] = row['marital_status']
        item['race'] = row['race']
        item['admitTime'] = str(row['admittime'])
        item['dischargeTime'] = str(row['dischtime'])
        item['admType'] = row['admission_type']
        item['admLocation'] = row['admission_location']
        item['insurance'] = row['insurance']

        hadmId = row['hadm_id']
        admitData[hadmId] = item

    #times = np.array(times)
    #print(times.min(), times.max(), times.mean(), times.std())

    pickle.dump(admitData, open('tmp.bin', 'wb'))
    #'''

    #TODO, get info on age, gender from patients.csv

    admitData = pickle.load(open('tmp.bin', 'rb'))
    print(len(admitData))

    data = {}
    csvPath = r"..\mimiciv\2.0\icu\chartevents.csv"
    with open(csvPath, 'r') as f:
        csvReader = csv.reader(f)

        rowHeaders = next(csvReader)
        print(rowHeaders)
        print('Number of columns:', len(rowHeaders))

        nItems = 100000000
        for cnt, row in enumerate(csvReader):
            #only use events that have an hadm_id that is in admitData
            hadmId = int(row[1])

            if hadmId not in admitData.keys():
                continue
            
            
            chartTime = row[3]

            #'''
            #check if chart time is valid, in between admit and discharge
            startStr = admitData[hadmId]['admitTime']
            stopStr = admitData[hadmId]['dischargeTime']
            if not is_valid_time(chartTime, startStr, stopStr):
                #print(hadmId, chartTime, startStr, stopStr)
                #print(row)
                continue
            #'''
                
            itemId = row[5]
            itemVal = row[6]
            valueUnit = row[8]

            event = {'itemId': itemId, 'itemVal': itemVal, 'valueUnit': valueUnit, 
                        'chartTime': chartTime}
            if hadmId not in data.keys():
                data[hadmId] = {}
                for key in admitData[hadmId].keys():
                    data[hadmId][key] = admitData[hadmId][key]

                data[hadmId]['events'] = [event] 
            else:
                data[hadmId]['events'].append(event)

            if cnt > nItems:
                break
            
    pickle.dump(data, open('data.bin', 'wb'), protocol=2)

    #TODO, get rid of admissions where they had no events


    return

if __name__ == '__main__':
    main()