import os
import csv
import pickle

import numpy as np
from tqdm import tqdm

from dataset import MimicDataset, isFloat, isListNumeric, epsilon

def main():

    #go through all the data in train and validation
    trainHadmIds = []
    with open('subtrain.txt', 'r') as txtFile:
        line = txtFile.readline()
        while line:
            trainHadmIds.append(line.strip())
            line = txtFile.readline()

    validHadmIds = []
    with open('subvalid.txt', 'r') as txtFile:
        line = txtFile.readline()
        while line:
            validHadmIds.append(line.strip())
            line = txtFile.readline()

    #make sure none of the hadm_ids in validation are not in train
    '''
    for item in validHadmIds:
        if item in trainHadmIds:
            print(item)
    #PASSED
    #'''

    #make sure data in both match the original csv file
    '''
    data = {}
    dataDir = r'E:\OneDrive - rush.edu\Research Capstone\mimiciv\2.0'
    filePath = os.path.join(dataDir, 'hosp', 'admissions.csv')
    with open(filePath, 'r', encoding='utf-8', newline='')  as csvFile:
        csvReader = csv.reader(csvFile)
        row = next(csvReader)
        print(row)
        for row in tqdm(csvReader):
            hadmId = row[1]
            subjectId = row[0]
            hospAdmTime = row[2]
            edRegTime = row[12]
            deathTime = row[4]
            maritalStatus = row[10]
            race = row[11]

            entry = {'subjectId':subjectId, 'deathTime':deathTime, 'maritalStatus':maritalStatus, 'race':race,
                     'hospAdmTime': hospAdmTime, 'edRegTime':edRegTime}
            if hadmId in trainHadmIds:
                data[hadmId] = entry
                data[hadmId]['split'] = 'train'

            if hadmId in validHadmIds:
                data[hadmId] = entry
                data[hadmId]['split'] = 'validation'

    print(len(trainHadmIds), len(validHadmIds), len(data.keys())) 
    pickle.dump(data, open('patient_data.bin', 'wb'))
    #'''
    '''
    data = pickle.load(open('patient_data.bin', 'rb'))
    dataDir = r'E:\OneDrive - rush.edu\Research Capstone\mimiciv\2.0'
    filePath = os.path.join(dataDir, 'hosp', 'patients.csv')
    with open(filePath, 'r', encoding='utf-8', newline='')  as csvFile:
        csvReader = csv.reader(csvFile)
        row = next(csvReader)
        print(row)
        for row in tqdm(csvReader):
            subjectId = row[0]
            gender = row[1]
            age = row[2]
            dod = row[5]
            for item in data.items():
                if subjectId == data[item[0]]['subjectId']:
                    data[item[0]]['gender'] = gender
                    data[item[0]]['age'] = age
    pickle.dump(data, open('patient_data.bin', 'wb'))
    #'''
    #create a hadm_id, subject_id with demographics csv file

    '''
    data = pickle.load(open('patient_data.bin', 'rb'))
    with open('patient_data.csv', 'w', newline='', encoding='utf-8') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['subject_id', 'hadm_id', 'split', 'age', 'marital_status', 'race', 'gender', 
                            'admit_time', 'ED_reg_time', 'time_of_death'])
        for item in data.items():
            hadmId = item[0]
            subjectId = item[1]['subjectId'] 
            split = item[1]['split']
            age = item[1]['age']
            maritalStatus = item[1]['maritalStatus']
            race = item[1]['race']
            gender = item[1]['gender']
            hospAdmTime = item[1]['hospAdmTime']
            edRegTime = item[1]['edRegTime']
            deathTime = item[1]['deathTime']
            csvWriter.writerow([subjectId, hadmId, split, age, maritalStatus, race, gender, hospAdmTime, edRegTime, deathTime])
    #'''
    
    '''
    data = pickle.load(open('patient_data.bin', 'rb'))
    #create an events csv file
    csvFile = open('events_data.csv', 'w', newline='', encoding='utf-8')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['subject_id', 'hadm_id', 'event_id', 'event_val', 'chart_time'])

    dataDir = r'E:\OneDrive - rush.edu\Research Capstone\mimiciv\2.0'
    filePath = os.path.join(dataDir, 'hosp', 'labevents.csv')
    with open(filePath, 'r', encoding='utf-8', newline='')  as csvFile:
        csvReader = csv.reader(csvFile)
        row = next(csvReader)
        print(row)
        for row in tqdm(csvReader):
            subjectId = row[1]
            hadmId = row[2]
            eventId = row[4]
            eventVal  = row[7]
            chartTime = row[5]
            if hadmId in data.keys():
                csvWriter.writerow([subjectId, hadmId, eventId, eventVal, chartTime])


    dataDir = r'E:\OneDrive - rush.edu\Research Capstone\mimiciv\2.0'
    filePath = os.path.join(dataDir, 'icu', 'chartevents.csv')
    with open(filePath, 'r', encoding='utf-8', newline='')  as csvFile:
        csvReader = csv.reader(csvFile)
        row = next(csvReader)
        print(row)
        for row in tqdm(csvReader):
            subjectId = row[0]
            hadmId = row[1]
            chartTime = row[3]
            eventId = row[5]
            eventVal = row[6]
            if hadmId in data.keys():
                csvWriter.writerow([subjectId, hadmId, eventId, eventVal, chartTime])
    #''' 

    #make sure the original data can be found in the validset and trainset 
    trainset = MimicDataset('subtrain.txt')
    validset = MimicDataset('subvalid.txt')

    chartEventStats = pickle.load(open('chartEventStats.bin', 'rb'))
    labEventStats = pickle.load(open('labEventStats.bin', 'rb'))\
    #combine event stats
    eventStats = {}
    for item in chartEventStats.items():
        eventStats[item[0]] = item[1]

    for item in labEventStats.items():
        eventStats[item[0]] = item[1]

    data = pickle.load(open('patient_data.bin', 'rb'))
    with open('events_data.csv', 'r', encoding='utf-8') as csvFile:
        csvReader = csv.reader(csvFile)
        row = next(csvReader)
        for row in tqdm(csvReader):
            #check to see if that event is there and if it has a corresponding value
            hadmId = row[1]
            eventId = row[2]
            eventVal = row[3]
            
            #convert eventVal to normalized value
            if eventId in chartEventStats.keys():
                paramType = chartEventTypes[eventId]
                if 'Numeric' in paramType:
                    eventVal = float(eventVal)
                    mean = eventStats[eventId]['mean']
                    std = eventStats[eventId]['std']
                    eventVal = (eventVal - mean) / std
                    eventVal = np.clip(eventVal, a_min=-3, a_max=3)
                else:
                    factors = eventStats[eventId]['factors']
                    factorsLen = len(factors)
                    if factorsLen == 0 or len(eventVal) == 0:
                        eventVal = -1.0
                    else:
                        idx = factors[eventVal]
                        eventVal = float(idx) / (factorsLen - 1 + epsilon)
            elif eventId in labEventStats.keys():
                if 'mean' in eventStats[eventId].keys():
                    if isFloat(eventVal):
                        eventVal = float(eventVal)
                        mean = eventStats[eventId]['mean']
                        std = eventStats[eventId]['std']
                        eventVal = (eventVal - mean) / std
                        eventVal = np.clip(eventVal, a_min=-3, a_max=3)
                    else:
                        eventVal = 0.0
                else:
                    factors = eventStats[eventId]['factors']
                    factorsLen = len(factors)
                    if factorsLen == 0 or len(eventVal) == 0:
                        eventVal = -1.0
                    else:
                        idx = factors[eventVal]
                        eventVal = float(idx) / (factorsLen - 1 + epsilon)
            else:
                print('ERROR:', eventId)
                return

            split = data[hadmId]['split']
            if split == 'train':
                found = checkEvent(hadmId, eventId, eventVal, trainset)
            else:
                found = checkEvent(hadmId, eventId, eventVal, validset)

            if not found:
                #if not, there is something wrong
                print(row)
                return


def checkEvent(hadmId, eventId, eventVal, dataset):

    for item in dataset:
        curHadmId = item['hadmId']
        if curHadmId != hadmId:
            continue

        allEvents = item['allEvents']
        for event in allEvents:
            curEventId = event['eventId']
            curEventVal = event['eventVal']

            if curEventId != eventId:
                continue

            dif = np.abs(float(eventVal) - curEventVal)
            if dif < epsilon:
                return True

    return False

if __name__ == '__main__':
    main()