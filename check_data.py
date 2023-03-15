import os
import csv
import pickle

from collections import defaultdict
import numpy as np
from tqdm import tqdm

from dataset import MimicDataset, isFloat, isListNumeric, epsilon

def main():

    dataDir = r'E:\OneDrive - rush.edu\Research Capstone\mimiciv\2.0'
    tokLabels = {}
    with open(os.path.join(dataDir, 'hosp', 'd_labitems.csv'), 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        row = next(csvReader)
        for row in csvReader:
            token = row[0]
            label = row[1]
            tokLabels[token] = label

    with open(os.path.join(dataDir, 'icu', 'd_items.csv'), 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        row = next(csvReader)
        for row in csvReader:
            token = row[0]
            label = row[1]
            tokLabels[token] = label

    #trainset = MimicDataset('subtrain.txt')
    validset = MimicDataset('subvalid.txt')

    data = defaultdict(dict)
    death = 0
    noDeath = 0
    for item in tqdm(validset):

        died = int(item['died'].item())
        if died == 1:
            death += 1
        else:
            noDeath += 1

        tokTen = item['tokTen']
        tmpSet = set()
        for tok in tokTen:
            token = validset.idxToTok[int(tok)]

            tmpSet.add(token)

        for tok in tmpSet:
            if tok not in data[died].keys():
                data[died][tok] = 1
            else:
                data[died][tok] += 1

    #get all the tokens
    tokens = {}
    for key in data[1].keys():
        ratio = data[1][key] / float(death)
        tokens[key] = {'death': ratio}

    for key in data[0].keys():
        ratio = data[0][key] / float(noDeath)
        if key not in tokens.keys():
            tokens[key] = {'noDeath': ratio}
        else:
            tokens[key]['noDeath'] = ratio

    csvFile = open('token_analysis.csv', 'w', newline='', encoding='utf-8')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['label', 'token', 'ratio in died', 'ratio in not died', 'difference'])
    for key in tokens.keys():
        item = tokens[key]

        if 'death' not in item.keys():
            deathRatio = 0
        else:
            deathRatio = item['death']
        
        if 'noDeath' not in item.keys():
            noDeathRatio = 0
        else:
            noDeathRatio = item['noDeath']
        
        dif = np.abs(deathRatio - noDeathRatio)

        if key in tokLabels.keys():
            label = tokLabels[key]
        else:
            label = ''

        csvWriter.writerow([label, key, deathRatio, noDeathRatio, dif])

    return

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



if __name__ == '__main__':
    main()