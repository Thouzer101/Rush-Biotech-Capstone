import os
import csv
import sqlite3
import pickle

from tqdm import tqdm
import numpy as np

def is_float(inStr):
    if inStr.replace('.', '').isnumeric():
        return True
    else:
        return False

def main():

    
    items = {}
    #make a dict of all d_items.csv
    dataDir = r'E:\OneDrive - rush.edu\Research Capstone\mimiciv\2.0'
    csvPath = os.path.join(dataDir, 'icu', 'd_items.csv')    
    with open(csvPath, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        row = next(csvReader)
        print(row)

        for row in tqdm(csvReader):
            itemId = row[0]
            paramType = row[6]
            items[itemId] = {'vals':[], 'paramType': paramType}
    
    #go through each event for valid hamdIds
    dbPath = 'mimic_iv.db'
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM chartevents")
    colHeaders = [i[0] for i in cursor.description]
    print(colHeaders)

    #get the value
    for row in tqdm(cursor):
        itemId = row[6]
        itemVal = row[7]
        paramType = items[itemId]['paramType']

        if 'Numeric' in paramType and is_float(itemVal):
            items[itemId]['vals'].append(float(itemVal))

    for itemKey in tqdm(items.keys()):
        item = items[itemKey]
        vals = np.array(item['vals'])

        if len(vals) != 0:
            items[itemKey]['mean'] = vals.mean()
            items[itemKey]['std'] = vals.std()
        else:
            items[itemKey]['mean'] = 0
            items[itemKey]['std'] = 1

    pickle.dump(items, open('chartevents_items.bin', 'wb'))


    csvFile = open('categories.csv', 'w', newline='', encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    #get total number of martial status from admissions table
    dbPath = 'mimic_iv.db'
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()

    #get total number of gender from patients table
    #get min and max from age patients table
    cursor.execute('SELECT * FROM patients')
    colHeaders = [i[0] for i in cursor.description]
    print(colHeaders)

    ages = {}
    genders = {}
    for row in cursor:
        gender = row[1]
        age = row[2]

        if gender not in genders.keys():
            genders[gender] = 1
        else:
            genders[gender] += 1

        if age not in ages.keys():
            ages[age] = 1
        else:
            ages[age] += 1

    for age in ages.items():
        csvWriter.writerow(['age_' + age[0], age[1]])

    for gender in genders.items():
        csvWriter.writerow(gender)

    cursor.execute('SELECT * FROM admissions')
    colHeaders = [col[0] for col in cursor.description]

    print(colHeaders)
    allRaces = {}
    allMaritalStatuses = {}
    for row in cursor:
        #print(row)
        maritalStatus = row[10]
        race = row[11]

        if maritalStatus not in allMaritalStatuses.keys():
            allMaritalStatuses[maritalStatus] = 1
        else:
            allMaritalStatuses[maritalStatus] += 1

        if race not in allRaces.keys():
            allRaces[race] = 1
        else:
            allRaces[race] += 1

    for item in allRaces.items():
        csvWriter.writerow(item)

    for item in allMaritalStatuses.items():
        if len(item[0]) == 0:
            csvWriter.writerow(['MARITAL_STATUS_UNKNOWN', item[1]])
        else:
            csvWriter.writerow(item)

    items = pickle.load(open('chartevents_items.bin', 'rb'))

    for item in items.items():
        itemId = item[0]
        vals = item[1]['vals']
        mean = item[1]['mean']
        std = item[1]['std']
        if len(vals) != 0:
            vals = np.array(vals)
            csvWriter.writerow([itemId, mean, std])
        else:
            csvWriter.writerow([itemId, 0, 1])



    


if __name__ == '__main__':
    main()