import os
import time
import pandas as pd
import pickle

def main():

    #reorganize MIMIC-IV to be based on hospital admissions
    dataDir = r'E:\OneDrive - rush.edu\Research Capstone\mimiciv\2.0'    

    #first get all admission IDs
    csvPath = os.path.join(dataDir, 'hosp', 'admissions.csv')
    df = pd.read_csv(csvPath, engine='pyarrow', keep_default_na=False)
    print(df.columns)

    '''
    data = {}
    for idx, row in df.iterrows():
        item = {}
        
        item['race'] = row['race']
        item['language'] = row['language']
        item['maritalStatus'] = row['marital_status']
        item['race'] = row['race']
        item['admitTime'] = row['admittime']
        item['dischargeTime'] = row['dischtime']
        item['admType'] = row['admission_type']
        item['admLocation'] = row['admission_location']
        item['insurance'] = row['insurance']

        hadmId = row['hadm_id']
        data[hadmId] = item

    pickle.dump(data, open('tmp.bin', 'wb'))
    #'''

    data = pickle.load(open('tmp.bin', 'rb'))
    print(len(data))

    return

if __name__ == '__main__':
    main()