import numpy as np
import os
import time
import csv
from datetime import datetime, timedelta
import ujson as json
from tqdm import tqdm
import sqlite3


def get_col_header(csvPath):
    with open(csvPath, 'r') as csvFile:
        csvReader = csv.reader(csvFile)

    return next(csvReader)

def build_table(conn, csvPath, tableName=None, primaryKey=None, fast=True, buffSize=1000000):
    #connection to database object
    cursor = conn.cursor()

    if fast:
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = normal;")

    if tableName is None:
        tableName = csvPath.split('\\')[-1].split('.')[0]
    
    with open(csvPath, 'r', encoding='utf-8') as csvFile:
        csvReader = csv.reader(csvFile)

        colHeader = next(csvReader)
        cmdStart = "CREATE TABLE IF NOT EXISTS %s ("%tableName
        cmdEnd = ");"

        columns = []

        if primaryKey is None:
            columns.append("entryId integer PRIMARY KEY AUTOINCREMENT")
        elif primaryKey not in colHeader:
            print('Pirmary key not found')
            return
        else:
            columns.append("%s text PRIMARY KEY"%str(primaryKey))
        
        for col in colHeader:
            if col == primaryKey:
                continue
            columns.append("%s text"%col)

        colStr = ', '.join(columns)
        cmd = cmdStart + colStr + cmdEnd
        #print(cmd)
        
        cursor.execute(cmd)

        for cnt, row in tqdm(enumerate(csvReader)):
            cmd = "INSERT INTO %s("%tableName + ','.join(colHeader) + ") VALUES("
            nValues = ','.join(['?' for i in range(len(colHeader))])
            cmd = cmd + nValues + ")"
            cursor.execute(cmd, tuple(row))

            if cnt % buffSize == 0:
                conn.commit()
        
        conn.commit()

def main():

    dataDir = r'E:\OneDrive - rush.edu\Research Capstone\mimiciv\2.0'
    dbPath = 'mimic_iv.db'
    '''
    if os.path.isfile(dbPath):
        os.remove(dbPath)
    #'''

    #create database
    conn = sqlite3.connect(dbPath)

    csvPath = os.path.join(dataDir, 'hosp', 'poe.csv')
    build_table(conn, csvPath)

    csvPath = os.path.join(dataDir, 'hosp', 'prescriptions.csv')
    build_table(conn, csvPath)

    csvPath = os.path.join(dataDir, 'hosp', 'emar.csv')
    build_table(conn, csvPath)

    #make table for microbiologyevents
    csvPath = os.path.join(dataDir, 'hosp', 'microbiologyevents.csv')
    build_table(conn, csvPath)

    #make table for omr
    csvPath = os.path.join(dataDir, 'hosp', 'omr.csv')
    build_table(conn, csvPath)

    #make table for prcedures_icd
    csvPath = os.path.join(dataDir, 'hosp', 'procedures_icd.csv')
    build_table(conn, csvPath)

    #make table for diagnoses_icd
    csvPath = os.path.join(dataDir, 'hosp', 'diagnoses_icd.csv')
    build_table(conn, csvPath)

    csvPath = os.path.join(dataDir, 'hosp', 'patients.csv')
    build_table(conn, csvPath, primaryKey='subject_id')

    return
    #convert csv to table in database
    
    csvPath = os.path.join(dataDir, 'hosp', 'admissions.csv')
    build_table(conn, csvPath, primaryKey='hadm_id')

    csvPath = os.path.join(dataDir, 'icu', 'chartevents.csv')
    build_table(conn, csvPath)
    cursor.execute("CREATE INDEX hadmIdx ON chartevents(hadm_id)")

    csvPath = os.path.join(dataDir, 'hosp', 'patients.csv')
    build_table(conn, csvPath, primaryKey='subject_id')

    conn.close()

if __name__ == '__main__':
    main()