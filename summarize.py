import os
import csv
from collections import defaultdict
import time
import pandas as pd
import numpy as np

def sortDict(inputDict, sortFunc, reverse=True):

    outputDict = [item for item in inputDict.items()]
    outputDict.sort(key=sortFunc, reverse=reverse)
    return outputDict

def main():

    csvPath = r"..\mimiciv\2.0\hosp\emar_detail.csv"
    csvPath = r"..\mimiciv\2.0\icu\chartevents.csv"

    #'''
    start = time.time()

    #if memory bound
    #'''

    columnIdx = 0
    nRows = 0
    items = defaultdict(int)
    with open(csvPath, 'r') as f:
        csvReader = csv.reader(f)

        rowHeaders = next(csvReader)
        print(rowHeaders)
        print('Number of columns:', len(rowHeaders))

        columnId = rowHeaders[columnIdx]
        print('Evaluating column:', columnId)

        for item in csvReader:
            items[item[columnIdx]] += 1
            nRows += 1
    
    print("Total Rows:", nRows)

    print('Unique elements', len(items.keys()))
    items = sortDict(items, lambda item: item[1])
    print(items[:3])
    print(items[-3:])

    end = time.time()
    print('elapsed time:', end - start)

    return
    #'''        

    '''
    #PyArrow is several times faster
    #BUT you have to read it all at once 
    df = pd.read_csv(csvPath, engine='pyarrow', keep_default_na=False)

    columnIdx = 0
    end = time.time()
    print('elapsed time:', end - start)
    print(df.columns)
    print('nRows:', len(df))
    print('nColumns:', len(df.columns))
    columnId = df.columns[columnIdx]
    print('Evaluating column:', columnId)

    items = defaultdict(int)
    for item in df[columnId]:
        items[item] += 1
        cnt += 1

    randRow = np.random.randint(len(df))
    print('row example:', randRow)
    print(df.iloc[randRow,:], '\n')

    print('Unique elements', len(items.keys()))
    items = sortDict(items, lambda item: item[1])
    print(items[:3])
    print(items[-3:])

    return
    #'''



if __name__ == '__main__':
    main()