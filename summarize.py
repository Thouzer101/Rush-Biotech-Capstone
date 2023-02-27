import os
import csv
from collections import defaultdict
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

def sortDict(inputDict, sortFunc, reverse=True):

    outputDict = [item for item in inputDict.items()]
    outputDict.sort(key=sortFunc, reverse=reverse)
    return outputDict

def main():

    csvPath = r"..\mimiciv\2.0\hosp\emar.csv"
    #csvPath = r"..\mimiciv\2.0\icu\inputevents.csv"

    #'''
    start = time.time()

    #if memory bound
    #'''

    columnIdx = 0
    nRows = 0
    items = defaultdict(int)
    with open(csvPath, 'r', encoding='utf-8') as f:
        csvReader = csv.reader(f)

        rowHeaders = next(csvReader)
        print(rowHeaders)
        print('Number of columns:', len(rowHeaders))

        columnId = rowHeaders[columnIdx]
        print('Evaluating column:', columnId)

        for item in tqdm(csvReader):
            #print(item)
            #return

            items[item[columnIdx]] += 1
            nRows += 1
    print(item) 
    print("Total Rows:", nRows)

    print('Unique elements', len(items.keys()))
    items = sortDict(items, lambda item: item[1])
    print(items[:3])
    print(items[-3:])

    end = time.time()
    print('elapsed time:', end - start)
    
    return

    #store the elements
    with open('data.txt', 'w') as f:
        for item in items:
            f.write(str(item[0]) + '\t' + str(item[1]) + '\n')

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