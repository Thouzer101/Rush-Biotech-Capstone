import os
import csv
from collections import defaultdict
import time
import pandas as pd

def sortDict(inputDict, sortFunc, reverse=True):

    outputDict = [item for item in inputDict.items()]
    outputDict.sort(key=sortFunc, reverse=reverse)
    return outputDict


def main():

    csvPath = r"..\mimiciv\2.0\hosp\pharmacy.csv"
    columnIdx = 12
    #'''
    start = time.time()
    df = pd.read_csv(csvPath, engine='pyarrow', keep_default_na=False)
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
    
    print('Unique elements', len(items.keys()))
    
    items = sortDict(items, lambda item: item[1])
    print(items[:3])
    print(items[-3:])

    return
    #'''

    csvReader = csv.reader(open(csvPath, encoding='utf-8'))
    header = next(csvReader)
    print(header)
    print('Column:', header[columnIdx])
    start = time.time()

    earlyStop = False
    nRows = 0
    items = defaultdict(int)

    for idx, row in enumerate(csvReader):
        nRows += 1
        item = row[columnIdx]
        items[item] += 1
        print(item, float(item))
        
        if earlyStop:
            print(row)
        if earlyStop and idx > 2:
            return

    end = time.time()
    print('elapsed time:', end - start)

    print('nRows:', nRows + 1)
    print('unique elements', len(items))

    if len(items) < 100:
        print(items)

    '''
    for item in items:
        print(item)
    #'''
    #return
    items = sortDict(items, lambda item: item[1])
    print(items[:3])
    print(items[-3:])
    return

    #items = items.items()
    #'''
    for item in items:
        if item[1] != 1:
            print(item)

    return
    #'''

    #'''
    x = []
    y = []
    for item in items:
        x.append(int(item[0]))
        y.append(item[1])

    plt.bar(x, y)
    plt.suptitle('Subject Age Distribution')
    plt.xlabel('Id')
    plt.ylabel('n')
    plt.show()
    #'''

    

    #sortedItems = sortDict(items, lambda item:item[1])




if __name__ == '__main__':
    main()