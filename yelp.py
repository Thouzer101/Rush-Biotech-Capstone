import os
import json

import numpy as np


def main():

    dataPath = r"D:\Datasets\yelp_dataset\yelp_academic_dataset_review.json"

    with open(dataPath, 'r', encoding='utf-8') as dataFile:
        line = dataFile.readline()
        cnt = 1
        while line:
            line = dataFile.readline()
            cnt += 1
    print(cnt)


if __name__ == '__main__':
    main()