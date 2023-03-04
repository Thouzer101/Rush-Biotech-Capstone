import os
import json
import csv
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spacy.lang.en import English

class YelpDataset(nn.Module):
    def __init__(self, split, maxLen=256,
                 reviewsFile='yelp_reviews.bin', vocabFile='vocab.txt'):
        super().__init__()

        self.maxLen = maxLen

        tokToIdx = {}
        tokToIdx['<pad>'] = 0
        tokToIdx['<start>'] = 1
        tokToIdx['<end>'] = 2
        tokToIdx['<cls>'] = 3
        tokToIdx['<unk>'] = 4
        
        with open(vocabFile, 'r', encoding='utf-8') as txtFile:
            line = txtFile.readline()
            cnt = 5

            while line:
                line = line.strip()
                tokToIdx[line] = cnt
                line = txtFile.readline()
                cnt += 1
        self.tokToIdx = tokToIdx

        idxToTok = {}
        for item in tokToIdx.items():
            idxToTok[item[1]] = item[0]
        self.idxToTok = idxToTok

        data = pickle.load(open(reviewsFile, 'rb'))

        dataLen = len(data)
        splitLen = int(dataLen * 0.8)
        if split == 'train':
            self.data = data[:splitLen]
        elif split == 'valid':
            self.data = data[splitLen:]
        else:
            print('Invalid split:', split)

        nlp = English()
        tokenizer = nlp.tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        rating = self.data[idx]['rating']

        tokens = list(self.tokenizer(text))

        tok = [self.tokToIdx['<start>']]
        for token in tokens:

            token = str(token).strip()
            if len(token) == 0:
                continue

            if token in self.tokToIdx.keys():
                tokIdx = self.tokToIdx[token]
                tok.append(tokIdx)
            else:
                tokIdx = self.tokToIdx['<unk>']
                tok.append(tokIdx)

        maxLen = min(self.maxLen - 1, len(tok))
        tok = tok[:maxLen]

        tok.append(self.tokToIdx['<end>'])

        tokTen = torch.tensor(tok, dtype=torch.long)
        ratingTen = torch.tensor(int(float(rating)), dtype=torch.long)

        return {'tokTen': tokTen, 'ratings': ratingTen}

    def tokIdToSentence(self, tokIds):
        return ' '.join([self.idxToTok[int(token)] for token in tokIds])

def collate_fn(batchItem):
    #batchItem is of type list
    batchSize = len(batchItem)
    maxLen = 0
    for item in batchItem:
        tokTen = item['tokTen']
        maxLen = max(maxLen, tokTen.shape[0])

    batchTen = torch.zeros(batchSize, maxLen).long()
    batchRatings = torch.zeros(batchSize).long()

    for i, item in enumerate(batchItem):
        tokTen = item['tokTen']
        rating = item['ratings']
        batchTen[i,:tokTen.shape[0]] = tokTen
        batchRatings[i] = rating

    return {'tokTen': batchTen, 'ratings': batchRatings}    

def main():


    #'''
    trainset = YelpDataset('train')
    validset = YelpDataset('valid')

    print(len(trainset), len(validset))

    batchSize = 32
    nWorkers = 4
    trainloader = DataLoader(trainset, batch_size=batchSize, num_workers=nWorkers, shuffle=True, collate_fn=collate_fn)

    #TODO, check that nWorkers 0 and 4 are the same data

    testIter = trainloader
    for item in testIter:
        tokTen = item['tokTen']
        print(tokTen.shape)

    return
    #'''

    dataPath = r"D:\Datasets\yelp_dataset\yelp_academic_dataset_review.json"

    data = []
    with open(dataPath, 'r', encoding='utf-8') as dataFile:
        line = dataFile.readline()
        while line:
            jsonData = json.loads(line)

            rating = jsonData['stars']
            text = jsonData['text']
            data.append({'rating':rating, 'text':text})

            line = dataFile.readline()
    print(len(data))

    pickle.dump(data, open('yelp_reviews.bin', 'wb'))
    return
    
    data = pickle.load(open('yelp_reviews.bin', 'rb'))

    nlp = English()
    tokenizer = nlp.tokenizer

    vocab = {}
    for item in tqdm(data):
        rating = item['rating']
        text = item['text']
        tokens = list(tokenizer(text))

        for token in tokens:
            token = str(token)
            if token not in vocab.keys():
                vocab[token] = 1
            else:
                vocab[token] += 1

    vocab = [item for item in vocab.items()]
    vocab.sort(key=lambda x:x[1])

    pickle.dump(vocab, open('vocab.bin', 'wb'))

    return

    vocab = pickle.load(open('vocab.bin', 'rb'))

    txtFile = open('vocab.txt', 'w', encoding='utf-8')
    idx = 0
    for item in vocab[::-1]:
        word = item[0].strip()

        if len(word) != 0:
            txtFile.write(word + '\n')
            idx += 1

        if idx > 50000 - 1:
            break

    return

if __name__ == '__main__':
    main()