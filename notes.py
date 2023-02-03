import os
import csv
import time
import pickle
import spacy

def main():

    #get positive and negative words
    filePath = 'positive-words.txt'
    positiveWords = set()
    with open(filePath, 'r') as f:
        cnt = 0
        line = f.readline()
        while line:
            if cnt < 34:
                line = f.readline()
                cnt += 1
                continue
            line = f.readline()
            word = line.strip()
            if len(word) > 0:
                positiveWords.add(word)
            cnt += 1

    filePath = r'negative-words.txt'
    negativeWords = set()
    with open(filePath, 'r') as f:
        cnt = 0
        line = f.readline()
        while line:
            if cnt < 34:
                line = f.readline()
                cnt += 1
                continue
            line = f.readline()
            word = line.strip()
            if len(word) > 0:
                negativeWords.add(word)
            cnt += 1

    #get note data
    '''
    filePath = r'..\mimiciv\2.0\mimic-iv-note-deidentified-free-text-clinical-notes-2.2\note\discharge.csv'
    patientData = {}
    with open(filePath, 'r', encoding='utf-8') as f:
        csvReader = csv.reader(f)
        header = next(csvReader)
        for row in csvReader:
            subjectId = row[1]
            text = row[-1]
            if subjectId in patientData.keys():
                patientData[subjectId]['text'] += '\n' #probably not necessary, but want to separate the notes
                patientData[subjectId]['text'] += text
                patientData[subjectId]['nNotes'] += 1
            else:
                patientData[subjectId] = {'text': text, 'nNotes':1,
                                          'positiveWords': 0, 'negativeWords': 0}

    pickle.dump(patientData, open('patientData.bin', 'wb'))

    return
    #'''


    '''
    patientData = pickle.load(open('patientData.bin', 'rb'))
         
    #for each patient, get age, gender, and race
    dataDir = '..\mimiciv\\2.0\hosp'
    filePath = os.path.join(dataDir, 'patients.csv')
    with open(filePath, 'r') as f:
        csvReader = csv.reader(f)
        header = next(csvReader)
        for row in csvReader:
            subjectId = row[0]
            if subjectId not in patientData.keys():
                continue
            sex = row[1]
            age = row[2]
            patientData[subjectId]['sex'] = sex
            patientData[subjectId]['age'] = age

    filePath = os.path.join(dataDir, 'admissions.csv')
    with open(filePath, 'r') as f:
        csvReader = csv.reader(f)
        header = next(csvReader)
        for row in csvReader:
            subjectId = row[0]
            if subjectId not in patientData.keys():
                continue
            race = row[11]
            patientData[subjectId]['race'] = race


    #count the number of positive and negative words in each patient note
    engTok = spacy.load('en_core_web_sm')
    for subjectId in patientData.keys():
        
        text = patientData[subjectId]['text']
        nNotes = patientData[subjectId]['nNotes']
        tokText = engTok.tokenizer(text)
        posWords = 0
        negWords = 0
        for word in tokText:
            word = word.text.lower()
            if word in positiveWords:
                posWords += 1
            if word in negativeWords:
                negWords += 1

        print(subjectId, posWords, posWords / nNotes, negWords, negWords / nNotes)
        patientData[subjectId]['positiveWords'] = posWords 
        patientData[subjectId]['avgPosWords'] = posWords / nNotes 

        patientData[subjectId]['negativeWords'] = negWords
        patientData[subjectId]['avgNegWords'] = negWords / nNotes

    pickle.dump(patientData, open('patientData.bin', 'wb'))

    return
    #'''
    patientData = pickle.load(open('patientData.bin', 'rb'))
        
    #write to csv
    with open('patient_notes.csv', 'w', newline='') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(['patient_id', 'sex', 'age', 'race', 'nNotes', 
                            'pos_words', 'neg_words',
                            'avgPosWords', 'avgNegWords'])

        
        for subjectId in patientData.keys():
            sex = patientData[subjectId]['sex']
            age = patientData[subjectId]['age']
            race = patientData[subjectId]['race']
            nNotes = patientData[subjectId]['nNotes']
            posWords = patientData[subjectId]['positiveWords']
            negWords = patientData[subjectId]['negativeWords']
            avgPosWords = patientData[subjectId]['avgPosWords']
            avgNegWords = patientData[subjectId]['avgNegWords']

            row = [subjectId, sex, age, race, nNotes, posWords, negWords, avgPosWords, avgNegWords]
            csvWriter.writerow(row)


if __name__ == '__main__':
    main()