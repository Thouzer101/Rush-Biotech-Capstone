import os
import csv
import time
import pickle
import spacy

def main():

    tarPatient = '15190587'
    filePath = r'..\mimiciv\2.0\mimic-iv-note-deidentified-free-text-clinical-notes-2.2\note\discharge.csv'
    patientData = {}
    with open(filePath, 'r', encoding='utf-8') as f:
        csvReader = csv.reader(f)
        header = next(csvReader)
        for row in csvReader:
            subjectId = row[1]
            
            if subjectId == tarPatient:
                text = row[-1]

                with open('many_positive_example.txt', 'w') as f:
                    f.write(text)
                return



if __name__ == '__main__':
    main()