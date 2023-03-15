import csv

from dataset import MimicDataset

def main():

    csvFile = open('train_tabular_data.csv', 'w', newline='', encoding='utf-8')
    trainset = MimicDataset('subtrain.txt', timeLimit=24)


    csvFile = open('valid_tabular_data.csv', 'w', newline='', encoding='utf-8')
    validset = MimicDataset('subvalid.txt', timeLimit=24)

if __name__ == '__main__':
    main()