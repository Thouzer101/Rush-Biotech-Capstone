import os

def main():

    #unzip all the zip files in a given directory
    dirItems = os.listdir('.')

    for dirItem in dirItems:
        dirPath = os.path.join('.', dirItem)

        if os.path.isdir(dirPath):
            dataDirItems = os.listdir(dirPath)
            for dataDirItem in dataDirItems:
                if dataDirItem.endswith('gz'):
                    dirItemPath = os.path.join(dirPath, dataDirItem)
                    outPath = '\\'.join(dirItemPath.split('\\')[:-1])
                    command = '7z x %s -o%s'%(dirItemPath, outPath)
                    print('Extracting %s to %s'%(dirItemPath, outPath))
                    os.system(command)



if __name__ == '__main__':
    main()
