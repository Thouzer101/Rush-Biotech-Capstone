

def main():

    #make sure you know order of column names
    #'''
    dbPath = 'mimic_iv.db'
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()

    '''
    cursor.execute("PRAGMA table_info(admissions)")
    colNames = cursor.fetchall()

    colIdx = get_col_idx(colNames, 'hadm_id') 
    print('hadm_id is col', colIdx)

    colIdx = get_col_idx(colNames, 'admittime')
    print('admittime is col', colIdx)

    colIdx = get_col_idx(colNames, 'dischtime')
    print('dischtime is col', colIdx)
    return
    #'''

    #'''
    cursor.execute("SELECT * FROM admissions")
    #get relevant hadmId
    validHadmId = []
    total = 0
    for row in tqdm(cursor):
        hadmId = row[0]
        admTime = row[2]
        disTime = row[3]
        stayTime = time_diff(admTime, disTime)
        total += 1
        #ignore entries that have negative stay time or longer than a month
        if stayTime <= 0 or stayTime > 86400 * 30:
            continue
    
        validHadmId.append(hadmId)

    print('%3.0f / %3.0f = %3.3f'%(len(validHadmId), total,len(validHadmId)/total))
    pickle.dump(validHadmId, open('hadmId.bin', 'wb'))
    return
    #'''

    '''
    validHadmId = pickle.load(open('hadmId.bin', 'rb'))

    #check to see if a hadmID has events, many do not
    #if so, its valid
    validHamdId = []
    for hadmId in tqdm(validHadmId):
        #cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM chartevents WHERE hadm_id = ?", (hadmId,))
        cursor.execute("SELECT * FROM chartevents WHERE hadm_id = ?", (hadmId,))

        rows = cursor.fetchall()
        if len(rows) != 0:
            #print(len(rows), type(rows))
            validHamdId.append(hadmId)

    with open('valid_hadmId.txt', 'w') as txtFile:
        for hamdId in validHamdId:
            txtFile.write(event + '\n')
    #'''
    #'''
    hadmIds = []
    with open('valid_hadmId.txt', 'r') as txtFile:
        line = txtFile.readline().strip()
        while line:
            hadmIds.append(line)
            line = txtFile.readline().strip()   

    np.random.shuffle(hadmIds)
    print(hadmIds[0])
    with open('valid_hadmId.txt', 'w') as txtFile:
        for hadmId in hadmIds:
            txtFile.write(hadmId + '\n')

    conn.close()

    #''' 

if __name__ == '__main__':
    main()