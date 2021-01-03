import features_reboot as features
import init
import numpy as np
import json
import jsonpickle
import sys
import datetime
from glob import glob
import csv
import sys
import os

#folders = [r"E:\KISA-CISC2017-Malware-1st"]
#csv_path = r"E:\label\2017_malware_1st_answer.csv"

folders = sys.argv[1] # 추출 대상 파일 경로
csv_path = sys.argv[2] # 정답지 경로
savename = csv_path.split('/')[2].split('_')[0]
# 피쳐 저장 이름

# 라벨 매핑
def label_check(hash):
    with open(csv_path, 'r') as reader:
        for line in reader:
            fields = line.strip().split(',')
            if fields[0].startswith(hash):
                return fields[1]

# 피쳐 추출
def extract(filename):
    mal_data = open(filename, 'rb').read() # 실행 파일 열기
    extractor = init.features.PEFeatureExtractor(2) #인자 2 고정
    filehash=filename.split("/")[-1].split('.vir')[0]
    try:
        features = extractor.raw_features(mal_data, filename) # 피쳐 추출
        features['label'] = label_check(filehash) # 라벨 매핑
        features = jsonpickle.encode(features) # 피쳐 업데이트(라벨)

    except Exception as e:
        print(filename + " error: " + str(e))
        return False
    return features


if __name__ == '__main__':
    files = glob(folders + '/*.vir')
    num = -1
    time = str(datetime.datetime.now())
    path=time.split()[0].replace('-','_')+'_'+time.split()[1].replace(':','')[:6]
    savepath = os.makedirs(r'./features/'+path)
    for i in range(len(files)):
        if i % 100 == 0:
            num += 1
            print(datetime.datetime.now(), savename +'_train_features_' + str(num) + '.jsonl')

        f = open('./features/'+ path+'/' + savename + '_train_features_'+str(num)+'.jsonl', 'a+', encoding='utf8')

        data = extract(files[i])
        if data==False:
            print(files[i],': False',)
            continue
        else:
            data = data+'\n'
            f.write(data)
        #f.close()
#dddd
'''
# 파일 하나만 뽑기
    f = open("test_new", 'a+', encoding='utf8')
    data=extract(r"5b712f3ced695dd1510320494ecac67b277c0b386ee465303504c89431f87c78.vir")
    f.write(data)
    f.close()
    print(data)
'''
