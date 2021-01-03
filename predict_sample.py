import init
import pandas as pd
import numpy as np
import sys
from train_model import *
from glob import glob
import datetime

csv_path = ""

def label_check(hash):
    with open(csv_path, 'r') as reader:
        for line in reader:
            fields = line.strip().split(',')
            if fields[0].startswith(hash):
                return fields[1]

if __name__ == '__main__':
    lgbm_model = lgb.Booster(model_file=sys.argv[1]) #model 경로
    files = glob(sys.argv[2] + "/*.vir") #preset 폴더 경로

    for i in range(len(files)):
        if i %1000==0 :
           print(datetime.datetime.now(), i)
           sha256 = files[i].split('.')[0].split('\\')[-1]

        mal_data = open(files[i], 'rb').read()
        predict = init.predict_sample(lgbm_model, mal_data, files[i])
        print(files[i], ":", predict)
        sha256 = files[i].split('/')[-1].split('.vir')[0]
        score = np.int32(predict > 0.70)
        data = sha256 + ", " + str(predict) + "\x0a"
        f = open("./"+sys.argv[1]+"_predict.csv", 'a+')
        f.write(data)
        f.close()