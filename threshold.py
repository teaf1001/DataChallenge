import os
from sys import argv
import numpy as np

# 임계치 설정해서 예측 값 정하기
if __name__=="__main__":
    answer= argv[1]
    file1=open(answer, 'r')

    predict = [0, 0]

    while (predict != []):
        file2 = open(answer.replace('model.txt_', ''), 'a+')
        predict = file1.readline().split()
        sha256= predict[0]
        score= predict[1]
        score2= np.int32(float(score) > 0.70)

        print(sha256, score2)

        new = sha256 + str(score2) + '\x0a'
        file2.write(new)
        file2.close()

    file1.close()
