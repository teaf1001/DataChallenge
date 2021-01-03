import sys
import datetime
import os

predict_csv = sys.argv[1]
answer_csv = sys.argv[2]

def label_check(hash):
    with open(answer_csv, 'r') as reader:
        for line in reader:
            fields = line.strip().split(',')
            if fields[0].startswith(hash):
                return fields[1]

if __name__ == '__main__':
    if os.path.isdir('./result/'+ sys.argv[2].split('/')[-1].split('.')[0]) == False:
        os.makedirs('./result/'+ sys.argv[2].split('/')[-1].split('.')[0])

    for i in range(30, 100):
        print(datetime.datetime.now(), 'Critical_Value: ', i/100)
        critical_value = i / 100 # 임계치
        # critical_value = 0.01
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        predict_value = 0
        f = open(predict_csv, 'rb')
        data = f.read()
        f.close()
        data = data.decode().split('\n')[:-1]
        data_dict = {}

        for i in range(len(data)):
            Hash = data[i].split(', ')[0]
            Score = data[i].split(', ')[1]
            data_dict[Hash] = Score

        for j in data_dict.keys():
            if float(data_dict[j]) > critical_value:
                predict_value = 1
            else:
                predict_value = 0

            if int(predict_value) == int(label_check(j)) and predict_value == 1:
                TP += 1
            elif int(predict_value) == int(label_check(j)) and predict_value == 0:
                TN += 1
            elif int(predict_value) != int(label_check(j)) and predict_value == 1:
                FP += 1
            elif int(predict_value) != int(label_check(j)) and predict_value == 0:
                FN += 1

        print('TP, TN, FP, FN : ', TP, TN, FP, FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 / ((1 / precision) + (1 / recall))
        print("accuracy, precision, recall, f1_score", accuracy, precision, recall, f1_score)

        f2 = open('./result/'+ sys.argv[2].split('/')[-1].split('.')[0]+'/'+sys.argv[2].split('/')[-1].split('.')[0]+'.csv', 'a+')
        #f2 = open(sys.argv[1].split('.txt')[0]+'.csv', 'a+')
        # f2 = open('result_2020data.csv', 'a+')
        f2.write('%f, %d, %d, %d, %d, %f, %f, %f, %f\n' % (critical_value, TP, TN, FP, FN, accuracy, precision, recall, f1_score))
        f2.close()
        print('%f success!' % critical_value)
