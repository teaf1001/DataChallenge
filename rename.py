import os
import sys

def rename():
    folder_path= sys.argv[1] #폴더 경로
    file_names= os.listdir(folder_path)

    train=round(len(file_names) / 10 * 7)
    test= len(file_names) - train

    for name in range(0,train):
        src = os.path.join(folder_path, file_names[name])
        dst = os.path.join(folder_path, "train_features_{}.jsonl".format(name))
        os.rename(src, dst)

    for name in range(train, len(file_names)):
        src = os.path.join(folder_path, file_names[name])
        dst = os.path.join(folder_path, "test_features_{}.jsonl".format(name))
        os.rename(src, dst)

if __name__=="__main__":
    rename()