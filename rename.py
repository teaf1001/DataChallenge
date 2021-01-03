import os
import sys

def rename():
    file_path= sys.argv[1] #폴더 경로
    file_names= os.listdir(file_path)
    # print(file_names)
    i=0
    j=0
    for name in file_names:
        #src = os.path.join(file_path, name)
        #print(file_names[i])
        if 'test' in file_names[i]:
            print(file_names[i])
            src = os.path.join(file_path, name)
            dst = "test_features_" + str(j) + ".jsonl"
            # dst = str(i)+".jsonl"
            dst = os.path.join(file_path, dst)
            os.rename(src, dst)
            j+=1
        i+=1

def combinebinary():
    file_path = sys.argv[1] #폴더 경로

if __name__=="__main__":
    rename()
    #combinebinary()