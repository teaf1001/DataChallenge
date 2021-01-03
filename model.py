import init
from train_model import *
import sys

if __name__ == '__main__':
    data_path = sys.argv[1] #폴더 이름을 인자로 받음(벡터화된 피처들 이름형식:train_features_n.jsonl/test_features_n.jsonl)
    init.create_vectorized_features(data_path) #피쳐 벡터화
    metadata_dataframe = init.create_metadata(data_path) #벡터화된 피쳐를 모델에 학습시키기 위한 형태로 변환 -> 엠버모델의 경우 사용
    X_train, Y_train, X_test, Y_test = init.read_vectorized_features(data_path) #벡터화된 피쳐 저장
    print("[+] Training Lightgbm model..")
    lgbm_model = init.train_model(data_path) # LightGBM 모델 생성
    lgbm_model.save_model(os.path.join(sys.argv[1] + "/model.txt")) # 모델 저장

    # lgbm_model = lgb.Booster(model_file="./model.txt")
    # path = r"C:\Users\schcsrc\Desktop\coding\sample\KakaoTalk_Setup.exe"
    # mal_data = open(path, 'rb').read()
    # print(init.predict_sample(lgbm_model, mal_data, path))

    #ember.create_vectorized_features(data_path)
    #metadata_dataframe = ember.create_metadata(data_path)
    #X_train, Y_train, X_test, Y_test = ember.read_vectorized_features(data_path)
    #lgbm_model = ember.train_model(data_path)
