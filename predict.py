import numpy as np
import os
from tensorflow.keras.models import load_model

# 저장된 모델 로드
model_path = 'final_model.h5'
model = load_model(model_path)
print(f'Model loaded from {model_path}')

# 전처리된 데이터 파일 경로
preprocessed_data_file = 'data.npy'

# 전처리된 데이터를 불러오는 함수
def load_preprocessed_data():
    if os.path.exists(preprocessed_data_file):
        data = np.load(preprocessed_data_file)
        return data
    else:
        raise FileNotFoundError(f"{preprocessed_data_file} 파일이 존재하지 않습니다.")

# 예측 함수
def predict(audio_data):
    audio_data = np.expand_dims(audio_data, axis=(0, -1))
    prediction = model.predict(audio_data)
    label = 'Human' if prediction < 0.5 else 'AI Generated'
    return label, prediction[0][0]

# 디렉토리 내 모든 파일 예측 함수
def predict_all(directory):
    preprocessed_data = load_preprocessed_data()
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.wav'):  # .wav 파일만 예측
            filepath = os.path.join(directory, filename)
            label, confidence = predict(preprocessed_data[i])
            print(f'FileName: {filepath}')
            print(f'Prediction: {label}')
            print(f'Confidence: {confidence:.2f}')

# 예시 예측 실행
predict_all('C:/Users/Jeong Taehyeon/OneDrive/바탕 화면/archive/predict/')
