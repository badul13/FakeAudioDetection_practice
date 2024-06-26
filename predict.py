import numpy as np
import os
from tensorflow.keras.models import load_model
from preprocessing import load_audio_data, pad_or_trim

# 저장된 모델 로드
model_path = 'final_model.h5'
model = load_model(model_path)
print(f'Model loaded from {model_path}')


# 예측 함수
def predict(audio_path):
    audio_data, sr = load_audio_data(audio_path)
    audio_data = pad_or_trim(audio_data, 16000)
    audio_data = np.expand_dims(audio_data, axis=(0, -1))

    prediction = model.predict(audio_data)
    label = 'Human' if prediction < 0.5 else 'AI Generated'
    print(f'FileName: {audio_path}')
    print(f'Prediction: {label}')
    print(f'Confidence: {prediction[0][0]:.2f}')

# 디렉토리 내 모든 파일 예측 함수
def predict_all(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):  # .wav 파일만 예측
            filepath = os.path.join(directory, filename)
            predict(filepath)

# 예시 예측 실행
predict_all('C:/Users/Jeong Taehyeon/OneDrive/바탕 화면/archive/predict/')
