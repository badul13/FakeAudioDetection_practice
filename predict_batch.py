import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow version:", tf.__version__)
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("Is GPU available: ", tf.test.is_gpu_available())
    except RuntimeError as e:
        print(e)

# 저장된 모델 로드
model_path = 'final_model.keras'
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

# 개별 예측 함수
def predict(audio_data):
    audio_data = np.expand_dims(audio_data, axis=0)  # 배치 차원 추가
    audio_data = np.expand_dims(audio_data, axis=-1)  # 채널 차원 추가
    prediction = model.predict(audio_data)
    return prediction  # 예측값 반환

# 모든 파일 예측 함수
def predict_all(directory):
    preprocessed_data = load_preprocessed_data()
    results = []
    # num_files = 50000
    filenames = [filename for filename in os.listdir(directory) if filename.endswith('.ogg')]
    num_files = len(filenames)

    for i in range(num_files):
        filename = f'TEST_{i:05d}.ogg'
        audio_data = preprocessed_data[i]
        prediction = predict(audio_data)

        fake_prob = float(prediction[0][0])  # ai_output
        real_prob = float(prediction[1][0])  # human_output

        results.append({
            'id': filename.split('.')[0],
            'fake': fake_prob,
            'real': real_prob
        })

        # 진행도 출력 (매 100개 파일마다)
        if (i + 1) % 100 == 0:
            print(f'Progress: {i + 1}/{num_files} ({((i + 1) / num_files) * 100:.2f}%)')

    # 예측 결과를 sample_submission.csv 파일에 저장
    df = pd.DataFrame(results)
    df.to_csv('../fakeAudio/sample_submission.csv', index=False, float_format='%.8f')
    print(f'Predictions saved to ../fakeAudio/sample_submission.csv')

# 예측 실행
predict_all('../fakeDetection/data/test')