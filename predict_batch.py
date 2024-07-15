import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from tqdm import tqdm

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

# 전처리된 테스트 데이터 파일 경로
test_data_file = 'test_data.npy'

# 전처리된 데이터를 불러오는 함수
def load_preprocessed_test_data():
    if os.path.exists(test_data_file):
        data = np.load(test_data_file)
        return data
    else:
        raise FileNotFoundError(f"{test_data_file} 파일이 존재하지 않습니다.")

# 파일 이름에서 숫자를 추출하는 함수
def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])

# 단일 파일 예측 함수
def predict_single(audio_data):
    prediction = model.predict(np.expand_dims(np.expand_dims(audio_data, axis=-1), axis=0))
    ai_prob, human_prob = prediction[0][0], prediction[1][0]
    return ai_prob, human_prob

# 디렉토리 내 모든 파일 예측 함수
def predict_all(directory):
    preprocessed_data = load_preprocessed_test_data()
    results = []
    filenames = [filename for filename in os.listdir(directory) if filename.endswith('.ogg')]
    filenames.sort(key=extract_number)

    for filename in tqdm(filenames, desc="Predicting"):
        audio_data = preprocessed_data[extract_number(filename)]
        ai_prob, human_prob = predict_single(audio_data)

        result = {
            'id': filename.split('.')[0],
            'fake': float(ai_prob),
            'real': float(human_prob)
        }
        results.append(result)

    return results

# 결과를 저장하는 함수
def save_results(results, output_file='../fakeAudio/sample_submission.csv'):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f'Results saved to {output_file}')

# 예측 실행
directory = '../fakeDetection/data/test'
results = predict_all(directory)
save_results(results)
