import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model

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

# 배치 예측 함수
def batch_predict(audio_data_batch):
    predictions = model.predict(audio_data_batch)
    return predictions

# 디렉토리 내 모든 파일 예측 함수
def predict_all(directory, batch_size=32):
    preprocessed_data = load_preprocessed_data()
    results = []
    filenames = [filename for filename in os.listdir(directory) if filename.endswith('.ogg')]
    num_files = len(filenames)
    num_batches = (num_files + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_filenames = filenames[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        audio_data_batch = np.array(
            [preprocessed_data[i] for i in range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, num_files))])
        audio_data_batch = np.expand_dims(audio_data_batch, axis=-1)
        predictions = batch_predict(audio_data_batch)

        # 각 파일에 대해 예측 결과 저장
        for filename, confidence in zip(batch_filenames, predictions):
            ai_prob = confidence[0]  # 모델의 첫 번째 출력이 AI 확률
            human_prob = confidence[1]  # 모델의 두 번째 출력이 실제 확률
            results.append({
                'id': filename.split('.')[0],
                'ai_prob': ai_prob,
                'human_prob': human_prob
            })

        # 진행도 출력
        print(f'Progress: {batch_idx + 1}/{num_batches} ({((batch_idx + 1) / num_batches) * 100:.2f}%)')

    # 예측 결과를 sample_submission.csv 파일에 저장
    df = pd.DataFrame(results)
    df.to_csv('F:/FADdata/sample_submission.csv', index=False)
    print(f'Predictions saved to F:/FADdata/sample_submission.csv')

# 예시 예측 실행
predict_all('F:/FADdata/test')
