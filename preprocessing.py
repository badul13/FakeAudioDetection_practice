import numpy as np
import librosa
import os
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, lfilter

# 오디오 데이터를 로드하는 함수
def load_audio_data(audio_path, target_sr=16000):
    y, sr = librosa.load(audio_path, sr=target_sr)
    return y, sr

# 오디오 데이터를 증강하는 함수
def augment_audio(audio_data):
    # 시간 축소
    time_stretch = librosa.effects.time_stretch(audio_data, rate=np.random.uniform(0.8, 1.0))
    # 노이즈 추가
    noise = np.random.randn(len(audio_data))
    audio_data_noise = audio_data + 0.005 * noise
    # 피치 변환
    pitch_shift = librosa.effects.pitch_shift(audio_data, sr=16000, n_steps=np.random.randint(-5, 5))
    return [time_stretch, audio_data_noise, pitch_shift]

# 오디오 데이터를 패딩하거나 트리밍하는 함수
def pad_or_trim(audio_data, max_len=16000):
    if len(audio_data) < max_len:
        audio_data = np.pad(audio_data, (0, max_len - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:max_len]
    return audio_data

# 밴드 패스 필터를 구성하는 함수
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# 밴드 패스 필터를 적용하는 함수
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 주요 특징값 추출 함수
def extract_features(audio_data, sr):
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr).mean(axis=1)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13).mean(axis=1)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data).mean()
    return np.concatenate([chroma, mfcc, [zero_crossing_rate]])

# 배치를 처리하고 저장하는 함수
def process_and_save_batch(df_batch, base_path, target_sr, max_len, augment, data_file, labels_file, total_processed):
    data = []
    labels = []
    lowcut = 100.0
    highcut = 1500.0

    for _, row in df_batch.iterrows():
        # 파일 경로 생성 수정
        if 'test' in base_path:
            filename = row['path'].split('/')[-1].replace('TEST', 'claer_TEST').replace('.ogg', '.wav')
            file_path = os.path.join(base_path, filename)
        else:
            file_path = os.path.join(base_path, row['path'].strip('./'))

        audio_data, sr = load_audio_data(file_path, target_sr)
        audio_data = bandpass_filter(audio_data, lowcut, highcut, target_sr)
        audio_data = pad_or_trim(audio_data, max_len)

        features = extract_features(audio_data, sr)

        data.append(features)
        if labels_file is not None:  # 테스트 데이터는 라벨이 없음
            labels.append(1 if row['label'] == 'fake' else 0)

        if augment and row['label'] == 'real':
            augmented_data = augment_audio(audio_data)
            augmented_features = [extract_features(aug_data, sr) for aug_data in augmented_data]
            data.extend(augmented_features)
            labels.extend([0] * len(augmented_features))

        total_processed += 1
        print(f'\rProcessed {total_processed} files', end='')

    data = np.array(data, dtype=np.float32)
    if labels_file is not None:
        labels = np.array(labels)

    if os.path.exists(data_file):
        np.save(data_file, np.concatenate((np.load(data_file), data)))
        if labels_file is not None:
            np.save(labels_file, np.concatenate((np.load(labels_file), labels)))
    else:
        np.save(data_file, data)
        if labels_file is not None:
            np.save(labels_file, labels)

    return total_processed

# 데이터셋 로드 및 저장
base_path = '../fakeDetection/data'
train_csv_path = '../fakeDetection/data/train.csv'
test_base_path = '../fakeDetection/data/test_nonBack'
test_csv_path = '../fakeDetection/data/test.csv'
train_data_file = 'data_f.npy'
train_labels_file = 'labels_f.npy'
test_data_file = 'test_data_f.npy'

# 기존 파일 삭제 (새로운 처리 시작)
if os.path.exists(train_data_file):
    os.remove(train_data_file)
if os.path.exists(train_labels_file):
    os.remove(train_labels_file)
if os.path.exists(test_data_file):
    os.remove(test_data_file)

# 학습 데이터 전처리
df = pd.read_csv(train_csv_path)
batch_size = 1000
total_files = len(df)
total_processed = 0

print(f"Total train files to process: {total_files}")

for i in tqdm(range(0, len(df), batch_size), desc="Processing train batches", unit="batch"):
    df_batch = df.iloc[i:i + batch_size]
    total_processed = process_and_save_batch(df_batch, base_path, target_sr=16000, max_len=16000,
                                             augment=False, data_file=train_data_file, labels_file=train_labels_file,
                                             total_processed=total_processed)
    percent_complete = (total_processed / total_files) * 100
    print(f'\rProgress: {percent_complete:.2f}% ({total_processed}/{total_files} files)', end='')

print('\nTrain preprocessing finished.')

# 테스트 데이터 전처리
df_test = pd.read_csv(test_csv_path)
batch_size = 1000
total_test_files = len(df_test)
total_processed_test = 0

print(f"Total test files to process: {total_test_files}")

for i in tqdm(range(0, len(df_test), batch_size), desc="Processing test batches", unit="batch"):
    df_batch_test = df_test.iloc[i:i + batch_size]
    total_processed_test = process_and_save_batch(df_batch_test, test_base_path, target_sr=16000, max_len=16000,
                                                  augment=False, data_file=test_data_file, labels_file=None,
                                                  total_processed=total_processed_test)
    percent_complete = (total_processed_test / total_test_files) * 100
    print(f'\rProgress: {percent_complete:.2f}% ({total_processed_test}/{total_test_files} files)', end='')

print('\nTest preprocessing finished.')
