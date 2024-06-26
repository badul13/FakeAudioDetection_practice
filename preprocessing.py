import numpy as np
import librosa
import os

# 오디오 데이터를 로드하고 전처리하는 함수
def load_audio_data(audio_path, target_sr=16000):
    y, sr = librosa.load(audio_path, sr=target_sr)
    return y, sr

# 오디오 데이터를 증강하는 함수
def augment_audio(audio_data):
    # 시간 축소
    time_stretch = librosa.effects.time_stretch(audio_data, rate=np.random.uniform(0.8, 1.2))
    # 노이즈 추가
    noise = np.random.randn(len(audio_data))
    audio_data_noise = audio_data + 0.005 * noise
    # 피치 변환
    pitch_shift = librosa.effects.pitch_shift(audio_data, sr=16000, n_steps=np.random.randint(-5, 5))
    return [time_stretch, audio_data_noise, pitch_shift]

# 오디오 데이터를 패딩하거나 자르는 함수
def pad_or_trim(audio_data, max_len=16000):
    if len(audio_data) < max_len:
        audio_data = np.pad(audio_data, (0, max_len - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:max_len]
    return audio_data

# 데이터셋 준비 함수
def load_dataset(base_path, target_sr=16000, max_len=16000, augment=False):
    data = []
    labels = []
    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(base_path, folder)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                audio_data, sr = load_audio_data(file_path, target_sr)
                audio_data = pad_or_trim(audio_data, max_len)
                data.append(audio_data)
                labels.append(label)
                if augment:
                    augmented_data = augment_audio(audio_data)
                    augmented_data = [pad_or_trim(aug_data, max_len) for aug_data in augmented_data]  # 패딩/자르기 적용
                    data.extend(augmented_data)
                    labels.extend([label] * len(augmented_data))
    return np.array(data), np.array(labels)

# 데이터셋 로드 및 저장
base_path = 'C:/Users/Jeong Taehyeon/OneDrive/바탕 화면/archive'
data, labels = load_dataset(base_path, augment=True)
np.save('data.npy', data)
np.save('labels.npy', labels)

print('preprocessing finished.')