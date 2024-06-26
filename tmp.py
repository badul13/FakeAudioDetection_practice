import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import librosa  # 오디오 처리용
import os  # 파일 경로, 디렉토리 처리용
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras_tuner.tuners import RandomSearch
from keras_tuner import HyperModel


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


# # 데이터셋 준비 함수
# def load_dataset(base_path, target_sr=16000, max_len=16000):
#     data = []
#     labels = []
#     for label, folder in enumerate(['real', 'fake']):
#         folder_path = os.path.join(base_path, folder)
#         for file_name in os.listdir(folder_path):
#             if file_name.endswith('.wav'):
#                 file_path = os.path.join(folder_path, file_name)
#                 audio_data, sr = load_audio_data(file_path, target_sr)
#                 if len(audio_data) < max_len:
#                     audio_data = np.pad(audio_data, (0, max_len - len(audio_data)), mode='constant')
#                 else:
#                     audio_data = audio_data[:max_len]
#                 data.append(audio_data)
#                 labels.append(label)
#     return np.array(data), np.array(labels)

def load_dataset(base_path, target_sr=16000, max_len=16000, augment=False):
    data = []
    labels = []
    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(base_path, folder)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                audio_data, sr = load_audio_data(file_path, target_sr)
                if len(audio_data) < max_len:
                    audio_data = np.pad(audio_data, (0, max_len - len(audio_data)), mode='constant')
                else:
                    audio_data = audio_data[:max_len]
                data.append(audio_data)
                labels.append(label)
                if augment:
                    augmented_data = augment_audio(audio_data)
                    data.extend(augmented_data)
                    labels.extend([label] * len(augmented_data))
    return np.array(data), np.array(labels)


# 데이터셋 로드
base_path = 'C:/Users/Jeong Taehyeon/OneDrive/바탕 화면/archive'
data, labels = load_dataset(base_path)

# 데이터셋 분할
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)


# 하이퍼모델 정의
class MyHyperModel(HyperModel):
    def build(self, hp):
        audio_input = Input(shape=(16000, 1))
        x = Conv1D(filters=hp.Int('filters_1', min_value=16, max_value=64, step=16),
                   kernel_size=hp.Choice('kernel_size_1', values=[3, 5]), activation='relu')(audio_input)
        # x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)

        x = Conv1D(filters=hp.Int('filters_2', min_value=65, max_value=128, step=65),
                   kernel_size=hp.Choice('kernel_size_2', values=[3, 5]), activation='relu')(x)
        # x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)

        x = Conv1D(filters=hp.Int('filters_3', min_value=64, max_value=256, step=64),
                   kernel_size=hp.Choice('kernel_size_3', values=[3, 5]), activation='relu')(x)
        # x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)

        x = Flatten()(x)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=128), activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=audio_input, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


# 데이터 차원 확장
train_data = np.expand_dims(train_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)

# 하이퍼파라미터 튜닝
tuner = RandomSearch(MyHyperModel(), objective='val_accuracy', max_trials=10, executions_per_trial=2,
                     directory='hyperparam_tuning')

# 학습률 조정
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

# 모델 체크포인트 저장
checkpoint = ModelCheckpoint('best_model_weights.h5', monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True)


# 튜닝 및 모델 학습
tuner.search(train_data, train_labels, epochs=10, validation_split=0.2, callbacks=[reduce_lr, checkpoint])

# 최적의 모델 가중치만 로드
best_model = tuner.get_best_models(num_models=1)[0]
best_model.load_weights('best_model_weights.h5')

# 옵티마이저를 수동으로 설정하여 모델 컴파일
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 평가
loss, accuracy = best_model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 학습 과정 시각화
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# 예측 함수
def predict(audio_path):
    audio_data, sr = load_audio_data(audio_path)
    if len(audio_data) < 16000:
        audio_data = np.pad(audio_data, (0, 16000 - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:16000]

    audio_data = np.expand_dims(audio_data, axis=(0, -1))

    prediction = model.predict(audio_data)
    label = 'Human' if prediction < 0.5 else 'AI Generated'
    print(f'Prediction: {label}')
    print(f'Confidence: {prediction[0][0]:.2f}')

#
# # 예시 예측 실행
# predict('C:/Users/Jeong Taehyeon/OneDrive/바탕 화면/archive/predict/biden-to-linus.wav')
