import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import librosa  # 오디오 처리용
import os  # 파일 경로, 디렉토리 처리용
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 오디오 데이터를 로드하고 전처리하는 함수
def load_audio_data(audio_path, target_sr=16000):
    y, sr = librosa.load(audio_path, sr=target_sr)
    return y, sr

# 데이터셋 준비 함수
def load_dataset(base_path, target_sr=16000, max_len=16000):
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
    return np.array(data), np.array(labels)

# 데이터셋 로드
base_path = 'C:/Users/Jeong Taehyeon/OneDrive/바탕 화면/archive'
data, labels = load_dataset(base_path)

# 데이터셋 분할
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# 모델 정의
audio_input = Input(shape=(16000, 1))
x = Conv1D(32, 3, activation='relu')(audio_input)
x = MaxPooling1D(2)(x)
x = Dropout(0.2)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.2)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.2)(x)
x = Conv1D(256, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=audio_input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 체크포인트 저장
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# 데이터 차원 확장
train_data = np.expand_dims(train_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)

# 모델 학습
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# 모델 평가
loss, accuracy = model.evaluate(test_data, test_labels)
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

# # 예시 예측 실행
# predict('C:/Users/Jeong Taehyeon/OneDrive/바탕 화면/archive/predict/biden-to-linus.wav')
