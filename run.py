import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras_tuner.tuners import RandomSearch
from keras_tuner import HyperModel
from preprocessing import load_audio_data, pad_or_trim, load_dataset

# 전처리된 데이터를 저장할 파일 경로
data_file = 'data.npy'
labels_file = 'labels.npy'

def preprocess_data():
    base_path = 'C:/Users/Jeong Taehyeon/OneDrive/바탕 화면/archive'
    data, labels = load_dataset(base_path, augment=True)
    np.save(data_file, data)
    np.save(labels_file, labels)
    return data, labels

# 하이퍼모델 정의
class MyHyperModel(HyperModel):
    def build(self, hp):
        audio_input = Input(shape=(16000, 1))
        x = Conv1D(filters=hp.Int('filters_1', min_value=16, max_value=64, step=16),
                   kernel_size=hp.Choice('kernel_size_1', values=[3, 5]), activation='relu')(audio_input)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)

        x = Conv1D(filters=hp.Int('filters_2', min_value=65, max_value=128, step=65),
                   kernel_size=hp.Choice('kernel_size_2', values=[3, 5]), activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)

        x = Conv1D(filters=hp.Int('filters_3', min_value=64, max_value=256, step=64),
                   kernel_size=hp.Choice('kernel_size_3', values=[3, 5]), activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)

        x = Flatten()(x)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=128), activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=audio_input, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

# 전처리된 데이터 로드 또는 전처리 수행
if os.path.exists(data_file) and os.path.exists(labels_file):
    print("Loading preprocessed data...")
    data = np.load(data_file)
    labels = np.load(labels_file)
else:
    print("Preprocessing data...")
    data, labels = preprocess_data()

# 데이터셋 분할
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# 데이터 차원 확장
train_data = np.expand_dims(train_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)

# 하이퍼파라미터 튜닝
tuner = RandomSearch(MyHyperModel(), objective='val_accuracy', max_trials=10, executions_per_trial=2, directory='hyperparam_tuning')

# 학습률 조정
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

# 모델 체크포인트 저장
checkpoint = ModelCheckpoint('best_model.weights.h5', monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True)

# 튜닝 및 모델 학습
tuner.search(train_data, train_labels, epochs=10, validation_split=0.2, callbacks=[reduce_lr, checkpoint])

# 최적의 하이퍼파라미터로 모델 훈련
best_model = tuner.get_best_models(num_models=1)[0]

# 최적의 가중치 로드 (옵티마이저 상태 제외)
best_model.load_weights('best_model.weights.h5')

# 모델 평가
loss, accuracy = best_model.evaluate(test_data, test_labels)
print(f'Pre-test Accuracy: {accuracy * 100:.2f}%')

# 학습 과정 시각화
history = best_model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# 모델 평가
loss, accuracy = best_model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 최종 모델 저장
final_model_path = 'final_model.h5'
best_model.save(final_model_path)
print(f'Model saved to {final_model_path}')

# 예측 함수
# def predict(audio_path):
#     audio_data, sr = load_audio_data(audio_path)
#     audio_data = pad_or_trim(audio_data, 16000)
#     audio_data = np.expand_dims(audio_data, axis=(0, -1))
#
#     prediction = best_model.predict(audio_data)
#     label = 'Human' if prediction < 0.5 else 'AI Generated'
#     print(f'FileName: {audio_path}')
#     print(f'Prediction: {label}')
#     print(f'Confidence: {prediction[0][0]:.2f}')
#
# # 디렉토리 내 모든 파일 예측 함수
# def predict_all(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith('.wav'):  # .wav 파일만 예측
#             filepath = os.path.join(directory, filename)
#             predict(filepath)
#
# # 예시 예측 실행
# predict_all('C:/Users/Jeong Taehyeon/OneDrive/바탕 화면/archive/predict/')
