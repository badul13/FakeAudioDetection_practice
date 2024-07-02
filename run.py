import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras_tuner.tuners import RandomSearch
from keras_tuner import HyperModel
from tensorflow.keras.utils import Sequence

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 데이터 생성기 클래스 정의
class DataGenerator(Sequence):
    def __init__(self, data_file, labels_file, batch_size=32, is_training=True, **kwargs):
        super().__init__(**kwargs)
        self.data = np.load(data_file, mmap_mode='r')
        self.labels = np.load(labels_file)
        self.batch_size = batch_size
        self.is_training = is_training

        if self.is_training:
            self.train_indices, self.val_indices = train_test_split(
                range(len(self.labels)), test_size=0.2, random_state=42
            )
        else:
            self.indices = range(len(self.labels))

    def __len__(self):
        if self.is_training:
            return int(np.ceil(len(self.train_indices) / float(self.batch_size)))
        else:
            return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.is_training:
            batch_indices = self.train_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = self.data[batch_indices]
        batch_y = self.labels[batch_indices]
        return np.expand_dims(batch_x, axis=-1), batch_y

    def get_validation_data(self):
        if self.is_training:
            val_x = self.data[self.val_indices]
            val_y = self.labels[self.val_indices]
            return np.expand_dims(val_x, axis=-1), val_y
        else:
            return None, None


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


# 메인 코드
if __name__ == "__main__":
    # 전처리된 데이터를 저장할 파일 경로
    data_file = 'data.npy'
    labels_file = 'labels.npy'

    # 전처리된 데이터 확인
    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        raise FileNotFoundError("Preprocessed data not found. Please ensure the data.npy and labels.npy files exist.")

    # 데이터 생성기 생성
    train_generator = DataGenerator(data_file, labels_file, batch_size=32, is_training=True)
    test_generator = DataGenerator(data_file, labels_file, batch_size=32, is_training=False)

    # 하이퍼파라미터 튜닝
    tuner = RandomSearch(
        MyHyperModel(),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,
        directory='hyperparam_tuning'
    )

    # 콜백 설정
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    checkpoint = ModelCheckpoint('best_model.weights.h5', monitor='val_accuracy', save_best_only=True, mode='max',
                                 save_weights_only=True)

    # 튜닝 및 모델 학습
    tuner.search(train_generator, epochs=1, validation_data=train_generator.get_validation_data(),
                 callbacks=[reduce_lr, checkpoint])

    # 최적의 하이퍼파라미터로 모델 훈련
    best_model = tuner.get_best_models(num_models=1)[0]

    # 최적의 가중치 로드 (옵티마이저 상태 제외)
    weights_path = 'best_model.weights.h5'
    if os.path.exists(weights_path):
        best_model.load_weights(weights_path)
    else:
        print(f"Warning: {weights_path} not found. Proceeding with the current model weights.")

    # 학습 과정
    history = best_model.fit(
        train_generator,
        epochs=1,
        validation_data=train_generator.get_validation_data(),
        callbacks=[checkpoint]
    )

    # 학습 과정 시각화
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # 모델 평가
    test_loss, test_accuracy = best_model.evaluate(test_generator)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    # 최종 모델 저장
    final_model_path = 'final_model.h5'
    best_model.save(final_model_path)
    print(f'Model saved to {final_model_path}')