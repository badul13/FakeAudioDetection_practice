import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras_tuner.tuners import RandomSearch
from keras_tuner import HyperModel
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
        # 레이블을 [AI확률, 사람확률] 형태로 변경
        batch_y = np.column_stack((batch_y, 1 - batch_y))
        return np.expand_dims(batch_x, axis=-1), batch_y

    def get_validation_data(self):
        if self.is_training:
            val_x = self.data[self.val_indices]
            val_y = self.labels[self.val_indices]
            # 검증 데이터의 레이블도 [AI확률, 사람확률] 형태로 변경
            val_y = np.column_stack((val_y, 1 - val_y))
            return np.expand_dims(val_x, axis=-1), val_y
        else:
            return None, None

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
        output = Dense(2, activation='sigmoid')(x)

        model = Model(inputs=audio_input, outputs=output)
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

if __name__ == "__main__":
    data_file = 'data.npy'
    labels_file = 'labels.npy'

    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        raise FileNotFoundError("Preprocessed data not found. Please ensure the data.npy and labels.npy files exist.")

    train_generator = DataGenerator(data_file, labels_file, batch_size=32, is_training=True)

    tuner = RandomSearch(
        MyHyperModel(),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,
        directory='hyperparam_tuning'
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

    tuner.search(train_generator, epochs=10, validation_data=train_generator.get_validation_data(),
                 callbacks=[reduce_lr, checkpoint])

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save('best_model.keras')
    print("Hyperparameter tuning completed and model saved to 'best_model.keras'")