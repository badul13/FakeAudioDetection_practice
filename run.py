import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from keras_tuner import HyperModel, RandomSearch, Objective
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Is GPU available: ", tf.test.is_gpu_available())


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
        ai_labels = batch_y
        human_labels = 1 - batch_y
        return np.expand_dims(batch_x, axis=-1), {'ai_output': ai_labels, 'human_output': human_labels}

    def get_validation_data(self):
        if self.is_training:
            val_x = self.data[self.val_indices]
            val_y = self.labels[self.val_indices]
            ai_labels = val_y
            human_labels = 1 - val_y
            return np.expand_dims(val_x, axis=-1), {'ai_output': ai_labels, 'human_output': human_labels}
        else:
            return None, None


class CNNHyperModel(HyperModel):
    def build(self, hp):
        audio_input = Input(shape=(16000, 1))
        x = Conv1D(filters=hp.Int('filters_1', min_value=16, max_value=64, step=16),
                   kernel_size=hp.Choice('kernel_size_1', values=[3, 4, 5]), activation='relu')(audio_input)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)

        x = Conv1D(filters=hp.Int('filters_2', min_value=64, max_value=128, step=64),
                   kernel_size=hp.Choice('kernel_size_2', values=[3, 4, 5]), activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)

        x = Conv1D(filters=hp.Int('filters_3', min_value=32, max_value=128, step=32),
                   kernel_size=hp.Choice('kernel_size_3', values=[3, 4, 5]), activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)

        x = Flatten()(x)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=128), activation='relu')(x)
        x = Dropout(0.5)(x)
        ai_output = Dense(1, activation='sigmoid', name='ai_output')(x)
        human_output = Dense(1, activation='sigmoid', name='human_output')(x)

        model = Model(inputs=audio_input, outputs=[ai_output, human_output])
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss={'ai_output': 'binary_crossentropy', 'human_output': 'binary_crossentropy'},
                      metrics={'ai_output': 'accuracy', 'human_output': 'accuracy'})

        print("Building CNN model...")
        return model


# 데이터 파일 경로
data_file = 'data.npy'
labels_file = 'labels.npy'

# 데이터 생성기 초기화
train_generator = DataGenerator(data_file, labels_file, batch_size=16, is_training=True)

# CNN 모델 하이퍼파라미터 튜닝 및 최적 모델 저장
print(f"Starting hyperparameter search for CNNHyperModel...")
tuner = RandomSearch(
    CNNHyperModel(),
    objective=Objective('val_ai_output_accuracy', direction='max'),
    max_trials=7,
    executions_per_trial=2,
    directory='hyperparam_tuning_cnn',
)
tuner.search(train_generator, epochs=5, validation_data=train_generator.get_validation_data(),
             callbacks=[
                 # ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
                 # EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
                 ModelCheckpoint(filepath='final_model.keras', monitor='val_loss', save_best_only=True, save_freq='epoch')
             ])
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('final_model.keras')
