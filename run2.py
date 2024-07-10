import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout, LSTM, GRU, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, average_precision_score
from keras_tuner import HyperModel, RandomSearch, Objective
from scikeras.wrappers import KerasClassifier
from sklearn.inspection import permutation_importance
import shap
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
        ai_output = Dense(1, activation='sigmoid', name='ai_output')(x)
        human_output = Dense(1, activation='sigmoid', name='human_output')(x)

        model = Model(inputs=audio_input, outputs=[ai_output, human_output])
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss={'ai_output': 'binary_crossentropy', 'human_output': 'binary_crossentropy'},
                      metrics={'ai_output': 'accuracy', 'human_output': 'accuracy'})

        print("Building CNN model...")
        return model


class LSTMHyperModel(HyperModel):
    def build(self, hp):
        audio_input = Input(shape=(16000, 1))

        x = LSTM(units=hp.Int('lstm_units_1', min_value=64, max_value=256, step=64),
                 return_sequences=True)(audio_input)

        for i in range(hp.Int('num_lstm_layers', min_value=1, max_value=3)):
            x = LSTM(units=hp.Int(f'lstm_units_{i + 2}', min_value=32, max_value=128, step=32),
                     return_sequences=True if i < (hp.Int('num_lstm_layers') - 1) else False)(x)

        x = Dense(units=hp.Int('dense_units', min_value=64, max_value=256, step=64), activation='relu')(x)
        x = Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)

        ai_output = Dense(1, activation='sigmoid', name='ai_output')(x)
        human_output = Dense(1, activation='sigmoid', name='human_output')(x)

        model = Model(inputs=audio_input, outputs=[ai_output, human_output])
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss={'ai_output': 'binary_crossentropy', 'human_output': 'binary_crossentropy'},
                      metrics=['accuracy', 'accuracy'])

        print("Building LSTM model...")
        return model


class GRUHyperModel(HyperModel):
    def build(self, hp):
        audio_input = Input(shape=(16000, 1))

        x = GRU(units=hp.Int('gru_units_1', min_value=64, max_value=256, step=64),
                return_sequences=True)(audio_input)

        for i in range(hp.Int('num_gru_layers', min_value=1, max_value=3)):
            x = GRU(units=hp.Int(f'gru_units_{i + 2}', min_value=32, max_value=128, step=32),
                    return_sequences=True if i < (hp.Int('num_gru_layers') - 1) else False)(x)

        x = Dense(units=hp.Int('dense_units', min_value=64, max_value=256, step=64), activation='relu')(x)
        x = Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)

        ai_output = Dense(1, activation='sigmoid', name='ai_output')(x)
        human_output = Dense(1, activation='sigmoid', name='human_output')(x)

        model = Model(inputs=audio_input, outputs=[ai_output, human_output])
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss={'ai_output': 'binary_crossentropy', 'human_output': 'binary_crossentropy'},
                      metrics=['accuracy', 'accuracy'])

        print("Building GRU model...")
        return model


class EnsembleHyperModel(HyperModel):
    def __init__(self, base_models):
        self.base_models = base_models

    def build(self, hp):
        audio_input = Input(shape=(16000, 1))

        base_model_outputs = []
        for model in self.base_models:
            base_model = load_model(model)
            for layer in base_model.layers:
                layer.trainable = False
            base_model_outputs.append(base_model(audio_input))

        concatenated = Concatenate()(base_model_outputs)
        x = Dense(units=hp.Int('dense_units', min_value=64, max_value=256, step=64), activation='relu')(concatenated)
        x = Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)

        ai_output = Dense(1, activation='sigmoid', name='ai_output')(x)
        human_output = Dense(1, activation='sigmoid', name='human_output')(x)

        model = Model(inputs=audio_input, outputs=[ai_output, human_output])
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4])),
                      loss={'ai_output': 'binary_crossentropy', 'human_output': 'binary_crossentropy'},
                      metrics=['accuracy', 'accuracy'])

        print("Building Ensemble model...")
        return model


# 데이터 파일 경로
data_file = 'data.npy'
labels_file = 'labels.npy'

# 데이터 생성기 초기화
train_generator = DataGenerator(data_file, labels_file, batch_size=32, is_training=True)

# 각 모델 하이퍼파라미터 튜닝 및 최적 모델 저장
base_models = []
model_classes = [CNNHyperModel, LSTMHyperModel, GRUHyperModel]

for i, model_class in enumerate(model_classes):
    print(f"Starting hyperparameter search for {model_class.__name__}...")
    tuner = RandomSearch(
        model_class(),
        objective=Objective('val_ai_output_accuracy', direction='max'),
        max_trials=7,
        executions_per_trial=2,
        directory=f'hyperparam_tuning_model_{i}',
        max_consecutive_failed_trials=5
    )
    tuner.search(train_generator, epochs=7, validation_data=train_generator.get_validation_data(),
                 callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(f'best_model_{model_class.__name__}.keras')
    base_models.append(f'best_model_{model_class.__name__}.keras')

# 앙상블 모델 하이퍼파라미터 튜닝
ensemble_tuner = RandomSearch(
    EnsembleHyperModel(base_models),
    objective=Objective('val_ai_output_accuracy', direction='max'),
    max_trials=7,
    executions_per_trial=2,
    directory='hyperparam_tuning_ensemble',
    max_consecutive_failed_trials=5
)
ensemble_tuner.search(train_generator, epochs=7, validation_data=train_generator.get_validation_data(),
                      callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])
best_ensemble_model = ensemble_tuner.get_best_models(num_models=1)[0]
best_ensemble_model.save('final_model.keras')
