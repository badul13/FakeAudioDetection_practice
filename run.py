import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout, LSTM, GRU, \
    GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, average_precision_score
from keras_tuner import HyperModel, RandomSearch, Objective
from scikeras.wrappers import KerasClassifier
from sklearn.inspection import permutation_importance
import shap

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
        # 각 클래스에 대해 독립적인 시그모이드 출력
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

        # 첫 번째 LSTM 레이어
        x = LSTM(units=hp.Int('lstm_units_1', min_value=64, max_value=256, step=64),
                 return_sequences=True)(audio_input)

        # 추가 LSTM 레이어
        for i in range(hp.Int('num_lstm_layers', min_value=1, max_value=3)):
            x = LSTM(units=hp.Int(f'lstm_units_{i + 2}', min_value=32, max_value=128, step=32),
                     return_sequences=True if i < (hp.Int('num_lstm_layers') - 1) else False)(x)

        # Dense 레이어
        x = Dense(units=hp.Int('dense_units', min_value=64, max_value=256, step=64), activation='relu')(x)
        x = Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)

        # Output 레이어
        ai_output = Dense(1, activation='sigmoid', name='ai_output')(x)
        human_output = Dense(1, activation='sigmoid', name='human_output')(x)

        # 모델 구성
        model = Model(inputs=audio_input, outputs=[ai_output, human_output])

        # 컴파일
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss={'ai_output': 'binary_crossentropy', 'human_output': 'binary_crossentropy'},
                      metrics=['accuracy', 'accuracy'])

        print("Building LSTM model...")
        return model


class GRUHyperModel(HyperModel):
    def build(self, hp):
        audio_input = Input(shape=(16000, 1))

        # 첫 번째 GRU 레이어
        x = GRU(units=hp.Int('gru_units_1', min_value=64, max_value=256, step=64),
                return_sequences=True)(audio_input)

        # 추가 GRU 레이어
        for i in range(hp.Int('num_gru_layers', min_value=1, max_value=3)):
            x = GRU(units=hp.Int(f'gru_units_{i + 2}', min_value=32, max_value=128, step=32),
                    return_sequences=True if i < (hp.Int('num_gru_layers') - 1) else False)(x)

        # Dense 레이어
        x = Dense(units=hp.Int('dense_units', min_value=64, max_value=256, step=64), activation='relu')(x)
        x = Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)

        # Output 레이어
        ai_output = Dense(1, activation='sigmoid', name='ai_output')(x)
        human_output = Dense(1, activation='sigmoid', name='human_output')(x)

        # 모델 구성
        model = Model(inputs=audio_input, outputs=[ai_output, human_output])

        # 컴파일
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss={'ai_output': 'binary_crossentropy', 'human_output': 'binary_crossentropy'},
                      metrics=['accuracy', 'accuracy'])

        print("Building GRU model...")
        return model


class EnsembleHyperModel(HyperModel):
    def __init__(self, base_models):
        self.base_models = base_models
        super().__init__()

    def build(self, hp):
        inputs = Input(shape=(16000, 1))
        outputs = [model(inputs) for model in self.base_models]

        x = Concatenate()(outputs)
        x = Dense(hp.Int('ensemble_dense_1', min_value=32, max_value=128, step=32), activation='relu')(x)
        x = Dropout(hp.Float('ensemble_dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
        x = Dense(hp.Int('ensemble_dense_2', min_value=16, max_value=64, step=16), activation='relu')(x)
        ai_output = Dense(1, activation='sigmoid', name='ai_output')(x)
        human_output = Dense(1, activation='sigmoid', name='human_output')(x)

        ensemble_model = Model(inputs=inputs, outputs=[ai_output, human_output])
        ensemble_model.compile(optimizer=Adam(hp.Choice('ensemble_lr', values=[1e-3, 1e-4, 1e-5])),
                               loss={'ai_output': 'binary_crossentropy', 'human_output': 'binary_crossentropy'},
                               metrics=['accuracy', 'accuracy'])
        print("Building ENSEMBLE model...")
        return ensemble_model


def evaluate_multi_label(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).astype(int)

    micro_f1 = f1_score(y_true, y_pred_binary, average='micro')
    macro_f1 = f1_score(y_true, y_pred_binary, average='macro')
    hl = hamming_loss(y_true, y_pred_binary)
    exact_match = accuracy_score(y_true, y_pred_binary)
    avg_precision = average_precision_score(y_true, y_pred)

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'hamming_loss': hl,
        'exact_match_ratio': exact_match,
        'avg_precision': avg_precision
    }


def perform_permutation_importance(model, X, y):
    wrapped_model = KerasClassifier(build_fn=lambda: model)
    results = permutation_importance(wrapped_model, X, y, n_repeats=10, random_state=42)
    return results


def perform_shap_analysis(model, X):
    explainer = shap.DeepExplainer(model, X[:100])
    shap_values = explainer.shap_values(X[:1000])
    return shap_values


if __name__ == "__main__":
    data_file = 'data.npy'
    labels_file = 'labels.npy'

    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        raise FileNotFoundError("Preprocessed data not found. Please ensure the data.npy and labels.npy files exist.")

    train_generator = DataGenerator(data_file, labels_file, batch_size=32, is_training=True)
    test_generator = DataGenerator(data_file, labels_file, batch_size=32, is_training=False)

    base_models = []
    model_classes = [CNNHyperModel, LSTMHyperModel, GRUHyperModel]

    for i, model_class in enumerate(model_classes):
        tuner = RandomSearch(
            model_class(),
            objective = Objective('ai_output_accuracy', 'human_output_accuracy'),
            max_trials=10,
            executions_per_trial=2,
            directory=f'hyperparam_tuning_model_{i}',
            max_consecutive_failed_trials=5
        )
        tuner.search(train_generator, epochs=10, validation_data=train_generator.get_validation_data())
        best_model = tuner.get_best_models(num_models=1)[0]
        base_models.append(best_model)

    ensemble_tuner = RandomSearch(
        EnsembleHyperModel(base_models),
        objective = Objective('ai_output_accuracy', 'human_output_accuracy'),
        max_trials=10,
        executions_per_trial=2,
        directory='hyperparam_tuning_ensemble',
        max_consecutive_failed_trials=5
    )

    ensemble_tuner.search(train_generator, epochs=10, validation_data=train_generator.get_validation_data())
    best_ensemble_model = ensemble_tuner.get_best_models(num_models=1)[0]

    checkpoint = ModelCheckpoint('final_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

    history = best_ensemble_model.fit(
        train_generator,
        epochs=50,
        validation_data=train_generator.get_validation_data(),
        callbacks=[checkpoint, reduce_lr]
    )

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    val_x, val_y = train_generator.get_validation_data()
    val_pred = best_ensemble_model.predict(val_x)
    eval_results = evaluate_multi_label(val_y, val_pred)

    for metric, value in eval_results.items():
        print(f'{metric}: {value:.4f}')

    print("Performing permutation importance analysis...")
    perm_importance = perform_permutation_importance(best_ensemble_model, val_x, val_y)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(perm_importance.importances_mean)), perm_importance.importances_mean)
    plt.title('Permutation Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.show()

    print("Performing SHAP analysis...")
    shap_values = perform_shap_analysis(best_ensemble_model, val_x)

    plt.figure()
    shap.summary_plot(shap_values, val_x)
    plt.show()

    final_model_path = 'final_model.keras'
    best_ensemble_model.save(final_model_path)
    print(f'Model saved to {final_model_path}')