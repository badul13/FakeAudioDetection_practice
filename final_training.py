import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence

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
        return np.expand_dims(batch_x, axis=-1), batch_y

    def get_validation_data(self):
        if self.is_training:
            val_x = self.data[self.val_indices]
            val_y = self.labels[self.val_indices]
            return np.expand_dims(val_x, axis=-1), val_y
        else:
            return None, None


if __name__ == "__main__":
    data_file = 'data.npy'
    labels_file = 'labels.npy'

    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        raise FileNotFoundError("Preprocessed data not found. Please ensure the data.npy and labels.npy files exist.")

    train_generator = DataGenerator(data_file, labels_file, batch_size=32, is_training=True)
    test_generator = DataGenerator(data_file, labels_file, batch_size=32, is_training=False)

    model_path = 'best_model.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please ensure the model file exists after hyperparameter tuning.")

    best_model = load_model(model_path)

    checkpoint = ModelCheckpoint('final_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

    history = best_model.fit(
        train_generator,
        epochs=1,
        validation_data=train_generator.get_validation_data(),
        callbacks=[checkpoint]
    )

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_accuracy = best_model.evaluate(test_generator)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    final_model_path = 'final_model.h5'
    best_model.save(final_model_path)
    print(f'Model saved to {final_model_path}')
