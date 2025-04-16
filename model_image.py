import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns


class ImageClassifier:
    def __init__(self, data_dir, name_model, img_size=(128, 128), batch_size=32, val_split=0.2, seed=123):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.model = None
        self.history = None
        self.model_path = f"{name_model}.h5"

        self._prepare_data()

    def _prepare_data(self):
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.val_split,
            subset="training",
            seed=self.seed,
            image_size=self.img_size,
            batch_size=self.batch_size
        )

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.val_split,
            subset="validation",
            seed=self.seed,
            image_size=self.img_size,
            batch_size=self.batch_size
        )

        normalization_layer = layers.Rescaling(1./255)
        self.train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        self.val_ds = self.val_ds.map(lambda x, y: (normalization_layer(x), y))

    def build_model(self):
        self.model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.img_size + (3,)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )


    def train(self, epochs=10):
        if self.model is None:
            self.build_model()

        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=[checkpoint_cb]
        )

    def predict_image(self, image_path):
        if self.model is None:
            print("O modelo ainda não foi carregado ou treinado.")
            return

        # Carregar e preprocessar a imagem
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normaliza
        img_array = np.expand_dims(img_array, axis=0)  # [1, h, w, 3]

        # Fazer predição
        prediction = self.model.predict(img_array)

        return prediction

    def plot_history(self):
        if not self.history:
            print("Modelo ainda não foi treinado.")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
    def plot_metrics(self):
        if self.model is None:
            print("Modelo não carregado.")
            return

        y_true = []
        y_pred = []
        y_scores = []

        for images, labels in self.val_ds:
            preds = self.model.predict(images)  # shape: (batch, 2)
            y_scores.extend(preds[:, 1])  # Probabilidade da classe 1
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(labels.numpy())

        # Matriz de Confusão
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusão')
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.show()

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Relatório
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, y_pred, digits=3))


    def load_best_model(self):
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print("Melhor modelo carregado com sucesso.")
        else:
            print("Modelo salvo não encontrado.")

