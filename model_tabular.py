import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import os

class TabularTransformerClassifier:
    def __init__(self, input_shape, model_path='tabular_transformer_model.h5', **kwargs):
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = self._build_model(input_shape, **kwargs)
        self.history = None

    def _build_model(self, input_shape, num_classes=2, dropout_rate=0.1):
        inputs = layers.Input(shape=input_shape)

        x = layers.Flatten()(inputs)  # Achata para fully-connected

        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)

        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        checkpoint = callbacks.ModelCheckpoint(
            self.model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        )
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint],
            verbose=1
        )
        # Carrega os melhores pesos
        self.model.load_weights(self.model_path)
        return self.model

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

        # Coleta dados reais e predições
        y_true = []
        y_scores = []

        for images, labels in self.val_ds:
            preds = self.model.predict(images).flatten()
            y_scores.extend(preds)
            y_true.extend(labels.numpy())

        y_pred = [1 if score >= 0.5 else 0 for score in y_scores]

        # Matriz de confusão
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

        # Classification report
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, y_pred, digits=3))

    def save_model(self, path=None):
        self.model.save(path or self.model_path)

    def load_model(self, path=None):
        self.model = models.load_model(path or self.model_path)
        return self.model
