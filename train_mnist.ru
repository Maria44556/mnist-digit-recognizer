import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# 1. Загрузка данных MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Нормализация (приведение значений пикселей к диапазону 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Создание модели (Простая полносвязная сеть)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 4. Компиляция
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Обучение
print("Начинаем обучение...")
model.fit(x_train, y_train, epochs=5)

# 6. Оценка точности
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nТочность на тестовых данных: {test_acc:.4f}")

# 7. Сохранение модели
model.save('mnist_model.h5')
print("Модель сохранена как mnist_model.h5")