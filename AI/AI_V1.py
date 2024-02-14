import os
import numpy as np
import tensorflow as tf
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Функция для загрузки аудиозаписи и применения преобразования Фурье
def load_and_transform_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    fft = np.abs(np.fft.fft(y))
    return fft

# Функция для чтения файла с данными о температуре
def read_temperature_data(temperature_file):
    df = pd.read_csv(temperature_file, header=None, names=['time', 'temperature'])
    return df

# Функция для поиска наиболее близкого значения температуры к данному времени
def find_closest_temperature(time, temperature_df):
    idx = np.argmin(np.abs(temperature_df['time'] - time))
    return temperature_df['temperature'].iloc[idx]

# Функция для загрузки данных из папки с записями
def load_data_from_folder(audio_folder, temperature_file):
    # Загрузка аудиофайлов
    audio_data = []
    temperature_data = []

    temperature_df = read_temperature_data(temperature_file)

    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(audio_folder, audio_file)
            time = int(audio_file.split('_')[1].split('.')[0])
            fft = load_and_transform_audio(audio_path)
            temperature = find_closest_temperature(time, temperature_df)
            audio_data.append(fft)
            temperature_data.append(temperature)

    X = np.array(audio_data)
    y = np.array(temperature_data)
    return X, y

# Функция для объединения данных из двух папок
def combine_data(folder1, folder2, temperature_file1, temperature_file2):
    X1, y1 = load_data_from_folder(folder1, temperature_file1)
    X2, y2 = load_data_from_folder(folder2, temperature_file2)
    X_combined = np.concatenate((X1, X2), axis=0)
    y_combined = np.concatenate((y1, y2), axis=0)
    return X_combined, y_combined

# Функция для обучения и оценки модели с использованием кросс-валидации
def train_and_evaluate_model(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits)
    losses = []
    all_history = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1],)),
            tf.keras.layers.Reshape((X.shape[1], 1)),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Ранняя остановка
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Обучение модели
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])
        all_history.append(history.history)

        # Оценка модели
        loss = model.evaluate(X_test, y_test)
        losses.append(loss)

    return losses, all_history, model

# Функция для предсказания температуры для тестовых данных
def predict_temperature(model, audio_folder, temperature_file):
    X_test, y_test = load_data_from_folder(audio_folder, temperature_file)
    predictions = model.predict(X_test)
    return predictions, y_test

# Пути к папкам с записями и файлам с данными о температуре
folder1 = '/Users/daniiltesluk/Desktop/ФТЛ/Проект/records/13.02/records'
temperature_file1 = '/Users/daniiltesluk/Desktop/ФТЛ/Проект/records/13.02/records/results.txt'

folder2 = '/Users/daniiltesluk/Desktop/ФТЛ/Проект/records/13.02/records2'
temperature_file2 = '/Users/daniiltesluk/Desktop/ФТЛ/Проект/records/13.02/records2/results.txt'

# Объединение данных из двух папок
X_combined, y_combined = combine_data(folder1, folder2, temperature_file1, temperature_file2)

# Вывод размерности входных данных
print("Размерность входных данных:", X_combined.shape)

# Вычисление общего количества точек данных
total_data_points = X_combined.shape[0] * X_combined.shape[1]
print("Всего точек данных:", total_data_points)

# Обучение и оценка модели на комбинированных данных с использованием сверточных слоев
losses_combined, all_history, trained_models = train_and_evaluate_model(X_combined, y_combined)
print("Mean Loss:", np.mean(losses_combined))


# Построение графиков на одном рисунке
plt.figure(figsize=(12, 6))

for i, history in enumerate(all_history):
    plt.plot(history['loss'], label='Fold {} Training Loss'.format(i+1))
    plt.plot(history['val_loss'], label='Fold {} Validation Loss'.format(i+1))

plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Предсказание температуры для тестовых данных из второй папки
predictions, y_test = predict_temperature(model, folder2, temperature_file2)

# Расчет погрешности
errors = np.abs(predictions - y_test)

# Вывод средней погрешности
print("Mean Absolute Error:", np.mean(errors))

# Вывод результатов
for i in range(len(predictions)):
    print("Predicted temperature:", predictions[i], "Actual temperature:", y_test[i])

# Создание интервалов для температур
temp_intervals = np.arange(np.min(y_test), np.max(y_test), 5)

# Нахождение средней погрешности для каждого интервала температур
mean_errors = []
for i in range(len(temp_intervals) - 1):
    mask = (y_test >= temp_intervals[i]) & (y_test < temp_intervals[i + 1])
    mean_error = np.mean(errors[mask])
    mean_errors.append(mean_error)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(temp_intervals[:-1], mean_errors, marker='o', linestyle='-')
plt.title('Mean Absolute Error vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Mean Absolute Error')
plt.grid(True)
plt.show()

# Сохранение модели в формате Keras
model.save("temperature_prediction_model.keras")

