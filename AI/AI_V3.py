import os
import os.path
import numpy as np
import re
import tensorflow as tf
from audio_analyzer import analyze_audio
from temperature_analyzer import analyze_temperature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='sigmoid', input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def match_temperature_to_audio(audio_files, temperature_data):
    matched_temperatures = []

    for audio_file in audio_files:
        audio_time = float(re.search(r"_(\d+)\.wav", audio_file).group(1))
        
        # Находим ближайшее время снятия замеров температуры
        closest_temperature_time = min(temperature_data, key=lambda x: abs(x[0] - audio_time))
        matched_temperatures.append(closest_temperature_time[1])

    return matched_temperatures

def prepare_data(folders):
    audio_results = []
    temperature_results = []


    features = []
    temperatures = []
    times = []
    for folder in folders:
        for l in open(f"{folder}/results.txt"):
            ts, temperature = l.strip().split(",")
            file_name = f"{folder}/recording_{ts}.wav"
            if not os.path.isfile(file_name):
                continue
            audio_data = analyze_audio(file_name, num_groups=10)
            features.append(audio_data)
            temperatures.append(float(temperature))
            times.append(float(ts))

    features = np.array(features)
    temperatures = np.array(temperatures).reshape(-1, 1)
    times = np.array(times)

    print("LOADING_DONE")
    return features, temperatures, times

def main():
    folders = [
        "/Users/daniiltesluk/Desktop/ФТЛ/Проект/records/13.02/records",
        "/Users/daniiltesluk/Desktop/ФТЛ/Проект/records/13.02/records2",
        "/Users/daniiltesluk/Desktop/ФТЛ/Проект/records/13.02/records3"
    ]

    audio_data, temperature_data, times = prepare_data(folders)
    X_train, X_test, y_train, y_test = train_test_split(audio_data, temperature_data, test_size=0.2, random_state=42)

    model = build_model(input_shape=(X_train.shape[1],))
    history = model.fit(X_train, y_train, epochs=3000, batch_size=32, validation_data=(X_test, y_test))

    # Выводим тесты
    predictions = model.predict(X_test)
    absolute_errors = np.abs(predictions - y_test)
    percentage_errors = np.abs(predictions/y_test - 1)
    mean_absolute_error = np.mean(absolute_errors)
    mean_absolute_percentage_error = np.mean(percentage_errors)*100
    print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error:.2f}%")

    # График предсказанной температуры и реальной температуры
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Temperature')
    plt.plot(predictions, label='Predicted Temperature')
    plt.xlabel('Sample')
    plt.ylabel('Temperature')
    plt.title('Actual vs Predicted Temperature')
    plt.legend()
    plt.show()

    predicted_temperature = model.predict(audio_data)


if __name__ == "__main__":
    main()
