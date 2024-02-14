from collections import defaultdict
from scipy.io import wavfile
import numpy as np
from numpy.fft import fft

def analyze_audio(file_path, num_groups):
    # Чтение аудиофайла
    sample_rate, audio_data = wavfile.read(file_path)

    # Применение преобразования Фурье
    fft_result = fft(audio_data)
    magnitudes = np.fft.fftshift(np.abs(fft_result))
    # frequencies = np.fft.fftfreq(magnitudes.shape[0], 1 / sample_rate)

    bins = magnitudes.reshape(num_groups, -1).sum(axis=1)
    bins /= bins.sum()
    # Группировка частот
    # freq_groups = [0]*num_groups
    # for freq, mag in zip(frequencies, magnitudes):
    #     group_idx = int(num_groups * freq / (sample_rate / 2))
    #     freq_groups[group_idx] += mag

    # # Нормировка значений в группах
    # total_sum = sum(freq_groups)
    # for group_idx, mag_sum in enumerate():
    #     freq_groups[group_idx] = mag_sum / total_sum

    return bins
