import os
from collections import defaultdict

def analyze_temperature(file_path):
    temperature_data = defaultdict(float)
    with open(file_path, "r") as file:
        for line in file:
            time, temperature = line.strip().split(",")
            temperature_data[float(time)] = float(temperature)

    return temperature_data

def main():
    temperature_folder = "/Users/daniiltesluk/Desktop/ФТЛ/Проект/records/13.02/records2/"
    results = defaultdict(list)

    for file_name in os.listdir(temperature_folder):
        if file_name == "results.txt":
            file_path = os.path.join(temperature_folder, file_name)
            temperature_data = analyze_temperature(file_path)
            results[file_name] = temperature_data

    for file_name, temperature_data in results.items():
        print(f"Результаты анализа для файла {file_name}:")
        for time, temperature in temperature_data.items():
            print(f"Время: {time}, Температура: {temperature}")

if __name__ == "__main__":
    main()
