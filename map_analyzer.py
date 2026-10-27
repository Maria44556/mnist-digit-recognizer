import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import re

def final_map_analysis(image_path):
    print("1. Загрузка карты...")
    img = cv2.imread(image_path)

    # Увеличиваем в 2 раза
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Переводим в оттенки серого (без жестких цветовых фильтров, как в прошлый раз)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Делаем контраст (черно-белое изображение)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 3)

    print("2. Поиск текста по квадратам (это займет 2-3 минуты)...")

    height, width = thresh.shape
    tile_size = 400 # Уменьшили квадрат, чтобы лучше видеть мелкий текст
    found_heights = []

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = thresh[y:y+tile_size, x:x+tile_size]

            # Ищем цифры (включая точку и запятую)
            config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.,'
            text = pytesseract.image_to_string(tile, config=config)

            # Ищем числа в тексте
            numbers = re.findall(r'\b\d{2,3}(?:[.,]\d)?\b', text)

            for num in numbers:
                try:
                    val = float(num.replace(',', '.'))
                    # Оставляем только реалистичные высоты (от 40 до 450 метров)
                    if 40.0 <= val <= 450.0:
                        found_heights.append(val)
                except:
                    pass

    # Убираем дубликаты
    found_heights = list(set(found_heights))
    found_heights.sort()

    print("\n" + "="*40)
    print("📊 ИТОГОВАЯ АНАЛИТИЧЕСКАЯ СПРАВКА")
    print("="*40)

    if found_heights:
        print(f"Всего найдено уникальных высот: {len(found_heights)}")
        print(f"Минимальная высота: {min(found_heights):.1f} м")
        print(f"Максимальная высота: {max(found_heights):.1f} м")
        print(f"Средняя высота: {np.mean(found_heights):.1f} м")

        # Выводим список найденных чисел
        print(f"\nНайденные высоты: {found_heights}")

        # Строим график
        plt.figure(figsize=(10, 5))
        plt.hist(found_heights, bins=15, color='#1E90FF', edgecolor='black')
        plt.title('Распределение высот на топографической карте')
        plt.xlabel('Высота (метры)')
        plt.ylabel('Количество отметок')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
    else:
        print("Ничего не найдено. Попробуйте карту в лучшем качестве.")

# Запуск
final_map_analysis('map.png')


import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_distance(image_path, point1, point2, pixels_per_km):
    # Загружаем карту
    img = cv2.imread(image_path)

    # Считаем расстояние в пикселях по теореме Пифагора
    dist_pixels = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    # Переводим в километры
    dist_km = dist_pixels / pixels_per_km

    print("="*40)
    print("📏 АНАЛИЗ РАССТОЯНИЙ")
    print("="*40)
    print(f"Расстояние в пикселях: {dist_pixels:.1f} px")
    print(f"Расстояние в километрах: {dist_km:.2f} км")

    # Рисуем линию для наглядности
    cv2.line(img, point1, point2, (0, 0, 255), 5) # Красная линия
    cv2.circle(img, point1, 10, (255, 0, 0), -1) # Синяя точка 1
    cv2.circle(img, point2, 10, (255, 0, 0), -1) # Синяя точка 2

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Измерение расстояния: {dist_km:.2f} км")
    plt.axis('off')
    plt.show()

# ВАЖНО: Координаты (x, y) в пикселях
# Вам нужно открыть карту в любом просмотрщике картинок и посмотреть координаты двух объектов
# Например, двух городов:
point_A = (300, 400) # Координаты первой точки
point_B = (800, 900) # Координаты второй точки

# КАК УЗНАТЬ pixels_per_km:
# Измерьте линейкой на экране длину отрезка в 1 км (внизу карты есть линейный масштаб).
# Посмотрите, сколько пикселей занимает этот отрезок в вашей картинке.
# Впишите это число сюда:
pixels_in_1km = 140 # <-- ЗАМЕНИТЕ НА ВАШЕ ЗНАЧЕНИЕ

calculate_distance('map.png', point_A, point_B, pixels_in_1km)