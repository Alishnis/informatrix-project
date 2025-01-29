from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('/Users/aliserromankul/Desktop/lessons/hackathon/mysite/myapp/almas.py')

# Предобработка изображения
def preprocess_image(image_path, target_size):
    # Открываем изображение
    image = Image.open(image_path).convert('RGB')
    # Изменяем размер
    image = image.resize(target_size)
    # Преобразуем в массив
    image_array = np.array(image)
    # Нормализуем значения пикселей
    image_array = image_array / 255.0
    # Добавляем batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Путь к изображению
image_path = "/Users/aliserromankul/Desktop/lessons/hackathon/mysite/data/test/adenocarcinoma/test.png"

# Предобработка
input_data = preprocess_image(image_path, target_size=(224, 224))  # Задайте нужный размер

# Предсказание
predictions = model.predict(input_data)
print("Предсказания:", predictions)