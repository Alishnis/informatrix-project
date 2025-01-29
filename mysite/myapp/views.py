from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.shortcuts import render, redirect


from rest_framework.response import Response
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .models import Analysis
from rest_framework.decorators import permission_classes
from rest_framework.permissions import IsAuthenticated

def analysis_page(request):
    if request.user.is_authenticated:  # Проверяем, авторизован ли пользователь
        return render(request, 'main.html')  # Шаблон для авторизованных
    else:
        return render(request, 'main2.html')  # Шаблон для всех


# def user_kab(request):
    
#     if request.user.is_authenticated:
        
#         analyses = Analysis.objects.filter(user=request.user).order_by('-id') # Получаем анализы пользователя
        
#         return render(request, 'user_kab.html', {
#             'user': request.user,
#             'analyses': analyses
#         })
#     return render(request,'error.html')



# def register_view(request):
#     if request.method == 'POST':
#         email = request.POST.get('email')
#         password = request.POST.get('password')
        
#         # Проверка существования пользователя
#         if User.objects.filter(username=email).exists():
#             messages.error(request, 'Пользователь с таким email уже зарегистрирован.')
#         else:
#             # Создаем нового пользователя
#             user = User.objects.create_user(username=email, password=password)
#             user.save()
#             login(request, user)
#             return redirect('analysis_page')
    
#     return render(request, 'register.html')
# def login_view(request):
#     if request.method == 'POST':
#         email = request.POST.get('email')
#         password = request.POST.get('password')
        
#         # Аутентификация пользователя
#         user = authenticate(request, username=email, password=password)
#         if user is not None:
#             login(request, user)
#             return redirect('analysis_page')  # Перенаправление на main.html
#         else:
#             messages.error(request, 'Неверный email или пароль.')

    # return render(request, 'register.html')
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

@login_required(login_url='/login/')  # Перенаправление на страницу входа
def main_view(request):
    return render(request, 'main.html')
 
 


from django.http import JsonResponse
from .models import Analysis

def handle_upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if uploaded_file:
            analysis = Analysis.objects.create(
                user=request.user,  # предполагается, что пользователь авторизован
                
                analysis_file=uploaded_file,
                result='Good',
            )
            return JsonResponse({'message': 'Файл успешно загружен!'}, status=200)
        return JsonResponse({'error': 'Файл не был загружен.'}, status=400)
    return JsonResponse({'error': 'Недопустимый метод запроса.'}, status=405)






# ai
from django.shortcuts import render
from django.http import JsonResponse
 # Если вы используете OpenAI GPT для анализа симптомов

# Инициализация OpenAI 


import subprocess



def analyze_symptoms(request):
    if request.method == 'POST':
        symptoms = request.POST.get('symptoms', '')
        if symptoms:
            try:
                # Запускаем скрипт для анализа симптомов
                result = subprocess.check_output(
                    ['python', 'myapp/ex.py'], 
                    input=symptoms, 
                    text=True
                )
                resultind=result.find('provide possible diagnoses:')
                result1=result[resultind:].replace('provide possible diagnoses:','')
                resultfin=result1.split(',')
                
                
                # Передаём результат в шаблон
                return render(request, 'symptom_form.html', {'diagnosis': resultfin})
            except Exception as e:
                return render(request, 'symptom_form.html', {'diagnosis': f"Ошибка: {e}"})
    return render(request, 'symptom_form.html')   





from django.shortcuts import render
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from myapp.models import Disease
import logging

# Логирование ошибок
logger = logging.getLogger(__name__)

def treatment_view(request):
    """
    Обработка запроса для получения рекомендаций по лечению.
    """
    recommendations = None
    if request.method == 'POST':
        # Получаем введённое заболевание
        illness = request.POST.get('illness', '').strip()
        if illness:
            try:
                # Проверяем наличие болезни в базе данных
                try:
                    treatment = Disease.objects.get(name__iexact=illness)
                except Disease.DoesNotExist:
                    return render(request, 'treatment.html', {
                        'recommendations': "No treatment information available. Consult your doctor."
                    })
                except Disease.MultipleObjectsReturned:
                    return render(request, 'treatment.html', {
                        'recommendations': "Multiple records found for disease. Contact administrator."
                    })

                # Извлечение лечения из базы данных
                recommendations = treatment.treatment
                if not recommendations:
                    return render(request, 'treatment.html', {
                        'recommendations': "No treatment information available. Consult your doctor."
                    })

                

                

            except Exception as e:
                # Логируем ошибки
                logger.error(f"Error processing request: {str(e)}")
                recommendations = f"An error occurred while processing data: {str(e)}"

    return render(request, 'treatment.html', {'recommendations': recommendations})




from django.shortcuts import render
from django.conf import settings
from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision.models import densenet121
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.transform import resize
import numpy as np

# Предобработка изображений
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка модели
def load_chexnet():
    """Загружаем предобученную модель DenseNet121."""
    model = densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 14)
    model.eval()
    return model

model = load_chexnet()

def analyze_predictions(input_tensor):
    """Анализируем предсказания модели и возвращаем список заболеваний."""
    with torch.no_grad():
        outputs = model(input_tensor)
    probabilities = torch.sigmoid(outputs).numpy()[0]

    # Список заболеваний
    classes = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
    ]

    # Порог вероятности (50%)
    detected = [
        f"{classes[i]}: {prob * 100:.2f}%" for i, prob in enumerate(probabilities) if prob > 0.3
    ]
    return detected

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from skimage.transform import resize
from matplotlib import pyplot as plt

def generate_gradcam_keras(model, input_tensor, last_conv_layer_name, image_path):
    """Генерация Grad-CAM визуализации для Keras модели."""
    # Создаем модель для получения активаций последнего сверточного слоя и предсказаний
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        predicted_class = tf.argmax(predictions[0])  # Находим наиболее вероятный класс
        loss = predictions[:, predicted_class]

    # Вычисляем градиенты
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Генерация карты активации
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # Нормализация

    # Наложение на исходное изображение
    img = Image.open(image_path).convert("RGB")
    heatmap_resized = resize(heatmap, (img.size[1], img.size[0]))
    overlay = show_cam_on_image(np.array(img) / 255.0, heatmap_resized)

    # Сохранение Grad-CAM
    gradcam_dir = os.path.join(settings.MEDIA_ROOT, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)
    gradcam_path = os.path.join(gradcam_dir, f"gradcam_{os.path.basename(image_path)}.png")
    Image.fromarray((overlay * 255).astype(np.uint8)).save(gradcam_path)

    return gradcam_path


from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Analysis

from django.shortcuts import render
from django.contrib import messages
from .models import Analysis
import os
from django.conf import settings
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from skimage.transform import resize
from pytorch_grad_cam.utils.image import show_cam_on_image

def analyze_image(request):
    """Обработка загрузки изображения, генерация Grad-CAM и сохранение результата."""
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']

        # Сохраняем файл временно
        temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp', uploaded_file.name)
        os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
        with open(temp_image_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        try:
            # Предобработка изображения
            img = Image.open(temp_image_path).convert("RGB").resize((128, 128))
            img_array = np.array(img) / 255.0
            input_tensor = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)

            # Выполнение анализа
            predictions = model.predict(input_tensor)
            probabilities = predictions[0]  # Вероятности для каждого класса

            # Список классов заболеваний
            classes = [
                "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
                "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
            ]

            # Форматирование результатов
            detected_diseases = [
                f"{classes[i]}: {prob * 100:.2f}%"
                for i, prob in enumerate(probabilities) if prob > 0.5  # Порог вероятности 50%
            ]
            result_text = "\n".join(detected_diseases)

            # Генерация Grad-CAM
            gradcam_path = generate_gradcam_keras(
                model=model,
                input_tensor=input_tensor,
                last_conv_layer_name="block5_conv3",  # Укажите последний сверточный слой модели
                image_path=temp_image_path
            )

            # Сохраняем Grad-CAM изображение как основное
            gradcam_file_path = os.path.join(settings.MEDIA_ROOT, 'uploads/analysis/', os.path.basename(gradcam_path))
            os.makedirs(os.path.dirname(gradcam_file_path), exist_ok=True)
            os.rename(gradcam_path, gradcam_file_path)  # Перемещаем файл в папку загрузок

            # Сохраняем запись в базе данных
            analysis = Analysis.objects.create(
                user=request.user,
                analysis_file=gradcam_file_path.replace(settings.MEDIA_ROOT, ''),  # Сохраняем относительный путь
                result=result_text
            )           

            # Удаляем временный исходный файл
            os.remove(temp_image_path)

            # Отображаем результат
            return render(request, 'result.html', {
                'diseases': detected_diseases,
                'gradcam_path': gradcam_file_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
                'analysis_id': analysis.id
            })

        except Exception as e:
            return render(request, 'upload.html', {'error': f"Ошибка анализа: {str(e)}"})

    return render(request, 'upload.html')
from .models import AnalysisCT

def save_results(request):
    """Подтверждение сохранения анализа."""
    if request.method == 'POST':
        analysis_id = request.POST.get('analysis_id')
        try:
            # Получаем анализ по ID
            analysis = Analysis.objects.get(id=analysis_id, user=request.user)
            analysis.is_saved = True  # Предположительно, добавлено поле is_saved
            analysis.save()
            messages.success(request, 'Results saved successfully!')
        except Analysis.DoesNotExist:
            messages.error(request, 'Analysis not found.')

        return redirect('user_kab')  # Перенаправление на кабинет пользователя
    return JsonResponse({'error': 'Invalid request method'}, status=405)
def save_results_ct(request):
    """Подтверждение сохранения анализа."""
    if request.method == 'POST':
        analysis_id = request.POST.get('analysis_id')
        try:
            # Получаем анализ по ID
            analysis = AnalysisCT.objects.get(id=analysis_id, user=request.user)
            analysis.is_saved = True  # Предположительно, добавлено поле is_saved
            analysis.save()
            messages.success(request, 'Results saved successfully!')
        except Analysis.DoesNotExist:
            messages.error(request, 'Analysis not found.')

        return redirect('user_kab')  # Перенаправление на кабинет пользователя
    return JsonResponse({'error': 'Invalid request method'}, status=405)


import os
import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Параметры
BASE_DIR = '/Users/aliserromankul/Desktop/lessons/hackathon/mysite/data'
MODEL_PATH = "trained_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Загрузка или обучение модели
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Модель успешно загружена из 'trained_model.h5'")
        return model

    # Генераторы данных
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'valid'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Модель VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Замораживаем слои базовой модели
    for layer in base_model.layers:
        layer.trainable = False

    # Новые слои
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Компиляция
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Обучение
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(train_generator, 
              validation_data=valid_generator, 
              epochs=15, 
              callbacks=[early_stopping])

    # Сохраняем модель
    model.save(MODEL_PATH)
    print(f"Модель сохранена в '{MODEL_PATH}'")
    return model

# Загружаем или обучаем модель
model = load_or_train_model()

# Карта классов
def get_class_labels(train_generator):
    return {v: k for k, v in train_generator.class_indices.items()}

# Обработчик для загрузки изображения и предсказания
# Словарь для преобразования классов в читаемые названия
CLASS_TRANSLATIONS = {
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "Большеклеточная карцинома",
    "adenocarcinoma": "Аденокарцинома",
    "squamous.cell.carcinoma": "Плоскоклеточная карцинома",
    "normal": "Нормальное состояние",
    # Добавьте остальные классы
}

# Обработчик для загрузки изображения и предсказания
def analyze_image2(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']  # Получаем файл из формы
        uploads_dir = os.path.join(BASE_DIR, 'uploads/ct')
        os.makedirs(uploads_dir, exist_ok=True)
        saved_image_path = os.path.join(uploads_dir, uploaded_file.name)

        with open(saved_image_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Сохраняем временно загруженный файл
        temp_image_path = os.path.join(BASE_DIR, 'temp', uploaded_file.name)
        os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
        with open(temp_image_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Предобработка изображения
        img = load_img(temp_image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Добавляем ось для батча

        # Предсказание
        predicted_probs = model.predict(img_array)
        predicted_index = np.argmax(predicted_probs, axis=1)[0]
        predicted_probability = predicted_probs[0][predicted_index] * 100  # Вероятность в процентах

        # Получение карты классов
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            os.path.join(BASE_DIR, 'train'),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        class_labels = get_class_labels(train_generator)

        # Преобразование класса в читаемый формат
        predicted_class = class_labels[predicted_index]
        readable_class = CLASS_TRANSLATIONS.get(predicted_class, "Unknown class")
        analysis = AnalysisCT.objects.create(
                user=request.user,
                analysis_file=os.path.relpath(saved_image_path, BASE_DIR),
                
              
                result=f'{predicted_class}: {predicted_probability:.2f}%'
            )   

        # Удаляем временный файл
        os.remove(temp_image_path)
        

        # Проверка вероятности
        if predicted_probability > 20:
            probability_message = f"{predicted_probability:.2f}"
        else:
            probability_message = "Less than 20%"

        # Рендеринг результата
        return render(request, 'result2.html', {
            'predicted_class': predicted_class,
            'predicted_probability': probability_message,
            'analysis_id':analysis.id,
        })

    return render(request, 'upload2.html')



import requests
from django.shortcuts import render
from .models import Disease

API_URL = "https://disease-info.p.rapidapi.com/diseases"
API_HEADERS = {
    "X-RapidAPI-Key": "a891e945f2msha7985eca0bc5957p1c3ee4jsnc4cd95b357ae",  # Замените на ваш RapidAPI ключ
    "X-RapidAPI-Host": "disease-info.p.rapidapi.com"
}

def fetch_disease_data(request):
    """Получить данные о болезнях из RapidAPI"""
    response = requests.get(API_URL, headers=API_HEADERS)

    if response.status_code == 200:
        data = response.json()

        for disease in data:  # Обработка списка заболеваний
            Disease.objects.update_or_create(
                name=disease.get("disease", "Unknown"),
                defaults={
                    "symptoms": disease.get("symptoms", "N/A"),
                    "description": disease.get("description", "N/A"),
                    "treatment": disease.get("treatment", "N/A"),
                },
            )

        return render(request, "fetch_data.html", {"diseases": Disease.objects.all()})
    else:
        return render(request, "error.html", {"error": response.json()})




