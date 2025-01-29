from django.shortcuts import render,redirect
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4
from PIL import Image
import numpy as np
from django.conf import settings
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.transform import resize
from .models import AnalysisSkin
from django.http import JsonResponse

# Предобработка изображений
preprocess = transforms.Compose([
    transforms.Resize((380, 380)),  # Для EfficientNet-B4
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка улучшенной модели EfficientNet-B4
def load_skin_disease_model():
    model = efficientnet_b4(pretrained=True)  # Загружаем предобученные веса
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)  # Настраиваем на 7 классов заболеваний
    model = nn.Sequential(
        model,
        nn.Dropout(0.5)  # Добавляем регуляризацию
    )
    model.eval()
    return model

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, input_tensor, target_layer, predicted_index):
    """
    Генерация Grad-CAM визуализации для анализа значимых областей.
    """
    # Создаём целевой класс для Grad-CAM
    targets = [ClassifierOutputTarget(predicted_index)]
    
    # Создаём Grad-CAM объект
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Генерация карты активации
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Убираем лишнее измерение
    
    return grayscale_cam

# Анализ изображений
def analyze_skin_image(request):
    """Обработка снимков кожи, предсказание заболеваний и визуализация Grad-CAM."""
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']

        # Сохраняем временное изображение
        temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp', uploaded_file.name)
        os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
        with open(temp_image_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        try:
            # Предобработка изображения
            image = Image.open(temp_image_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0)

            # Загрузка модели
            model = load_skin_disease_model()

            # Прогнозирование
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Определяем наибольшую вероятность
            predicted_index = probabilities.argmax().item()
            predicted_class = CLASSES[predicted_index]
            predicted_probability = probabilities[predicted_index].item() * 140

            # Генерация Grad-CAM
            target_layer = model[0].features[-1]  # Последний сверточный слой EfficientNet-B4
            heatmap = generate_gradcam(model[0], input_tensor, target_layer, predicted_index)

            # Наложение Grad-CAM на изображение
            heatmap_resized = np.array(Image.fromarray(heatmap).resize(image.size, Image.LANCZOS))
            heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
            heatmap_colored = np.uint8(255 * heatmap_normalized)
            heatmap_overlay = (np.array(image) * 0.5 + np.expand_dims(heatmap_colored, axis=2) * 0.5).astype(np.uint8)

            # Сохранение Grad-CAM изображения
            gradcam_dir = os.path.join(settings.MEDIA_ROOT, 'gradcam')
            os.makedirs(gradcam_dir, exist_ok=True)
            gradcam_path = os.path.join(gradcam_dir, f"gradcam_{os.path.basename(uploaded_file.name)}")
            Image.fromarray(heatmap_overlay).save(gradcam_path)
            analysis = AnalysisSkin.objects.create(
                user=request.user,
                analysis_file=gradcam_path.replace(settings.MEDIA_ROOT, ''),  # Сохраняем относительный путь
                result=f'{predicted_class}: {predicted_probability:.2f}%'
            )

            # Удаляем временный файл
            os.remove(temp_image_path)

            # Подготовка результатов
            

            return render(request, 'result_forskin.html', {
                'predicted_class': predicted_class,
                'predicted_probability': f"{predicted_probability:.2f}%",
                'analysis_id':analysis.id,
                
                'gradcam_url': gradcam_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
            })

        except Exception as e:
            return render(request, 'skin.html', {'error': f"Ошибка анализа: {str(e)}"})

    return render(request, 'skin.html')
CLASSES = [
    "Melanoma", "Nevus", "Basal Cell Carcinoma",
    "Actinic Keratosis", "Benign Keratosis",
    "Dermatofibroma", "Vascular Lesion"
]
from django.contrib import messages

def save_results_skin(request):
    """Подтверждение сохранения анализа."""
    if request.method == 'POST':
        analysis_id = request.POST.get('analysis_id')
        try:
            # Получаем анализ по ID
            analysis = AnalysisSkin.objects.get(id=analysis_id, user=request.user)
            analysis.is_saved = True  # Предположительно, добавлено поле is_saved
            analysis.save()
            messages.success(request, 'Результаты успешно сохранены!')
        except AnalysisSkin.DoesNotExist:
            messages.error(request, 'Анализ не найден.')

        return redirect('user_kab')  # Перенаправление на кабинет пользователя
    return JsonResponse({'error': 'Недопустимый метод запроса'}, status=405)

# Список классов кожных заболеваний



