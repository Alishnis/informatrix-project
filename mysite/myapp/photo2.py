import torch
from torchvision.models import densenet121
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.transform import resize
from matplotlib import pyplot as plt

# Загрузка модели
def load_chexnet():
    model = densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 14)  # 14 классов CheXNet
    return model

model = load_chexnet()
model.eval()

# Метки классов для CheXNet
chexnet_classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
]

# Порог для определения болезни
thresholds = {
    "Atelectasis": 0.6, "Cardiomegaly": 0.7, "Effusion": 0.65, "Infiltration": 0.6,
    "Mass": 0.7, "Nodule": 0.65, "Pneumonia": 0.6, "Pneumothorax": 0.65,
    "Consolidation": 0.65, "Edema": 0.6, "Emphysema": 0.6, "Fibrosis": 0.60,
    "Pleural Thickening": 0.7, "Hernia": 0.6
}

# Настройка Grad-CAM
def visualize_gradcam(input_tensor, image):
    target_layers = [model.features[-1]]  # Только последний слой  # Несколько слоёв
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    # Генерация тепловой карты
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = np.mean(grayscale_cam, axis=0)  # Усреднение активаций

    # Приведение размеров тепловой карты к размеру изображения
    grayscale_cam_resized = resize(grayscale_cam, (image.size[1], image.size[0]))

    # Наложение тепловой карты
    img_array = np.array(image) / 255.0
    visualization = show_cam_on_image(img_array, grayscale_cam_resized)

    # Отображение результата
    plt.imshow(visualization)
    plt.axis('off')
    plt.title("Grad-CAM Visualization")
    plt.show()

# Функция для анализа предсказаний
def analyze_predictions(input_tensor):
    # Выполнение предсказания
    with torch.no_grad():
        output = model(input_tensor)
    predicted_probs = torch.sigmoid(output).detach().numpy()[0]

    # Интерпретация предсказаний
    detected_diseases = []
    print("Detected diseases with probabilities:")
    for i, prob in enumerate(predicted_probs):
        if prob > thresholds[chexnet_classes[i]]:
            detected_diseases.append((chexnet_classes[i], prob * 100))
            print(f"- {chexnet_classes[i]}: {prob * 100:.2f}%")
    return detected_diseases

# Основной блок
if __name__ == "__main__":
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_path = "chest-xray3.png"
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    # Анализ предсказаний
    detected_diseases = analyze_predictions(input_tensor)

    # Визуализация Grad-CAM
    visualize_gradcam(input_tensor, image)