import matplotlib.pyplot as plt
import time  # Importar módulo para medir o tempo de execução
import os
import random
import numpy as np
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from dotenv import load_dotenv
from model import UNet

# Carregar variáveis do arquivo .env
load_dotenv()

# ==============================
# Carregar Parâmetros do .env
# ==============================

IMAGES_DIR = os.getenv('IMAGES_DIR')  # Diretório de imagens para teste
MASKS_DIR = os.getenv('MASKS_DIR')  # Diretório de máscaras (não usado no teste, mas pode ser necessário para outros propósitos)
CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR')  # Diretório onde os checkpoints são armazenados

NUM_CLASSES = int(os.getenv('NUM_CLASSES'))
IN_CHANNELS = int(os.getenv('IN_CHANNELS'))

IMAGE_SIZE = (int(os.getenv('IMAGE_SIZE_HEIGHT')), int(os.getenv('IMAGE_SIZE_WIDTH')))

BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))

TRAIN_RATIO = float(os.getenv('TRAIN_RATIO'))
VAL_RATIO = float(os.getenv('VAL_RATIO'))
TEST_RATIO = float(os.getenv('TEST_RATIO'))

# ==============================
# Verificar Disponibilidade da GPU
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # Redimensiona a imagem para o tamanho especificado
    transforms.ToTensor(),          # Converte a imagem para tensor PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza imagens RGB
])

# ==============================
# Função para Testar o Modelo em Imagens Aleatórias
# ==============================

def test_random_images(model, images_dir, transform, device, num_images=3):
    """
    Testa o modelo em imagens aleatórias e plota os resultados.

    Args:
        model (nn.Module): Modelo treinado.
        images_dir (str): Diretório contendo imagens de teste.
        transform (callable): Transformação a ser aplicada às imagens.
        device (torch.device): Dispositivo (CPU ou GPU) onde o modelo está sendo executado.
        num_images (int): Número de imagens aleatórias para testar.
    """
    # Listar todos os arquivos de imagem no diretório de teste
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    # Selecionar um número especificado de imagens aleatórias
    random_images = random.sample(image_files, num_images)

    # Configurar a tela para exibir todas as imagens e suas máscaras previstas
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(12, 6 * num_images))

    for idx, image_file in enumerate(random_images):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)  # Adiciona uma dimensão para o batch e move para o dispositivo

        model.eval()  # Configura o modelo para o modo de avaliação
        with torch.no_grad():  # Desativa o cálculo do gradiente para economizar memória
            start_time = time.time()  # Inicia a contagem de tempo

            # Fazer a previsão
            output = model(input_image)
            predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()  # Aplica sigmoid e converte para numpy

            end_time = time.time()  # Termina a contagem de tempo

        # Calcular o tempo de execução
        execution_time = end_time - start_time
        print(f"Tempo de execução da inferência para {image_file}: {execution_time:.4f} segundos")

        # Plotar a imagem original e a máscara prevista na grade de subplots
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Imagem Original: {image_file}')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(predicted_mask, cmap='gray')
        axes[idx, 1].set_title('Máscara Prevista')
        axes[idx, 1].axis('off')

    # Ajustar o layout para evitar sobreposição de títulos e garantir que tudo seja exibido corretamente
    plt.tight_layout()
    plt.show()

# ==============================
# Execução Principal
# ==============================

if __name__ == "__main__":
    # Caminho para o diretório de teste
    test_images_dir = IMAGES_DIR  # Diretório definido no .env ou fornecido manualmente

    # Defina o caminho para o modelo salvo (checkpoint)
    model_path = 'D:\\Redes_Neurais\\U-Net-pytorch\\checkpoints\\epoch_6\\checkpoint.pth.tar'  # Defina o caminho para o checkpoint desejado

    # Verifica se o caminho do modelo é válido
    if not os.path.isfile(model_path):
        raise ValueError(f"Checkpoint não encontrado no caminho fornecido: {model_path}")

    # Carregar o modelo salvo
    model = UNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(device)

    # Carregar o checkpoint corretamente
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])  # Extrai o state_dict do checkpoint

    # Testar o modelo em imagens aleatórias e calcular o tempo de execução
    test_random_images(model, test_images_dir, transform, device, num_images=2)
