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
from dotenv import load_dotenv  # Importar a biblioteca para carregar .env
from model import UNet

# Carregar variáveis do arquivo .env
load_dotenv()

# ==============================
# Definição da Seed para Reprodutibilidade
# ==============================

SEED = 42  # Escolha qualquer número inteiro

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==============================
# Carregar Parâmetros do .env
# ==============================

IMAGES_DIR = os.getenv('IMAGES_DIR')
MASKS_DIR = os.getenv('MASKS_DIR')
CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR')

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

# Definindo o caminho para o arquivo CSV dentro do diretório de checkpoints
CSV_PATH = os.path.join(CHECKPOINT_DIR, 'training_metrics.csv')

# ==============================
# Verificar Disponibilidade da GPU
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CustomSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, csv_path=None, transform=None):
        """
        Args:
            images_dir (str): Diretório contendo as imagens.
            masks_dir (str): Diretório contendo as máscaras em formato JPG.
            csv_path (str, opcional): Caminho para o arquivo CSV contendo as máscaras (se necessário).
            transform (callable, opcional): Transformação a ser aplicada às imagens e máscaras.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

        # Carrega o CSV se necessário e se existir
        self.masks_csv = None
        if csv_path and os.path.isfile(csv_path):
            self.masks_csv = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Carregar imagem
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        # Carregar máscara
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        mask = Image.open(mask_path).convert("L")  # "L" para escala de cinza que representa os rótulos de classe

        # Aplicar transformações
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize(IMAGE_SIZE)(mask)  # Redimensiona a máscara para o tamanho esperado
            mask = transforms.functional.to_tensor(mask).float()  # Converte para tensor e mantém máscara binária

        return image, mask

# ==============================
# Definição das Transformações
# ==============================

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # Redimensiona a imagem para o tamanho especificado
    transforms.ToTensor(),          # Converte a imagem para tensor PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza imagens RGB
])

# ==============================
# Preparação e Divisão do Dataset
# ==============================

# Carregar o dataset completo
dataset = CustomSegmentationDataset(
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    csv_path=None,  # Não usamos o CSV para carregar o dataset
    transform=transform
)

# Definir tamanhos para treino, validação e teste
train_size = int(TRAIN_RATIO * len(dataset))
val_size = int(VAL_RATIO * len(dataset))
test_size = len(dataset) - train_size - val_size

# Dividir o dataset em treino, validação e teste com seed para reprodutibilidade
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED))

# Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# ==============================
# Função para Calcular Métricas de Avaliação
# ==============================

def calculate_metrics(pred, target):
    """
    Calcula o Dice Coefficient e o IoU para os dados previstos e de alvo.

    Args:
        pred (torch.Tensor): Previsões do modelo (saída).
        target (torch.Tensor): Máscaras de alvo.

    Returns:
        dice (float): Coeficiente Dice.
        iou (float): Intersection over Union.
    """
    smooth = 1e-6  # Evita divisões por zero

    pred = (pred > 0.5).float()  # Converte para binário (0 ou 1)
    target = (target > 0.5).float()

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    iou = (intersection + smooth) / (pred.sum() + target.sum() - intersection + smooth)

    return dice.item(), iou.item()

# ==============================
# Função para Salvar o Modelo
# ==============================

def save_checkpoint(state, epoch):
    """
    Função para salvar o checkpoint do modelo para cada época em uma pasta separada.
    
    Args:
        state (dict): Estado atual do modelo e otimizador.
        epoch (int): Número da época atual.
    """
    # Criar um diretório base se não existir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Criar um diretório para cada época
    epoch_dir = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Salvar o modelo no diretório específico da época
    checkpoint_path = os.path.join(epoch_dir, "checkpoint.pth.tar")
    torch.save(state, checkpoint_path)

# ==============================
# Bloco Principal de Execução
# ==============================

if __name__ == '__main__':
    # Inicializando o modelo para segmentação binária
    model = UNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(device)

    # Definição da Função de Custo e Otimizador
    criterion = nn.BCEWithLogitsLoss()  # Função de custo para segmentação binária
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Inicializando listas para salvar métricas
    train_losses = []
    val_losses = []
    val_dice_scores = []
    val_iou_scores = []

    # Checkpoint: determinar a última época treinada a partir do CSV
    start_epoch = 0
    if os.path.isfile(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if not df.empty:
            start_epoch = df['Epoch'].iloc[-1]  # Define a última época registrada no CSV
            print(f"Continuando o treinamento a partir da época {start_epoch + 1} (aumente o número de épocas))")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{start_epoch + 1}", "checkpoint.pth.tar")
    if os.path.isfile(checkpoint_path):
        print(f"=> Carregando checkpoint de {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint carregado: começando na época {start_epoch + 1}")

    # Loop de Treinamento
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()  # Define o modelo para modo de treinamento
        running_loss = 0.0
        
        for images, masks in train_loader:
            # Mover dados para a GPU
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()  # Zera os gradientes

            outputs = model(images)  # Forward pass

            loss = criterion(outputs, masks)  # Calcula a perda

            loss.backward()  # Backward pass e cálculo dos gradientes
            optimizer.step()  # Atualiza os pesos do modelo

            running_loss += loss.item() * images.size(0)

        # Calcula a perda média de treinamento
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validação
        model.eval()  # Muda para modo de validação
        val_loss = 0.0
        dice_score = 0.0
        iou_score = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)

                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Calcular Dice e IoU para cada batch
                dice, iou = calculate_metrics(torch.sigmoid(outputs), masks)
                dice_score += dice * images.size(0)
                iou_score += iou * images.size(0)

        val_loss /= len(val_loader.dataset)
        dice_score /= len(val_loader.dataset)
        iou_score /= len(val_loader.dataset)

        val_losses.append(val_loss)
        val_dice_scores.append(dice_score)
        val_iou_scores.append(iou_score)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Dice Score: {dice_score:.4f}, IoU: {iou_score:.4f}")

        # Salvar modelo após cada época
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, epoch)

        # Atualizar o CSV de métricas
        if os.path.isfile(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
        else:
            df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Validation Loss', 'Dice Score', 'IoU'])

        new_row = {'Epoch': epoch + 1, 'Train Loss': train_loss, 'Validation Loss': val_loss, 'Dice Score': dice_score, 'IoU': iou_score}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

