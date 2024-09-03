import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_from_csv(csv_path):
    """
    Função para plotar gráfico de Train Loss e Validation Loss a partir de um arquivo CSV.
    
    Args:
        csv_path (str): Caminho para o arquivo CSV contendo as métricas de treinamento.
    """
    # Carregar o CSV
    df = pd.read_csv(csv_path)

    # Plotando as métricas
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Uso da função
csv_path = f'D:\\Redes_Neurais\\U-Net-pytorch\\checkpoints\\training_metrics.csv'  # Certifique-se de que o caminho esteja correto
plot_loss_from_csv(csv_path)