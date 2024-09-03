
## Dataset utilizando para teste do modelo

https://www.kaggle.com/c/carvana-image-masking-challenge


# U-Net para Segmentação de Imagens com PyTorch

Este projeto implementa uma rede neural U-Net para segmentação de imagens utilizando PyTorch. A U-Net é uma arquitetura popular para segmentação de imagens, especialmente usada em imagens médicas, mas também pode ser aplicada a outros tipos de imagens.

## Índice

1. [Descrição do Projeto](#descrição-do-projeto)
2. [Pré-requisitos](#pré-requisitos)
3. [Configuração](#configuração)
4. [Variáveis de Entrada](#variáveis-de-entrada)
5. [Execução](#execução)
6. [Treinamento](#treinamento)
7. [Avaliação](#avaliação)
8. [Estrutura de Pastas](#estrutura-de-pastas)
9. [Contribuições](#contribuições)
10. [Licença](#licença)

## Descrição do Projeto

Este projeto utiliza a arquitetura U-Net para segmentação de imagens. A U-Net é composta por um encoder (para capturar o contexto) e um decoder (para segmentação precisa), permitindo o aprendizado eficiente de mapeamentos complexos de imagem para imagem.

### Características Principais:

- **Modelo U-Net Personalizado:** Configurável para segmentação de uma ou várias classes.
- **Pipeline de Treinamento Completo:** Inclui carregamento de dados, treinamento, validação e checkpoints automáticos.
- **Métricas de Avaliação:** Calcula métricas como Dice Coefficient e IoU para avaliar a performance do modelo.

## Pré-requisitos

- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- Pillow (PIL)
- matplotlib
- python-dotenv

## Configuração

1. **Clonar o Repositório:**

   ```bash
   git clone https://github.com/seu-usuario/unet-segmentacao.git
   cd unet-segmentacao
