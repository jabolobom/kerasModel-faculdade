from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image

# PARA TESTAR, ESCOLHA UMA IMAGEM DA PASTA "IMAGENS" E MUDE O NOME DA CONSTANTE "IMG"

IMG_FOLDER = 'imagens/'
IMG = 'F_caps.002.png' # MUDE ESSA VARIAVEL COM O NOME

IMG_PATH = os.path.join(IMG_FOLDER, IMG)

# CARREGANDO MODELO
model = load_model('kaggle_letras.keras') # carrega o modelo 

# PROCESSANDO
def processar_imagem(img):
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0 # igual ao treinamento do modelo, transforma a imagem em
    # 28x28 grayscale
    return img

# ABRE A IMAGEM COMO UM ARRAY NO NUMPY
img_teste = Image.open(IMG_PATH).convert('L').resize((28, 28)) # L = GRAYSCALE
img_array = np.array(img_teste)

img_processada = processar_imagem(img_array)

predicao = model.predict(img_processada) # roda o modelo por cima e retorna um vetor de probabilidades
# de letras possíveis.

# lista de letras
label_map = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + \
            [chr(i) for i in range(ord('a'), ord('z') + 1)]

classepred = np.argmax(predicao, axis=1)[0] # o que tiver maior probabilidade é a classe previda
letter = label_map[classepred - 10] if classepred >= 10 else 'Numero' # se colocado um digito na imagem retorna Numero

print(f'Letra reconhecida {letter}')