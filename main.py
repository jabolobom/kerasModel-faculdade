import tensorflow as tf
from tensorflow.keras import layers, models

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/train',
    labels='inferred', # label a partir do nome das pastas
    label_mode='int', # labels inteiros (numericos)
    color_mode='grayscale',
    image_size=(28, 28),
    batch_size=128, # tamanho dos lotes de treino
    shuffle=True # embaralha os dados pro modelo não aprender sequencias
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/test',
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    image_size=(28, 28),
    batch_size=128,
    shuffle=False # modelo de teste não precisa embaralhar
)

# NORMALIZANDO VALORES, transformando os valores de pixel para valores entre 0 e 1
# ajuda na precisão diminuindo o range de possibilidades
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# criacao do modelo
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)), # input de imagens 28x28 grayscale
    layers.Conv2D(32, 3, activation='relu'), # filtro
    layers.MaxPooling2D(), # reduz a resolução
    layers.Conv2D(64, 3, activation='relu'), 
    layers.MaxPooling2D(), # novamente, filtro e resolução
    layers.Flatten(), # Mathmagics, transforma o output 2D em 1D
    layers.Dense(128, activation='relu'), 
    layers.Dense(62, activation='softmax') # 62 "neuronios" ativam quando reconhecem algo que stava no dataset (numeros, maiusculas e minusculas)
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# TREINAR O MODELO
model.fit(train_ds, validation_data=test_ds, epochs=35) # 35 Epochs foi um bom valor que encontrei
# 10 epochs accuracy ficaria mais ou menos em .7
# 15+ começa a melhorar pra .8~
# 35 fica nos .9~

model.save('kaggle_letras.keras') # salva o modelo treinado pra rodar depois
