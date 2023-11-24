from keras import models
from keras import layers
from keras import activations
from keras import initializers
from keras import regularizers
from keras import metrics
from keras import optimizers
from keras import losses
from keras import callbacks
from keras.preprocessing import image
from keras import saving
import numpy as np

model = models.Sequential()

model.add(layers.Conv2D(
    128,
    (5,5),
    input_shape=(128,128,3),
    activation='relu'
))

model.add(layers.MaxPooling2D(
    pool_size = (2, 2) 
))

model.add(layers.Conv2D(
    64,
    (5,5),
    input_shape=(62,62,3),
    activation='relu'
))

model.add(layers.MaxPooling2D(
    pool_size = (2, 2) 
))

model.add(layers.Conv2D(
    32,
    (5,5),
    input_shape=(29,29,3),
    activation='relu'
))

model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
))

model.add(layers.Conv2D(
    16,
    (5,5),
    input_shape=(13,13,3),
    activation='relu'
))

model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
))



model.add(layers.Flatten())

model.add(layers.Dense(
    256,
    # kernel_initializer= initializers.RandomNormal(stddev=1), 
    # bias_initializer= initializers.Zeros(),
    activation=activations.relu
))


model.add(layers.Dense(
    128,
    # kernel_initializer= initializers.RandomNormal(stddev=1), 
    # bias_initializer= initializers.Zeros(),
    activation=activations.relu
))


model.add(layers.Dense(
    2,
    # kernel_initializer= initializers.RandomNormal(stddev=1),
    # bias_initializer= initializers.Zeros(),
    activation=activations.softmax
))


model.compile(
  optimizer=optimizers.Adam(),
  loss= losses.CategoricalCrossentropy(),
  metrics = [metrics.CategoricalAccuracy()]
)

dataGen = image.ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.2
)


X_train = dataGen.flow_from_directory(
    "beauty/",
    target_size=(128,128),
    batch_size=10,
    class_mode='categorical',
    subset='training'
)

X_test = dataGen.flow_from_directory(
    "beauty/",
    target_size=(128,128),
    batch_size=10,
    class_mode='categorical',
    subset='validation'
)

model.fit(
    x = X_train,
    validation_data=X_test,
    steps_per_epoch=320,
    epochs=50,
    validation_steps=32,# 10 vezes menor que o steps_per_epoch
    callbacks=[
        callbacks.ModelCheckpoint(filepath='model/model.{epoch:02d}-{loss:.2f}.keras')
    ]
)

model.save("finalModel.keras")


def Predict(image):
    loadedModel = saving.load_model("finalModel.keras")
    test_image = image.load_img(image, target_size=(128,128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loadedModel.predict(test_image)
    print(result)

Predict("mulher.jpeg")