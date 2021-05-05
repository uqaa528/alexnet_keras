import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential, model_from_json, save_model
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization, LayerNormalization

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)

config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

(train_images, train_labels), (test_images, test_labels) = cifar100.load_data(label_mode='fine')

superclass_labels = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
              ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
              ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
              ['bottle', 'bowl', 'can', 'cup', 'plate'],
              ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
              ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
              ['bed', 'chair', 'couch', 'table', 'wardrobe'],
              ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
              ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
              ['bridge', 'castle', 'house', 'road', 'skyscraper'],
              ['cloud', 'forest', 'mountain', 'plain', 'sea'],
              ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
              ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
              ['crab', 'lobster', 'snail', 'spider', 'worm'],
              ['baby', 'boy', 'girl', 'man', 'woman'],
              ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
              ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
              ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
              ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
              ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
superclass_labels = np.sort(np.concatenate(superclass_labels, axis=0))

alexnet = Sequential()

# conv 1
alexnet.add(Conv2D(filters=96, kernel_size=(11, 11), input_shape=(32, 32, 3), padding='same', strides=(4,4), activation='relu'))
alexnet.add(LayerNormalization())
# maxpool 1
alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))
# conv 2
alexnet.add(ZeroPadding2D((2, 2)))
alexnet.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', strides=(1,1), activation='relu'))
alexnet.add(BatchNormalization())
# maxpool 2
alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))
# conv 3
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
alexnet.add(LayerNormalization())
# conv 4
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
alexnet.add(BatchNormalization())
# conv 5
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
alexnet.add(BatchNormalization())
# maxpool 5
alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))
# flattening
alexnet.add(Flatten())
# fc 6
alexnet.add(Dense(4096, activation='relu'))
alexnet.add(BatchNormalization())
alexnet.add(Dropout(0.5))
# fc 7
alexnet.add(Dense(4096,activation='relu'))
alexnet.add(BatchNormalization())
alexnet.add(Dropout(0.5))
# fc 8
alexnet.add(Dense(100))
alexnet.add(Activation('softmax'))

alexnet.build(input_shape=(32, 32, 3))
alexnet.summary()
alexnet_json = alexnet.to_json()

alexnet.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1,32,32,3)
test_images = test_images.reshape(-1,32,32,3)
train_labels = np_utils.to_categorical(train_labels, 100)
test_labels = np_utils.to_categorical(test_labels, 100)

history = alexnet.fit(train_images, train_labels, batch_size=25, epochs=10, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = alexnet.evaluate(test_images, test_labels, verbose=0)

with open("alexnet.json", "w") as json_file:
    json_file.write(alexnet_json)
alexnet.save_weights("alexnet.h5")

plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower left')
plt.show()