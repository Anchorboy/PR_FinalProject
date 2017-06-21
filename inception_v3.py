import os
import cv2
import numpy as np
from sklearn.preprocessing import label_binarize
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# this could also be the output a different Keras model or layer
def read_data():
    img_size = (224, 224)

    train_all = []
    test_all = []
    current_base = os.path.abspath('.')
    train_path = os.path.join(current_base, "train")
    test_path = os.path.join(current_base, "test")

    # read train
    for dir_name in os.listdir(train_path):
        dir_path = os.path.join(train_path, dir_name)
        class_id = int(dir_name)

        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img_vec = cv2.imread(img_path, flags=1)
            # print img_vec.shape
            # res = cv2.resize(img_vec, (int(img_vec.shape[0]*0.5), int(img_vec.shape[1]*0.5)), interpolation=cv2.INTER_CUBIC)
            res = cv2.resize(img_vec, img_size, interpolation=cv2.INTER_CUBIC)
            nor_res = np.zeros_like(res)
            nor_res = cv2.normalize(src=res, dst=nor_res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # print nor_res.shape
            train_all.append((class_id, nor_res))

    # read test
    for dir_name in os.listdir(test_path):
        dir_path = os.path.join(test_path, dir_name)
        class_id = int(dir_name)

        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img_vec = cv2.imread(img_path, flags=1)
            # print img_vec.shape
            # res = cv2.resize(img_vec, (int(img_vec.shape[0]*0.5), int(img_vec.shape[1]*0.5)), interpolation=cv2.INTER_CUBIC)
            res = cv2.resize(img_vec, img_size, interpolation=cv2.INTER_CUBIC)
            nor_res = np.zeros_like(res)
            nor_res = cv2.normalize(src=res, dst=nor_res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            test_all.append((class_id, nor_res))

    return train_all, test_all

if __name__ == "__main__":
    input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

    train_all, test_all = read_data()
    X_train = np.array([i[1] for i in train_all])
    # print X_train
    # print X_train.shape
    y_train = np.array([i[0] for i in train_all])
    y_train_bin = label_binarize(y_train, classes=[(i + 1) for i in xrange(10)])

    X_test = np.array([i[1] for i in test_all])
    y_test = np.array([i[0] for i in test_all])
    y_test_bin = label_binarize(y_test, classes=[(i + 1) for i in xrange(10)])

    # create the base pre-trained model
    base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # train the model on the new data for a few epochs
    hist = model.fit(X_train, y_train_bin, batch_size=30, epochs=50, validation_data=(X_test, y_test_bin))
    print hist.history

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    hist = model.fit(X_train, y_train_bin, batch_size=30, epochs=50, validation_data=(X_test, y_test_bin))
    print hist.history

    model.save_weights("flower_classify.h5")
    json_string = model.to_json()
    open("folwer_classify.json", 'w').write(json_string)
