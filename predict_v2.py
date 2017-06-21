import os
import random
from itertools import cycle

import cv2
import matplotlib as mpt
import numpy as np
mpt.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from keras.models import load_model
from keras.models import model_from_json
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D


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

def draw_multi_ROC(y_test_bin, decf, score):
    n_classes = 10
    # y_test_bin = label_binarize(y_test, classes=[(i+1) for i in xrange(n_classes)])

    # Compute macro-average ROC curve and ROC area
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], decf[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_bin.ravel(), decf.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    # print mean_tpr
    for i in range(n_classes):
        # print interp(all_fpr, fpr[i], tpr[i])
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(20,10))
    plt.title("Score: " + str(score))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    lw = 2

    colors = cycle([ ((round(random.uniform(0.0, 1.0), 2)), (round(random.uniform(0.0, 1.0), 2)), (round(random.uniform(0.0, 1.0), 2)))
                     for i in xrange(n_classes) ])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc='lower right')
    plt.savefig("task3/"+'googlenet.png')
    plt.show()
    # plt.close()

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
    model = model_from_json(open("flower_classify.json").read())
    model.load_weights("flower_classify.h5", by_name=True)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = model.evaluate(X_test, y_test_bin, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    # for ind, x in enumerate(X_test):
    #     predict = model.predict(x, verbose=1)
    #     print "predict:", np.argmax(predict), "true:", np.argmax(y_test_bin[ind])

