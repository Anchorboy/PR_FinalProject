import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier

def read_data():
    img_size = (200, 200)

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
            nor_res = nor_res.reshape(nor_res.shape[0] * nor_res.shape[1] * nor_res.shape[2], )
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
            nor_res = nor_res.reshape(nor_res.shape[0] * nor_res.shape[1] * nor_res.shape[2], )
            test_all.append((class_id, nor_res))

    return train_all, test_all

def draw_multi_ROC(y_test, prob, paras):
    y_test_bin = label_binarize(y_test, classes=[(i+1) for i in xrange(n_classes)])

    # Compute macro-average ROC curve and ROC area
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_bin.ravel(), prob.ravel())
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
    plt.figure(str(paras[1]), figsize=(20,10))
    plt.title("Score: " + str(paras[0]) + " knn_k: " + str(paras[1]) + " pca_k: " + str(paras[2]))
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
    if not os.path.exists("task1"):
        os.mkdir("task1")
    if not os.path.exists("task1/color/"+str(paras[2])):
        os.mkdir("task1/color/"+str(paras[2]))
    plt.savefig("task1/color/"+str(paras[2])+'/'+str(paras[1])+'.png')
    # plt.show()
    plt.close()

def draw_ROC(y_test, prob):
    y_test_bin = label_binarize(y_test, classes=[(i + 1) for i in xrange(n_classes)])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_bin.ravel(), prob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    train_all, test_all = read_data()
    X_train = [ i[1] for i in train_all ]
    y_train = [ i[0] for i in train_all ]

    X_test = [ i[1] for i in test_all ]
    y_test = [ i[0] for i in test_all ]

    n_components = 1
    n_classes = len(set(y_train))
    while n_components < 201.1:
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_pca_train = pca.transform(X_train)
        X_pca_test = pca.transform(X_test)

        print "pca:", n_components
        expl = pca.explained_variance_ratio_
        sim = 0.0
        for i in expl:
            sim += i
        print "expl ratio:", sim

        # X_mat = np.mat(X_train)
        # print 1 - np.sum(np.square(X_mat - X_recompose_train)) / np.sum(np.square(X_mat)), "%"

        k = 1
        while k < 6:
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(X_pca_train, y_train)

            # predict = neigh.predict(X_pca_test)
            score = neigh.score(X_pca_test, y_test)
            prob = neigh.predict_proba(X_pca_test)

            print "score:", "%.4f" % score, "knn_k:", k, "pca_k:", n_components
            # input predict prob as ROC parameter
            draw_multi_ROC(y_test, prob, (score, k, n_components))

            k += 1

        n_components += 50