# 模式識別大作業

Environment: Python 2.7.10+

將train, test文件夾置於與腳本同樣的位置即可。

格式範例
	train	/0	/01.jpg
				/02.jpg
				...
			/1	/01.jpg
				...

	test	/0	/01.jpg
				/02.jpg
				...
			/1	/01.jpg
				...

# task1

Dependency: opencv3(cv2), matplotlib, scipy, sklearn, numpy, skimage

實現PCA + KNN，並將結果繪製成ROC曲線存放在task1文件夾、並用matplotlib顯示。

Entry: task1.py

Input: train, test文件夾
Output: ROC曲線以及繪製結果儲存
	./task1

# task2

Dependency: opencv3(cv2), matplotlib, scipy, sklearn, numpy, skimage

實現HOG + SVM，並將結果繪製成ROC曲線存放在task3文件夾、並用matplotlib顯示。

Entry: task2.py

Input: train, test文件夾
Output: ROC曲線以及繪製結果儲存 
	./task2

實現

# task3

Dependency: opencv3(cv2), matplotlib, scipy, sklearn, numpy, skimage

實現PCA + SVM，並將結果繪製成ROC曲線存放在task2文件夾、並用matplotlib顯示。

Entry: task3.py

Input: train, test文件夾
Output: ROC曲線以及繪製結果儲存
	./task3

# GoogLeNet

Dependency: opencv3(cv2), numpy, sklearn, keras, scipy

實現GoogLeNet並利用ImageNet的權重作為預訓練模型，加快模型收斂。

分為兩個部分：模型訓練、模型預測。

模型訓練
Entry: inception_v3.py

Input: train, test文件夾
Output: model_weights=("flower_classify.h5"), model_json=("folwer_classify.json")

模型預測
需先經過模型訓練得到權重、模型後才得以使用。
Entry: predict_v2.py

Input: train, test文件夾、model_weights=("flower_classify.h5"), model_json=("folwer_classify.json")
Output: Accuracy of test data
