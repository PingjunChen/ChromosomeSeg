# -*- coding: utf-8 -*-

import os, sys
import json
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def load_json_data(json_path):
	fea_dict = json.load(open(json_path))

	fea_category_dict = {}
	for key in fea_dict.keys():
		cat = key[:key.find('_')]
		if cat not in fea_category_dict:
			fea_category_dict[cat] = [fea_dict[key]]
		else:
			fea_category_dict[cat].append(fea_dict[key])

	return fea_category_dict


def generate_data_label(feas):
	Data = []
	Label = []
	for iter in range(0, 24):
		if iter<22:
			tn = len(feas[str(iter+1)])
			for jter in range(tn):
				temp = feas[str(iter+1)][jter]
				Data.append(np.array(temp))
				Label.append([iter])

		elif iter==22:
			tn = len(feas['X'])
			for jter in range(tn):
				temp = feas['X'][jter]
				Data.append(np.array(temp))
				Label.append([iter])
		else:
			tn = len(feas['Y'])
			for jter in range(tn):
				temp = feas['Y'][jter]
				Data.append(np.array(temp))
				Label.append([iter])

	Data = np.array(Data)
	Label = np.squeeze(Label)

	return Data, Label


def data_normalize(trainData, testData):
	trainData = normalize(trainData, norm='max')
	testData = normalize(testData, norm='max')

	mean_data = np.mean(trainData, axis=0, keepdims=True)
	trainData = trainData - np.matmul(np.ones((trainData.shape[0],1)), mean_data)
	testData = testData - np.matmul(np.ones((testData.shape[0],1)), mean_data)

	return trainData, testData


def construct_graph(Label):
	tn = Label.shape[0]
	uLabel = np.unique(Label)
	W = np.zeros((tn,tn))

	for iter in range(len(uLabel)):
		index = np.squeeze(np.where(Label==uLabel[iter]))

		W[np.ix_(index, index)]=1.0/index.size

	return W


def LDA(trainData, trainLabel):
	class_mean = []
	uLabel = np.unique(trainLabel)
	class_num = len(uLabel)
	W = construct_graph(trainLabel)

	Sw = np.matmul(np.matmul(np.transpose(trainData), W), trainData)

	Sb = np.matmul(np.transpose(trainData),trainData)


	U,D,_ = np.linalg.svd(Sb+0.0001*np.eye(Sb.shape[0]),full_matrices=True)

	D = np.diag(np.sqrt(D))

	uD = np.matmul(U, np.linalg.inv(D))

	S = np.matmul(np.matmul(np.matmul(np.linalg.inv(D),np.transpose(U)), Sw), uD)

	Y,_,_= np.linalg.svd(S)

	A = np.matmul(uD, Y)

	trainVector = np.matmul(trainData, A)

	for iter in range(len(uLabel)):
		index = np.squeeze(np.where(trainLabel==uLabel[iter]))
		temp = np.mean(trainVector[index,:], axis=0, keepdims=True)
		class_mean.extend(temp)                     # saved based on the labels

	return A, np.array(class_mean)



def Classify(testData, testLabel, A, class_mean):
	testVector = np.matmul(testData, A)
	ten = testVector.shape[0]
	class_num =class_mean.shape[0]
	dist = []
	for iter in range(ten):
		temp = np.sum((np.matmul(np.ones((class_num,1)), np.expand_dims(testVector[iter,:], axis=0)) - class_mean)**2, axis=1)
		dist.append(temp)
	dist = np.array(dist)

	predict = np.squeeze(np.argmin(dist, axis=1))
	accuracy = accuracy_score(testLabel, predict)
	CM = confusion_matrix(testLabel, predict)

	return dist, accuracy, CM



def lda_pred(fea, project_mat, class_mean):
	testVector = np.matmul(fea, project_mat)
	class_num = class_mean.shape[0]
	dist = np.sum((np.matmul(np.ones((class_num,1)), np.expand_dims(testVector, axis=0)) - class_mean)**2, axis=1)

	label, min_dist = np.argmin(dist), min(dist)

	return label, min_dist


if __name__ == "__main__":
	train_fea_path = "./data/OverlapSep/chromosome_train_feas.json"
	test_fea_path = "./data/OverlapSep/chromosome_test_feas.json"

	# Both train_feas and test_feas are dictionary with 24 keys, '1', '2',...,'X','Y'
	# For each key, there are a list of feature descriptor for chromosome
	# Each feature descriptor is also a list with 25 elements
	train_feas = load_json_data(train_fea_path)
	test_feas = load_json_data(test_fea_path)
	# Organize data
	trainData, trainLabel = generate_data_label(train_feas)
	testData, testLabel = generate_data_label(test_feas)

	#  LDA classification
	# trainData, testData = data_normalize(trainData, testData)
	project_mat, class_mean= LDA(trainData, trainLabel) # A is the projection matrix

	# # save lda model
	# lda_model_dict = {
	# 	'ProjectMat': project_mat,
	# 	'ClassMean': class_mean}
	#
	# with open('lda_model.pkl', "wb") as handle:
	# 	pickle.dump(lda_model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# # sample-by-sample prediction
	# correct_num, total_num = 0, testData.shape[0]
	# for it in np.arange(total_num):
	# 	label, _ = lda_pred(testData[it], project_mat, class_mean)
	# 	if label == testLabel[it]:
	# 		correct_num += 1
	# test_acc = correct_num * 1.0 / total_num
	# print("Testing accuracy is: {:.3f}".format(test_acc))


	# prediction and
	dist, lda_accuracy, CM = Classify(testData, testLabel, project_mat, class_mean)   # dist is with the size test number * class_num
	print("The lda accuracy is:{}".format(lda_accuracy))

	# plt.imshow(CM, interpolation='nearest', cmap=plt.cm.Blues)
	# plt.title("Chromosome Contour Prediction Confusion Matrix")
	# plt.xlabel('Predicted label')
	# plt.ylabel('True label')
	# labels = np.arange(24)
	# tick_marks = np.arange(len(labels))
	# plt.xticks(tick_marks, labels, rotation=90)
	# plt.yticks(tick_marks, labels)
	# plt.tight_layout()
	# plt.show()
