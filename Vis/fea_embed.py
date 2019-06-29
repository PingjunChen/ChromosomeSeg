# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pylab as plt
from sklearn.manifold import TSNE
import json, pickle

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


if __name__ == "__main__":
    test_fea_path = "chromosome_test_feas.json"
    test_feas = load_json_data(test_fea_path)
    feas, labels = generate_data_label(test_feas)

    y_data = labels
    num_label = len(np.unique(y_data))

    print("tsne...")
    # # # original space
    # ori_embed_feas = TSNE(n_components=2).fit_transform(feas)
    # ori_vis_x = ori_embed_feas[:, 0]
    # ori_vis_y = ori_embed_feas[:, 1]
    # project space
    lda_paras = pickle.load(open("lda_model.pkl", "rb"))
    prj_feas = np.matmul(feas, lda_paras['ProjectMat'])
    prj_embed_feas = TSNE(n_components=2).fit_transform(prj_feas)
    prj_vis_x = prj_embed_feas[:, 0]
    prj_vis_y = prj_embed_feas[:, 1]

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=300)
    # ori = ax1.scatter(ori_vis_x, ori_vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", num_label))
    # plt.colorbar(ori, ax=ax1)
    # ax1.set_title("Original chromosome feature embedding", fontsize=8)
    # prj = ax2.scatter(prj_vis_x, prj_vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", num_label))
    # plt.colorbar(prj, ax=ax2)
    # ax2.set_title("LDA projected feature embedding", fontsize=8)

    fig, axes = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    prj = axes.scatter(prj_vis_x, prj_vis_y, s=3, c=y_data, cmap=plt.cm.get_cmap("jet", num_label))
    plt.colorbar(prj, ax=axes)
    axes.set_title("LDA projected feature embedding", fontsize=8)

    plt.show()
