import numpy as np
import sys
import pandas as pd


from MyData_mouse.Interaction_data import miRNA_mRNA_pairwise_Intrctions
from MyData_mouse.miRNA_features import miRNA_pairwise_featuresCombined
from MyData_mouse.mRNA_features import mRNA_pairwise_featuresCombined

def load_data(selected_miRNA_list, selected_mRNA_list):
    Y = miRNA_mRNA_pairwise_Intrctions.Interactions(selected_miRNA_list, selected_mRNA_list)
    print ("imported interction data")

    XD = miRNA_pairwise_featuresCombined.combine_miRNA_features(selected_miRNA_list)
    print("imported miRNA pairwise data")

    XT = mRNA_pairwise_featuresCombined.combine_mRNA_features(selected_mRNA_list)
    print("imported mRNA pairwise data")
    return XD, XT, Y

def settingC_split(selected_miRNA_list, selected_mRNA_list):
    np.random.seed(1)

    mbs_test_cnt = 120 # select number of mbs for testing

    XD, XT, Y = load_data(selected_miRNA_list, selected_mRNA_list)
    drug_ind = list(range(Y.shape[0]))
    target_ind = list(range(Y.shape[1]))
    np.random.shuffle(target_ind)

    train_target_ind = target_ind[mbs_test_cnt:]  # mbs for training
    test_target_ind = target_ind[:mbs_test_cnt]  # remaining number of mbs for testing

    #train_target_ind = target_ind[:mbs_test_cnt]  # mbs for training
    #test_target_ind = target_ind[mbs_test_cnt:]  # remaining number of mbs for testing

    #Setting 3: split according to targets
    Y_train = Y[:, train_target_ind]
    Y_test = Y[:, test_target_ind]
    Y_train = Y_train.ravel(order='F')
    Y_test = Y_test.ravel(order='F')
    XD_train = XD
    XT_train = XT[train_target_ind]
    XD_test = XD
    XT_test = XT[test_target_ind]
    return XD_train, XT_train, Y_train, XD_test, XT_test, Y_test

