from rlscore.learner import KronRLS
from rlscore.measure import cindex
from rlscore.measure import spearman
from rlscore.measure import sqerror
from rlscore.measure import accuracy
import data_load_validation_settings1_mouse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import csv
from sklearn.metrics import f1_score
import collections
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def main():
    output_file = open(
        "/MyData_mouse/Results/SettingA/output_SetA.txt",'w')


    ## Zheng data (mouse only), 327 miRNA X 32709 mbs
    mouse_data = pd.read_csv(
        "/MyData_mouse/Interaction_data/mouse_data/mouse_data.csv",sep=",", header=0)

    miRNA_ids = mouse_data.iloc[:, 12]
    miRNA_ids = miRNA_ids[miRNA_ids.notna()]

    ## get mbs data
    mbs_mouse_seqLenAll = mouse_data.iloc[:, 13]  # all mbs, # 2675
    mbs_mouse_seqLenAll = mbs_mouse_seqLenAll[mbs_mouse_seqLenAll.notna()]

    mbs_mouse_seqLen10 = mouse_data.iloc[:, 14]  # mbs with seq length >= 10, #2592
    mbs_mouse_seqLen10 = mbs_mouse_seqLen10[mbs_mouse_seqLen10.notna()]

    mbs_mouse_seqLen20 = mouse_data.iloc[:, 15]  # mbs with seq length >= 20, #2506
    mbs_mouse_seqLen20 = mbs_mouse_seqLen20[mbs_mouse_seqLen20.notna()]

    mbs_mouse_seqLen30 = mouse_data.iloc[:, 16]  # mbs with seq length >= 32, #1200
    mbs_mouse_seqLen30 = mbs_mouse_seqLen30[mbs_mouse_seqLen30.notna()]

    mbs_mouse_seqLen40 = mouse_data.iloc[:, 17]  # mbs with seq length >= 40, #272
    mbs_mouse_seqLen40 = mbs_mouse_seqLen40[mbs_mouse_seqLen40.notna()]

    mbs_mouse_seqLen50 = mouse_data.iloc[:, 18]  # mbs with seq length >= 50, #33
    mbs_mouse_seqLen50 = mbs_mouse_seqLen50[mbs_mouse_seqLen50.notna()]

    mbs_ids = mbs_mouse_seqLen30  # select here which seq Length of mbs is tested

    #################################################################

    auc_list = []
    Fscore_list = []
    for i in range(1):

        print("############### START #############")
        print("Counter # ", i)
        ## selecting random miRNAs
        print(" Total mouse miRNA_ids", len(miRNA_ids))
        #miRNA_ids = miRNA_ids.sample(n=327)  # select random no of mbs chosen, total miRNA = 327
        random_miRNA_list = list(miRNA_ids)
        print("selected mouse miRNA= ", len(random_miRNA_list), random_miRNA_list[0:10])

        ## selecting random mbs
        print(" Total mouse mbs_ids", len(mbs_ids))
        #mbs_ids = mbs_ids.sample(n=1200)  # select random no of mbs chosen, total mbs is based on SeqLen, see above
        random_mbs_list = list(mbs_ids)

        print("selected mouse mRNA = ", len(random_mbs_list), random_mbs_list[0:10])

        ## performing model prediction
        X1, X2, Y = data_load_validation_settings1_mouse.load_data(random_miRNA_list, random_mbs_list)

        print("XD dimensions %d %d" % X1.shape)
        print("XT dimensions %d %d" % X2.shape)
        print("miRNA-mbs pairs: %d" % (Y.shape[0] * Y.shape[1]))
        Y = Y.ravel(order='F')
        learner = KronRLS(X1=X1, X2=X2, Y=Y)

        log_regparams = range(35, 36)  # this parameter should be adjusted for best predictions, # 35 is optimal
        for log_regparam in log_regparams:
            learner.solve(2. ** log_regparam)
            P = learner.in_sample_loo()

            Y1 = np.array(Y)
            Y1 = Y1 > 50  # this is just to replace 100 with 1, in the adjacancy matrix
            Y1 = Y1.astype(int)
            Y1 = list(Y1)

            fpr1, tpr1, _ = roc_curve(Y1, P)
            auc_score = roc_auc_score(Y1, P)
            print(' AUC (LOOCV): %.3f' % round(auc_score, 2))

            # F1-score and other parameter calculation
            for thres in [0.5]:  # 0.5 is selected as best
                P_01 = P > thres
                P_01 = P_01.astype(int)

                # Accuracy, Sensitivity, Specificity, F score from confusion matrix
                cm1 = confusion_matrix(Y1, P_01)
                # print('Confusion Matrix : \n', cm1)
                total1 = sum(sum(cm1))

                accuracy = (cm1[0, 0] + cm1[1, 1]) / total1
                # print('Accuracy : ', accuracy1)
                sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
                # print('Sensitivity : ', sensitivity1)
                specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
                # print('Specificity : ', specificity1)
                F_score = f1_score(Y1, P_01, average='micro')
                # print("F_score : ", F_score)

            auc_list.append(auc_score)
            Fscore_list.append(F_score)
            print("-------log_regparam=", log_regparam, "--------")
            print("accuracy", "sensitivity", "specificity", "F_score", "auc_score", accuracy, sensitivity, specificity,
                  F_score, auc_score)

    print("auc_list over different data randomization =", auc_list)
    print("Fscore_list over different data randomization =", Fscore_list)
    Avg_AUC = (sum(auc_list) / len(auc_list))
    Avg_Fscore = (sum(Fscore_list) / len(Fscore_list))
    s1 = (str(log_regparam) + "," + str(Avg_AUC) + "," + str(Avg_Fscore))
    output_file.write(s1 + "\n")
    output_file.close()

if __name__=="__main__":
    main()
