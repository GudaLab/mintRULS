from rlscore.learner import KronRLS
from rlscore.measure import cindex
from rlscore.measure import spearman
from rlscore.measure import sqerror
from rlscore.measure import accuracy
import data_load_validation_settings1
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


    # importing input data

    ## Zheng data (human only), 845 miRNA X 32709 mbs
    human_data = pd.read_csv(
        "/MyData/Interaction_data/human_data/human_data.csv",sep=",", header=0)

    miRNA_ids = human_data.iloc[:, 12]
    miRNA_ids = miRNA_ids[miRNA_ids.notna()]

    ## get mbs data
    mbs_human_seqLenAll = human_data.iloc[:, 13]  # all mbs, # 32709
    mbs_human_seqLenAll = mbs_human_seqLenAll[mbs_human_seqLenAll.notna()]

    mbs_human_seqLen10 = human_data.iloc[:, 14]  # mbs with seq length >= 10, #32138
    mbs_human_seqLen10 = mbs_human_seqLen10[mbs_human_seqLen10.notna()]

    mbs_human_seqLen20 = human_data.iloc[:, 15]  # mbs with seq length >= 20, #32551
    mbs_human_seqLen20 = mbs_human_seqLen20[mbs_human_seqLen20.notna()]

    mbs_human_seqLen30 = human_data.iloc[:, 16]  # mbs with seq length >= 32, #20893
    mbs_human_seqLen30 = mbs_human_seqLen30[mbs_human_seqLen30.notna()]

    mbs_human_seqLen40 = human_data.iloc[:, 17]  # mbs with seq length >= 40, #18036
    mbs_human_seqLen40 = mbs_human_seqLen40[mbs_human_seqLen40.notna()]

    mbs_human_seqLen50 = human_data.iloc[:, 18]  # mbs with seq length >= 50, #9598
    mbs_human_seqLen50 = mbs_human_seqLen50[mbs_human_seqLen50.notna()]

    mbs_ids = mbs_human_seqLen40  # select here which seq Length of mbs is tested

    #################################################################

    output_file = open(
        "/MyData/Results/SettingA/output_SetA.txt",'w')
    actual_pred =open(
        "/MyData/Results/SettingA/actual_pred.csv",'w')

    auc_list = []
    Fscore_list = []
    for i in range(1):

        print("############### START #############")
        print("Counter # ", i)
        ## selecting random miRNAs
        print(" Total human miRNA_ids", len(miRNA_ids))
        #miRNA_ids = miRNA_ids.sample(n=845)  # select random no of mbs chosen, total miRNA = 845, No need to randomize if all 845 miRNAs are being used
        random_miRNA_list = list(miRNA_ids)
        print("selected human miRNA= ", len(random_miRNA_list), random_miRNA_list[0:10])

        ## selecting random mbs
        print(" Total human mbs_ids", len(mbs_ids))
        mbs_ids = mbs_ids.sample(n=3000)  # select random no of mbs chosen, total mbs = ?
        random_mbs_list = list(mbs_ids)
        print("selected human mRNA = ", len(random_mbs_list), random_mbs_list[0:10])

        ## performing model prediction
        X1, X2, Y = data_load_validation_settings1.load_data(random_miRNA_list, random_mbs_list)

        print("XD dimensions %d %d" % X1.shape)
        print("XT dimensions %d %d" % X2.shape)
        print("miRNA-mbs pairs: %d" % (Y.shape[0] * Y.shape[1]))
        Y = Y.ravel(order='F')
        learner = KronRLS(X1=X1, X2=X2, Y=Y)
        s = str(i) + ","

        log_regparams = range(35, 36)  # this parameter should be adjusted for best predictions #37 is best
        for log_regparam in log_regparams:  # nummber of times random selection occur on main interaction data
            # MINT-RULS
            Y = Y.ravel(order='F')
            learner.solve(2. ** log_regparam)
            P = learner.in_sample_loo()

            # converting Y test data into 0-1
            Y1 = np.array(Y)
            Y1 = Y1 > 50  # this is just to replace 100 with 1, in the adjacancy matrix
            Y1 = Y1.astype(int)
            Y1 = list(Y1)

            # AUC calculate
            fpr1, tpr1, threshold = roc_curve(Y1, P)
            auc_score = roc_auc_score(Y1, P)
            print('AUC (LOOCV): %.3f' % auc_score)
            print('auto calculated Thres = ', threshold)

            # F1-score and other parameter calculation
            for thres in [0, 0.2, 0.3, 0.35, 0.4, 0.5, 0.7, 0.8, 1.5]:  # 0.3 is selected as best threshold
                P_01 = P > thres
                P_01 = P_01.astype(int)

                # saving actual and pred vals
                TP = 0
                FN = 0
                FP = 0
                TN = 0
                for i in range (len(Y1)):
                    if (Y1[i] == 1) and (P_01[i] == 1):
                        actual_pred.write(str(Y1[i]) + "," + str(P_01[i]) + "," + "TP" + "\n")
                        TP = TP + 1
                    if (Y1[i] == 1) and (P_01[i] == 0):
                        actual_pred.write(str(Y1[i]) + "," + str(P_01[i]) + "," + "FN" + "\n")
                        FN = FN + 1
                    if (Y1[i] == 0) and (P_01[i] == 1):
                        actual_pred.write(str(Y1[i]) + "," + str(P_01[i]) + "," + "FP" + "\n")
                        FP = FP + 1
                    if (Y1[i] == 0) and (P_01[i] == 0):
                        actual_pred.write(str(Y1[i]) + "," + str(P_01[i]) + "," + "TN" + "\n")
                        TN = TN + 1
                #print ("manual TN, FP, FN, TP =", TN, FP, FN, TP)

                # Accuracy, Sensitivity, Specificity, F score from confusion matrix
                cm1 = confusion_matrix(Y1, P_01)
                print('Confusion Matrix : \n', cm1)
                total1 = sum(sum(cm1))
                #############
                TP = cm1[0, 0]
                TN = cm1[1, 1]
                FN = cm1[0, 1]
                FP = cm1[1, 0]
                print("TN, FP, FN, TP", TN, FP, FN, TP)
                print ("ravel = ", cm1.ravel())


                accuracy = (cm1[0, 0] + cm1[1, 1]) / total1
                # print('Accuracy : ', accuracy1)
                sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
                # print('Sensitivity : ', sensitivity1)
                specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
                # print('Specificity : ', specificity1)
                F_score = f1_score(Y1, P_01, average='micro') ## initial micro , average='weighted'
                #F_score = f1_score(Y1, P_01)
                MCC = matthews_corrcoef(Y1, P_01)
                print("MCC : ", MCC)
                num = (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)
                MCC1 = ((TP * TN) - (FP * FN)) / pow(num, 0.5)
                print("MCC1 : ", MCC1)

                f1 = TP / (TP + 0.5*(FN + FP))
                print("F_score : ", f1)

                print(thres, threshold, "accuracy", "sensitivity", "specificity", "F_score", "auc_score", accuracy, sensitivity,
                      specificity, F_score, auc_score)


                TP = cm1[0, 0]
                TN = cm1[1, 1]
                FN = cm1[0, 1]
                FP = cm1[1, 0]

                Accuracy = (TP + TN) / (TP + FP + FN + TN)
                Specificity = TN / (TN + FP)
                Recall = TP / (TP + FN)
                Precision = TP / (TP + FP)
                F1_Score = 2 * ((Recall * Precision) / (Recall + Precision))
                print("------- new =", Accuracy, Specificity, Recall, Precision, F1_Score)

            auc_list.append(auc_score)
            Fscore_list.append(F_score)
            print("-------log_regparam=", log_regparam, "--------")
            print("accuracy", "sensitivity", "specificity", "F_score", "auc_score", accuracy, sensitivity, specificity,
                  F_score, auc_score)


    print ("auc_list over different data randomization =", auc_list)
    print("Fscore_list over different data randomization =", Fscore_list)
    Avg_AUC = (sum(auc_list) / len(auc_list))
    Avg_Fscore = (sum(Fscore_list) / len(Fscore_list))
    s1 = (str(log_regparam) + "," + str(Avg_AUC) + "," + str(Avg_Fscore))
    output_file.write(s1 + "\n")
    output_file.close()
    actual_pred.close()
if __name__=="__main__":
    main()
