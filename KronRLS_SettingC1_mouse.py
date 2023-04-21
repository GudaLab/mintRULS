from rlscore.learner import KronRLS
from rlscore.measure import cindex
from rlscore.measure import spearman
from rlscore.measure import sqerror
from rlscore.measure import accuracy
import data_load_validation_settings1_mouse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_score, accuracy_score, recall_score, average_precision_score, \
    confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import collections


def main():
    output_file = open(
        "/MyData_mouse/Results/SettingC/output_SetC.txt",'w')

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
    ##################################################################################

    auc_list = []
    Fscore_list = []
    for i in range(1): # nummber of times random selection occur on main interaction data

        print("############### START #############")
        print("Counter # ", i)
        ## selecting random miRNAs
        print(" Total human miRNA_ids", len(miRNA_ids))
        #miRNA_ids = miRNA_ids.sample(n=327)  # select random no of mbs chosen, total miRNA = 845
        random_miRNA_list = list(miRNA_ids)
        print("selected human miRNA= ", len(random_miRNA_list), random_miRNA_list[0:10])

        ## selecting random mbs
        print(" Total human mbs_ids", len(mbs_ids))
        #mbs_ids = mbs_ids.sample(n=4000)  # select random no of mbs chosen, total mbs = ?
        random_mbs_list = list(mbs_ids)
        print("selected human mRNA = ", len(random_mbs_list), random_mbs_list[0:10])


        ## performing model prediction
        X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = data_load_validation_settings1_mouse.settingC_split(random_miRNA_list, random_mbs_list)

        y_test_count = collections.Counter(Y_test)
        y_train_count = collections.Counter(Y_train)
        print("y_train_count = ", y_train_count)
        print("y_test_count = ", y_test_count)

        print("XD train dimensions %d %d" % X1_train.shape)
        print("XT train dimensions %d %d" % X2_train.shape)
        #print("miRNA-mbs train pairs: %d" % (Y_train.shape[0] * Y_train.shape[1]))
        print("-----------------------------------")
        print("XD test dimensions %d %d" % X1_test.shape)
        print("XT test dimensions %d %d" % X2_test.shape)
        #print("miRNA-mbs test pairs %d %d" % (Y_test.shape[0] * Y_test.shape[1]))

        log_regparams = range(35, 36)  # 35 is optimal
        learner = KronRLS(X1=X1_train, X2=X2_train, Y=Y_train)
        s = str(i) + ","
        for log_regparam in log_regparams:

            learner.solve(2.**log_regparam)
            P = learner.predict(X1_test, X2_test)

            # calculating AUC
            Y_test1 = np.array(Y_test)
            Y_test1 = Y_test1 > 50  # this is just to replace 100 with 1, in the adjacancy matrix
            Y_test1 = Y_test1.astype(int)
            Y_test1 = list(Y_test1)
            fpr2, tpr2, threshold = roc_curve(Y_test1, P)
            auc_score = roc_auc_score(Y_test1, P)
            print('AUC (LmiTOCV): %.3f' % auc_score)


            # F1-score and other parameter calculation
            for thres in [0.2, 0.3, 0.35, 0.4, 0.5]: # 0.5 is selected as best
                P_01 = P > thres
                P_01 = P_01.astype(int)

                # Accuracy, Sensitivity, Specificity, F score from confusion matrix
                cm1 = confusion_matrix(Y_test1, P_01)
                #print('Confusion Matrix : \n', cm1)
                total1 = sum(sum(cm1))

                accuracy = (cm1[0, 0] + cm1[1, 1]) / total1
                # print('Accuracy : ', accuracy1)
                sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
                # print('Sensitivity : ', sensitivity1)
                specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
                # print('Specificity : ', specificity1)
                F_score = f1_score(Y_test1, P_01, average='micro')
                # print("F_score : ", F_score)

                print(thres, "accuracy", "sensitivity", "specificity", "F_score", "auc_score", accuracy,
                      sensitivity,
                      specificity, F_score, auc_score)


            auc_list.append(auc_score)
            Fscore_list.append(F_score)
            print("-------log_regparam=", log_regparam, "--------")
            print ("accuracy", "sensitivity", "specificity", "F_score", "auc_score", accuracy, sensitivity, specificity, F_score, auc_score)

    print("auc_list over different data randomization =", auc_list)
    print("Fscore_list over different data randomization =", Fscore_list)
    Avg_AUC = (sum(auc_list) / len(auc_list))
    Avg_Fscore = (sum(Fscore_list) / len(Fscore_list))
    s1 = (str(log_regparam) + "," + str(Avg_AUC) + "," + str(Avg_Fscore))
    output_file.write(s1 + "\n")
    output_file.close()
if __name__=="__main__":
    main()
