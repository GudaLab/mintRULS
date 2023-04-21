from rlscore.learner import KronRLS
from rlscore.measure import cindex
from rlscore.measure import spearman
from rlscore.measure import sqerror
from rlscore.measure import accuracy
import data_load_validation_settings1
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import collections

def main():
    output_file = open(
        "/MyData/Results/SettingC/output_SetC.txt",'w')


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

    auc_list = []
    Fscore_list = []
    for i in range(1): # nummber of times random selection occur on main interaction data

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


        ## MINT-RULS
        X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = data_load_validation_settings1.settingC_split(random_miRNA_list, random_mbs_list)

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

        learner = KronRLS(X1 = X1_train, X2 = X2_train, Y = Y_train)
        log_regparams = range(35, 36)  # 37 is best
        s = str(i) + ","
        for log_regparam in log_regparams:
            learner.solve(2.**log_regparam)
            P = learner.predict(X1_test, X2_test)

            # calculating AUC
            Y_test1 = np.array(Y_test)
            Y_test1 = Y_test1 > 50  # this is just to replace 100 with 1, in the adjacancy matrix
            Y_test1 = Y_test1.astype(int)
            Y_test1 = list(Y_test1)

            auc_score = roc_auc_score(Y_test1, P)
            fpr2, tpr2, threshold = roc_curve(Y_test1, P)
            print('AUC (LmiTOCV): %.3f' % auc_score)

            # F1-score and other parameter calculation
            for thres in [0.3, 0.35, 0.4, 0.5]:  # 0.35 is selected as best
                P_01 = P > thres
                P_01 = P_01.astype(int)

                # Accuracy, Sensitivity, Specificity, F score from confusion matrix
                cm1 = confusion_matrix(Y_test1, P_01)
                # print('Confusion Matrix : \n', cm1)
                total1 = sum(sum(cm1))

                accuracy = (cm1[0, 0] + cm1[1, 1]) / total1
                # print('Accuracy : ', accuracy1)
                sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
                # print('Sensitivity : ', sensitivity1)
                specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
                # print('Specificity : ', specificity1)
                F_score = f1_score(Y_test1, P_01, average='micro')
                # print("F_score : ", F_score)

                print(thres, threshold, "accuracy", "sensitivity", "specificity", "F_score", "auc_score", accuracy,
                      sensitivity,
                      specificity, F_score, auc_score)

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
