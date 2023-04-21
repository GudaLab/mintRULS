import numpy as np
import csv
from collections import Counter


def main():
    ####  gene and associated mbs dict ######################
    with open(
            "mouse_data_gene_miTS_info.csv") as csv_file:
        sample_list = csv.reader(csv_file, delimiter=',')
        ### prepare miTS dicts
        all_mbs_list = []
        all_geneSymb_list = []
        gene_mbs_dict = dict()
        mbs_seq_dict = dict()
        mbs_gene_dict = dict()
        a = 0
        for row in sample_list:
            a = a + 1
            # print (a)
            if a >= 2:
                mbs_gene_id = row[3]
                mbs_seq = row[1]
                mbs = row[0]
                all_mbs_list.append(mbs)
                all_geneSymb_list.append(mbs_gene_id)
                # print (mbs_gene_id, mbs)
                mbs_gene_dict [mbs] = mbs_gene_id

                if mbs_gene_id in gene_mbs_dict:
                    gene_mbs_dict[mbs_gene_id].append(mbs)
                else:
                    gene_mbs_dict[mbs_gene_id] = [mbs]
                mbs_seq_dict [mbs] = mbs_seq

    #### miRNA and associated dict ######################
    with open(
            "mouse_data_miRNA_info.csv") as csv_file:
        sample_list = csv.reader(csv_file, delimiter=',')
        ### prepare miRNAs dicts
        all_miRNA_list = []
        all_miRNA_symb_list = []
        miRNA_syn_dict = dict()
        miRNA_seq_dict =dict()
        a = 0
        for row in sample_list:
            a = a + 1
            # print (a)
            if a >= 2:
                miRNA_id = row[0]
                miRNA_seq = row[1]
                miRNA_syn = row[2]
                all_miRNA_list.append(miRNA_id)
                all_miRNA_symb_list.append(miRNA_syn)

                if miRNA_id in miRNA_syn_dict:
                    miRNA_syn_dict[miRNA_id].append(miRNA_syn)
                else:
                    miRNA_syn_dict[miRNA_id] = [miRNA_syn]

                if miRNA_id not in miRNA_seq_dict.keys():
                    miRNA_seq_dict[miRNA_id] = miRNA_seq

    ##################################################################
    # example:  get miTS of a query gene
    ##################################################################

    ## creating dictionry for predicted values
    mintRULS_pred_dict = dict()
    Original_Val_dict = dict()
    with open(
            "All_predictions_SeqLenAll.txt") as csv_file:
        sample_list = csv.reader(csv_file, delimiter=',')
        a = 0
        for row in sample_list:
            a = a + 1
            # print (a)
            if a >= 1:
                miRNA_id = row[0]
                miTS_id = row[1]
                actual_value = row[2]
                pred_value = row[3]
                pair_name = miRNA_id + "_" + miTS_id

                mintRULS_pred_dict[pair_name] = float(pred_value)
                Original_Val_dict[pair_name] = float(actual_value)

                #print (miRNA_id + "," + miTS_id + "," + str(actual_value) + "," + str(pred_value))


    print("----- imported all actual and predicted values-----")
    print("len Original_Val_dict = ", len(Original_Val_dict.keys()))
    print("len mintRULS_pred_dict = ", len(mintRULS_pred_dict.keys()))

    # shrinking outliers mannually
    for item in mintRULS_pred_dict.keys():
        if mintRULS_pred_dict[item] > 2:
            mintRULS_pred_dict[item] = 2
        elif mintRULS_pred_dict[item] < -0.5:
            mintRULS_pred_dict[item] = -0.5
        else:
            None

    ##################################################################################

    ## normalizing predicted values ###############
    norm_mintRULS_pred_dict = dict()

    ## normlizing  by min, max, .... between a and b, miRNA wise
    for miRNAID in all_miRNA_list:
        tmp_dict = dict()
        for miTS in all_mbs_list:
            pair = miRNAID + "_" + miTS
            val = float(mintRULS_pred_dict[pair])
            tmp_dict[pair] = val

        min_tmp_dict = np.min(list(tmp_dict.values()))
        max_tmp_dict = np.max(list(tmp_dict.values()))

        a = 0
        b = 1
        for pairs in tmp_dict.keys():
            v = tmp_dict[pairs]
            norm_v = (b - a) * ((v - min_tmp_dict) / (max_tmp_dict - min_tmp_dict)) + a
            norm_mintRULS_pred_dict[pairs] = norm_v
    print ("Normalization of predicted values ---done-----")
    #######################################################################################

    ## Qurtile caluclation #######################
    Quantile25th = np.percentile(list(norm_mintRULS_pred_dict.values()), [25])
    Quantile75th = np.percentile(list(norm_mintRULS_pred_dict.values()), [75])
    ##############################################

    ### Query here, for miRNAs which are in literature ###
    output_file = open(
        "Query_output.txt",'w')
    with open("query_interactions.csv") as csv_file:
        sample_list = csv.reader(csv_file, delimiter=',')
        a = 0
        for row in sample_list:
            a = a + 1
            print (a)
            if a >= 2:
                miRNA_name = row[0]
                miRNA_ID = row[1]
                miRNA_seq = row[2]
                targetGene = row[3]


                if miRNA_name in all_miRNA_symb_list:
                    ifYes_miRNA = "Yes"
                else:
                    ifYes_miRNA = "No"

                if targetGene in all_geneSymb_list:
                    ifYes_gene = "Yes"
                else:
                    ifYes_gene = "No"


                if ifYes_miRNA == "No":
                    s = miRNA_ID + "," + "miRNA not matched" + "," + " " + "," + " " + "," + targetGene + "," + " " + "," + " " + "," + " " + "," + " " + "," +" " + "," + " "
                    output_file.write(s + "\n")
                else:
                    if ifYes_gene == "No":
                        s = miRNA_ID + "," + miRNA_seq + "," +str(miRNA_name) + "," + targetGene + "," + "No gene matched" + "," + " " + "," + " " + "," + " " + "," + " " + "," + ""
                        output_file.write(s + "\n")
                    else:
                        print(miRNA_ID, miRNA_seq, targetGene, "Both miRNA & gene matched")
                        # get all miTS of gene
                        all_miTS = gene_mbs_dict[targetGene]

                        temp_dict = dict()
                        for miTS in all_miTS:
                            pair = miRNA_ID + "_" + miTS
                            print ("pair = ", pair)
                            miTS_seq = mbs_seq_dict[miTS]

                            norm_MINT_RULS_pred_Score = norm_mintRULS_pred_dict[pair]
                            temp_dict[pair] = float(norm_MINT_RULS_pred_Score)

                            
                        top_temp_dict = dict(Counter(temp_dict).most_common(1))  # select top N interactions here
                        #print("top_temp_dict", top_temp_dict)

                        for pairs, pred_values in top_temp_dict.items():
                            miRNA = pairs.split("_")[0]
                            miRNA_seq = miRNA_seq_dict[miRNA]
                            miRNA_syn = miRNA_syn_dict[miRNA]
                            miRNA_syn_new = ':'.join(str(d) for d in miRNA_syn)

                            miTS = pairs.split("_")[1]
                            miTS_seq = mbs_seq_dict[miTS]
                            miTS_gene = mbs_gene_dict[miTS]
                            #print (miRNA, miTS_gene, pred_value)

                            original_val = Original_Val_dict[pair]
                            pred_value = top_temp_dict[pairs]

                            # if float(pred_value) < 0.3:
                            #     classs = "Weak-Target"
                            #     #print(miRNA, miTS_gene, pred_value, classs)
                            # elif 0.3 <= float(pred_value) < 0.6:
                            #     classs = "Mild-Target"
                            #     #print(miRNA, miTS_gene, pred_value, classs)
                            # elif float(pred_value) >= 0.6:
                            #     classs = "Strong-Target"
                            #     #print(miRNA, miTS_gene, pred_value, classs)
                            # else:
                            #     None

                            if float(pred_value) < Quantile25th:
                                classs = "Weak-Target"
                                #print(miRNA, miTS_gene, pred_value, classs)
                            elif Quantile25th <= float(pred_value) < Quantile75th:
                                classs = "Mild-Target"
                                #print(miRNA, miTS_gene, pred_value, classs)
                            elif float(pred_value) >= Quantile75th:
                                classs = "Strong-Target"
                                #print(miRNA, miTS_gene, pred_value, classs)
                            else:
                                None


                            # write file
                            s = miRNA + "," + miRNA_seq + "," +str(miRNA_syn_new) + "," + miTS_gene + "," + miTS + "," + miTS_seq + "," + "Both miRNA and gene matched" + "," + str(original_val) + "," +str(pred_value) + "," + classs
                            output_file.write(s + "\n")
    output_file.close()
if __name__=="__main__":
    main()

