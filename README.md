# mintRULS Work Flow
## About
mintRULS is a machine learning-based tool developed in python environment to predict interactions between miRNA and gene target sites. The methodology employs features of miRNAs and their target sites using pairwise similarity metrics based on free energy, sequence and repeat identities, and target site accessibility to predict miRNA-target site interactions.
## Software Download
The user needs to download the complete software code in zip format from https://doi.org/10.5281/zenodo.5639816. Unzip the file and confirm the presence of the following data folders.	

## Data folders
Please find the following folders.
1)	Folder “MyData” contains all required sub-folders e.g. “Interaction_data”, “miRNA_features”, “mRNA_features”, “Results”, and corresponding materials.
2)	Similarly, in case of simulation using mouse data, MyData_mouse folder contains the same pattern of folders and sub-folders.
Inside individual folders-
a)	The interaction data (known or imputed) has to be filled in the “\MyData\Interaction_data\human_data”.
b)	The different similarity scores (e.g. FE, SSR, Gaussian, and Needleman) of query miRNA with the existing ones should be calculated separately and kept in the “\MyData\miRNA_features”. 
c)	The different similarity scores of query miTS should be in the “\MyData\mRNA_features”.
d)	The result folder “\MyData\Result” contains simulation results in text file.

The same pattern of folders and sub-folders can be followed in the case of mouse data. 

## Run Your Query
Running query interactions on human/mouse data
The folder “Query_interactions_Human_and_Mouse” contains separate sub-folders for human and mouse. Inside each sub-folder, run the following scripts for predicting the score of your query pair- 
1)	Write query pairs in the file “query_interactions.csv” [given an example in the Excel file].
2)	Run “1_Human_data_query_interactions.py”, and check the output in file “Query_output.txt”. 
A similar workflow can be followed in the case of mouse data. 

## Scripts for cross-validations
If interested, the user can also run the following scripts to perform cross-validations in different settings. 
Simulations on human data
KronRLS_settingA1.py is for running simulations in LOOCV conditions.
KronRLS_settingC1.py is for running simulations in LmiTOCV conditions.
Simulations on mouse data
KronRLS_settingA1_mouse.py is for running simulations in LOOCV conditions.
KronRLS_settingC1_mouse.py is for running simulations in LmiTOCV conditions.
## Citation
Shakyawar S, Southekal S, Guda C. (2022). mintRULS: Prediction of miRNA-mRNA target site interactions using regularized least square method. Genes, 13(9), 1528. https://doi.org/10.3390/genes13091528
