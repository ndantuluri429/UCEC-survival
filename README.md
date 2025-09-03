# Predicting Survival Outcomes for Patients with Uterine Corpus Endometrial Carcinoma (UCEC) using Machine-Learning Based Risk Prediction Models

## Project Overview
This study develops a machine learning framework to predict survival in Uterine Corpus Endometrial Carcinoma (UCEC) by integrating clinical variables, copy number variations (CNV’s) and DNA methylation profiles from The Cancer Genome Atlas (TCGA) dataset (~560 patients). UCEC is the most common gynecological malignancy, causing significant morbidity and mortality in postmenopausal women. A novel multimodal feature engineering pipeline was used to train Cox Proportional Hazards Regression (CoxPH) and Random Survival Forest (RSF) models with feature selection informed by statistical analysis and logistic regression. RSF demonstrated superior performance (C-index 0.68 with 10-fold CV vs CoxPH 0.63), though non-CV RSF performance (~0.90) requires external validation due to possibilities of overfitting. Significant variables with CoxPH hazard ratios > 1 include FIGO stage, CDO1 (implicated in UCEC), and prior treatment. Important clinical and genomic features identified for increasing risk of death in UCEC patients with RSF included: FIGO stage, age, whether or not prior treatment was received, differential methylation of genes such as CDO1 and C9orf125 (implicated in triple negative breast cancer). Kaplan-Meier analysis indicated a 3-fold difference with distinct differentiation between high- and low-risk groups. Though this study was limited by sample size, feature-space imbalance, and the lack of an external validation set, multimodal ML integration shows promise for stratified UCEC prognosis, particularly when accounting for molecular heterogeneity through methylation-CNV interplay. These findings have the potential to inform personalized risk stratification and treatment regimens for UCEC, while also enabling the development of early detection protocols and advancing understanding of the disease.

## Repository Contents
### /UCEC-how-to-download-data.pdf
### /clinical_baseline-2.py
### /p_test_and_save_dfs_multiple_genes.py


## Acknowledgements
Developed by Ananya Prasad, Surabhi Ghatti, Neha Dantuluri at [Harvard Medical School's Department of Biomedical Informatics](https://dbmi.hms.harvard.edu). Mentored by Grey Kuling, Courtney Shearer, and Marinka Zitnik for [BMI 702: Biomedical AI](https://zitniklab.hms.harvard.edu/BMI702/) at Harvard Medical School. 
