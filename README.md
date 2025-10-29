# SolarFlare_Nowcasting_Transformers_Classic_models

## Data

GOES data was captured using the sunpy_5 library, using the sunkit_instruments.goes_xrs function. Explosion data was captured from class M1. In this database, data from May 1, 2010 to Abril 30, 2024 were used.
The SHARP data were obtained through a URL provided by JSOC, which allowed the download of the hmi.sharp_720s database, allowing us to choose the time period and parameters. The SHARP data were selected from the list of solar flares obtained by GOES.
This script was created from the script by Jacob Conrad Trinidad (https://stacks.stanford.edu/file/druid:yv269xg0995/Predicting%20Solar%20Flares%20using%20SVMs.html), carried out from the work of Bobra and Couvidat (2015) (https://doi.org/10.1088/0004-637X/798/2/135). 

The final base is available on Zenodo (https://zenodo.org/records/10800828)


In the data directory we have some files that are important for capturing data. If you want to make a new collection, it is important that they are deleted so that the script creates updated versions of them:

### all_harps_with_noaa_ars.txt

Lists the active regions mapped, showing their number corresponding to HARPNUM, which is used in the SHARP database and their NOAA_ARS number, used by GOES. This file is constantly updated and the script captures the latest available version. His official address is:

http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt

## Model

All models used (MLP, SVM, LSTM and Transformers) were created using the [TensorFlow](https://www.tensorflow.org) library, together with [Keras](https://keras.io). Additionally, the [Sklearn](https://scikit-learn.org/stable/) library was used for other features related to the models, such as data division, normalization and evaluation metrics. For SMOTE balancing, the [Imbalanced Learn](https://imbalanced-learn.org/) library was used.



## Balancing

All models can be trained with unbalanced or balanced data, using undersampling, oversampling, and SMOTE techniques.

## Division of sets

The division between the training, validation and test sets can be done randomly or chronologically, considering the recording date of each event.

## Normalization

In all models, data normalization occurred using the StandardScaler library, which sets the mean to 0 and standard deviation to 1.




## Hyperparameters

|Model| Description|
|------------|------------|
|MLP         | epochs = 100 (early stopping 10), batch = 64, architecture = 256–128–64–32–1, L2 = [0.001,0.003,0.003], dropout = [0.4,0.3,0.2], Adam (1 × 10−4), binary loss, SMOTE (ratio 0.6; k = 1)|
|LSTM        |epochs = 100, batch = 64, input seq.: 18 × 1, LSTM stacked: 32 → 16, regularization: BatchNorm + Dropout 0.3, final denses: 16 ReLU → 1 Sigmoid, Adam (1 × 10−4), binary loss, SMOTE (ratio 0.6; k = 1).|
|SVM         |epochs = 100, batch = 64, input = 18 features, RFF: output dim = 1024, scale = 5.0, kernel gaussian, architecture = 256–128–64–1, Adam (1 × 10−4), binary loss, SMOTE (ratio 0.6; k = 1)|
|Transformers-base| epochs = 100 (early stopping 10), batch = 64, head size = 128, num heads = 8, FF-dim = 128, blocks = 6, dropout = 0.2, MLP dropout = 0.2, Adam (1 × 10−4), FocalLoss (γ = 2, α = 0.25), SMOTE (ratio 1.0; k standard).|
|FT_Transformers| epochs = 100 (early stopping 10), batch = 64, embed dim = 128, num heads = 8, FF-dim = 128, blocks = 6, dropout = 0.3, Adam (1 × 10−4), FocalLoss (γ = 2, α = 0.25), SMOTE (ratio 1.0; k standard)|
|Transformers (Time Series)|epochs = 100 (early stopping 10), batch = 64, head size = 192, num heads = 12, FF-dim = 256, blocks = 6, MLP units = [256,128,64], MLP dropout = 0.3, dropout global = 0.2, Adam (5 × 10−5), FocalLoss (γ = 2, α = 0.85), SMOTE (ratio 1.0; k = 3)|
|TabPFN|manual adjustment = none, balancing = undersampling, train samples = 1024.|
|XGBoost|n_estimators = 50, learning rate = 0.02, max depth = 8, subsample = 0.8, colsample_bytree = 0.8, gamma = 0.10, reg_alpha = 0.10, scale_pos_weight = 190–200, eval_metric = {auc, aucpr, logloss}, tree_method = hist, random_state = 42, n_jobs = -1, SMOTE (ratio 0.6; k = 3).|
|CatBoost|iterations = 150, learning rate = 0.09, depth = 7, min_data_in_leaf = 2, subsample = 0.8, rsm = 0.8, random_strength = 0.10, l2_leaf_reg = 1.0, bagging_temperature = 0.1, eval_metric = PRAUC, custom_metric = {AUC, Logloss}, random_seed = 42, verbose = 100, SMOTE (ratio 0.6; k = 3)|
|Random Forest|n_estimators = 600, max depth = 50, min samples split = 2, min samples leaf = 1, max features = sqrt, class weight = balanced, n_jobs = 1, random_state = 42, oob score = True. |
|Logistic Regresion|penalty = l2, C = 1.0, solver = lbfgs, max iter = 1000, class weight = balanced, n_jobs = -1, random_state = 42 |
