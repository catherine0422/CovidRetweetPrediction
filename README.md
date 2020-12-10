# Covid Retweet Prediction
Retweet prediction of the tweets about Covid 19.

## Data

We use the dataset of kaggle challenge [COVID19 Retweet Prediction Challenge 2020](https://www.kaggle.com/c/covid19-retweet-prediction-challenge-2020/data).

Please put the origin data file `evaluation.csv` and `train.csv` under the `data` repository.

## Features extraction

First we need to do features extraction from the origin data. 

Use the file `features_extraction.ipynb` which is under the `code` repository. Run all the cells. The data with extracted features will be saved under the `data` repository on the names: `evaluation_transformed.csv` and `train_transformed.csv`.

### Features analyses

All the files used are under `code` repository.

* The file `features_selection.ipynb` is a representation of how we choose the components to keep for tf-idf feature after dimension reduction.
* The file `features_analyse.Rmd` is some analyses of the relation ship between extracted features and retweet count. The file `features_analyse.html` is a representation on HTML format.

## Run Models

We experimented 3 basic models and 4 enhanced models, each with a jupyter notebook file to that you can run. The origin file is use some default arguments of each model (to keep consistency to make a comparison between different models), to get the best predicting result of each model, please modify some arguments as explained follow.

### SVM (regressor_linearSVR.ipynb)

You should put your data in the `data` repository. This file uses the data after features extraction so the name should be `evaluation_transformed.csv` and `train_transformed.csv`. 
<br>
<br>
You could select features for training by modifying `features_selected` and for normalization by modifying `features_need_scaled` in the 1st part of code. 
<br>
<br>
You could choose model's parameters (`n_estimators` and `max_depth`) in the first part of code. 
<br>
<br>
The prediction on `evaluation_transformed.csv` will be writed into `prediction` repository, whose file name could be personalized in the last cell. 
<br>
<br>
After running all the cells, you will: get the mean MAE score of cross validation (2nd part of code) and get the prediction result on `evaluation_transformed.csv`. 
<br>
<br>
This guide works also for GBR, RF, RF after logarithmization, GBR after logarithmization, SVM-enhanced RF and RF-enhanced SVM. 

### GBR (regressor_gradient_boosting.ipynb)

Parameters settings : n_estimators = 100, max_depth = 18.

### RF (regressor_random_forest.ipynb)

Parameters settings : n_estimators = 100, max_depth = 18.

### RF after logarithmization (regressor_log_ randomforest.ipynb)

Parameters settings : n_estimators = 500, max_depth = 18. (This corresponds to our best MAE score)

### GBR after logarithmization (regressor_log_gradient_boosting_regressor.ipynb)

Parameters settings : n_estimators = 100, max_depth = 18.

### SVM-enhanced RF (regressor_svr_enhanced_random_forest.ipynb)

Parameters settings : n_estimators = 100, max_depth = 18.

### RF-enhanced SVM (regressor_random_forest_enhanced_svr.ipynb)

Parameters settings : n_estimators = 100, max_depth = 18.
