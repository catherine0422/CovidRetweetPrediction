# Covid Retweet Prediction
Retweet prediction of the tweets about Covid 19.

## Data

We use the dataset of kaggle challenge [COVID19 Retweet Prediction Challenge 2020](https://www.kaggle.com/c/covid19-retweet-prediction-challenge-2020/data).

Please put the origin data file `evaluation.csv` and `train.csv` under the `data` repository.

## Features extraction

First we need to do features extraction from the origin data. 

Use the file `features_extraction.ipynb` which is under the `code` repository. Run all the cells. The data with extracted features will be saved under the `data` repository on the names: `evaluation_transformed.csv` and `train_transformed.csv`.

## Run Models

We experimented 3 basic models and 4 enhanced models, each with a jupyter notebook file to that you can run. The origin file is use some default arguments of each model (to keep consistency to make a comparison between different models), to get the best predicting result of each model, please modify some arguments as explained follow.

### SVM

### GBR

### RF

### RF after logarithmization

### GBR after logarithmization

### SVM-enhanced RF

### RF-enhanced SVM

