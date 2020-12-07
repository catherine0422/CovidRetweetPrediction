import csv
import pandas as pd
from sklearn.dummy import DummyRegressor

# Load the training data
train_data = pd.read_csv("train.csv")
# Load the evaluation data
eval_data = pd.read_csv("evaluation.csv")

# Initialize the Dummy Regressor to use the Mean value of our data
dummy_regr = DummyRegressor(strategy="mean")
# Fit the regressor with our data
dummy_regr.fit(train_data, train_data['retweet_count'])
# Pass the evaluation data through the predict function which just gets the same value for every tweet
dummy_pred = dummy_regr.predict(eval_data)

# Dump the results into a file that follows the required Kaggle template
with open("mean_predictions.txt", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "NoRetweets"])
    for index, prediction in enumerate(dummy_pred):
        writer.writerow([str(eval_data['id'].iloc[index]) , str(int(prediction))])

# Initialize the Dummy Regressor that will constantly predicts 0 retweets
dummy_regr = DummyRegressor(strategy="constant", constant=0)
# Fit the regressor with our data, which does nothing in action
dummy_regr.fit(train_data, train_data['retweet_count'])
# Pass the evaluation data through the predict function which just gets value 0 for every tweet
dummy_pred = dummy_regr.predict(eval_data)

# Dump the results into a file that follows the required Kaggle template
with open("constant_predictions.txt", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "NoRetweets"])
    for index, prediction in enumerate(dummy_pred):
        writer.writerow([str(eval_data['id'].iloc[index]) , str(int(prediction))])