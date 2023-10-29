from pyspark import SparkContext
import sys
import time
import json
from datetime import datetime
import numpy as np
from xgboost import XGBRegressor


folder_path = sys.argv[1]
output_file_path = sys.argv[3]
validation_file_path = sys.argv[2]

sc = SparkContext('local[*]', 'model_base')
sc.setLogLevel("ERROR")

start = time.time()

yelp_data = '/yelp_train.csv'
yelp_train_data_rdd = sc.textFile(folder_path + yelp_data)
row_one = yelp_train_data_rdd.first()
yelp_train_data_rdd = yelp_train_data_rdd.filter(lambda a: a != row_one).map(lambda a: a.split(','))

current_year = sc.broadcast(datetime.now().year)
user_file = '/user.json'
user_doc = sc.textFile(folder_path + user_file)

def process_user_data(user):
    user_id = user["user_id"]
    review_count = float(user["review_count"])
    yelping_since = current_year.value - int(user["yelping_since"][:4])
    average_stars = float(user["average_stars"])
    return (user_id, (review_count, yelping_since, average_stars))

user_features_rdd = user_doc.map(lambda u: json.loads(u)).map(process_user_data)
uz_features = dict(user_features_rdd.collect())

biz_file = '/business.json'
biz_doc = sc.textFile(folder_path + biz_file)


def process_biz_data(biz):
    biz_id = biz['business_id']
    stars = float(biz['stars'])
    biz_review_count = float(biz['review_count'])
    is_open = biz['is_open']
    return (biz_id, (stars, biz_review_count, is_open))

biz_features_rdd = biz_doc.map(lambda b: json.loads(b)).map(process_biz_data)
biz_features = dict(biz_features_rdd.collect())

def extract_features(record):
    user, biz, rating = record
    if user in uz_features:
        review_count, yelping_since, average_stars = uz_features[user]
    else:
        review_count, yelping_since, average_stars = None, None, None

    if biz in biz_features:
        stars, biz_review_count, is_open = biz_features[biz]
    else:
        stars, biz_review_count, is_open = None, None, None

    return (review_count, yelping_since, average_stars, stars, biz_review_count, is_open, rating)


X_train_and_Y_train = yelp_train_data_rdd.map(extract_features)
X_train = X_train_and_Y_train.map(lambda a: a[:-1]).collect()
Y_train = X_train_and_Y_train.map(lambda a: a[-1]).collect()

# print(X_train.take(5))
# print(Y_train.take(5))

yelp_val_data_rdd = sc.textFile(validation_file_path)
row_one_val = yelp_val_data_rdd.first()
yelp_val_data_rdd = yelp_val_data_rdd.filter(lambda a:a != row_one_val).map(lambda a: a.split(","))
yelp_val_uz_biz = yelp_val_data_rdd.map(lambda a: (a[0], a[1]))
uz_biz = yelp_val_uz_biz.collect()
#print('1: ',yelp_val_uz_biz.take(10))

def extract_validation_features(record):
    user, biz = record
    if user in uz_features:
        review_count, yelping_since, average_stars = uz_features[user]
    else:
        review_count, yelping_since, average_stars = None, None, None

    if biz in biz_features:
        stars, biz_review_count, is_open = biz_features[biz]
    else:
        stars, biz_review_count, is_open = None, None, None

    return (review_count, yelping_since, average_stars, stars, biz_review_count, is_open)

validation_dataset = yelp_val_uz_biz.map(extract_validation_features)

X_val = validation_dataset.map(lambda a: a).collect()

X_train = np.array(X_train, dtype='float32')
Y_train = np.array(Y_train, dtype='float32')
X_val = np.array(X_val, dtype='float32')

model_xgboost = XGBRegressor(
    min_child_weight=100,
    random_state=1234,
    learning_rate=0.02,
    colsample_bytree=0.6,
    max_depth=18,
    subsample=0.7,
    n_estimators=350,
    alpha=0.26611,
    reg_lambda=7.5738
)
model_xgboost.fit(X_train, Y_train)
predicted_val = model_xgboost.predict(X_val)

heading = "user_id, business_id, prediction\n"
with open(output_file_path, "w") as output_file:
    output_file.write(heading)
    for i in range(0, len(predicted_val)):
        output_file.write(f'{uz_biz[i][0]},{uz_biz[i][1]},{str(predicted_val[i])}\n')
end = time.time()
print('Duration: ', end - start)
sc.stop()




