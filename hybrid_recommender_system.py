from pyspark import SparkContext
import sys
import time
import json
from datetime import datetime
import numpy as np
from xgboost import XGBRegressor
import math
from math import sqrt
from itertools import combinations
from collections import defaultdict

folder_path = sys.argv[1]
validation_file_path = sys.argv[2]
output_file_path = sys.argv[3]

sc = SparkContext('local[*]', 'hybrid_r')
sc.setLogLevel("ERROR")

start = time.time()

# Case 1
input_file_path = folder_path+'/yelp_train.csv'
yelp_train_data_rdd = sc.textFile(input_file_path)
row_one = yelp_train_data_rdd.first()
yelp_train_data_rdd = yelp_train_data_rdd.filter(lambda a: a != row_one).map(lambda a: a.split(','))
biz_user_tups = yelp_train_data_rdd.map(lambda a: (a[0], a[1])).collect()

biz_users_rdd = yelp_train_data_rdd.map(lambda a: (a[1], {a[0]})).reduceByKey(lambda a,b: a|b).collectAsMap()
user_biz_rdd = yelp_train_data_rdd.map(lambda a: (a[0], {a[1]})).reduceByKey(lambda a,b: a|b).collectAsMap()

biz_ratings_rdd = yelp_train_data_rdd.map(lambda a: (a[0], (float(a[2]), 1)))
biz_sum_ct = biz_ratings_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
biz_avg_rating = biz_sum_ct.mapValues(lambda a: a[0] / a[1])
biz_avg_dict = biz_avg_rating.collectAsMap()

user_ratings = yelp_train_data_rdd.map(lambda a: (a[1], {a[0]: float(a[2])})) \
                 .reduceByKey(lambda a, b: a.update(b) or a)

user_ratings_dict = user_ratings.collectAsMap()


def pearson_similarity(biz1, biz2, common_users, user_ratings):

    r1 = sum(user_ratings[biz1][user] for user in common_users) / len(common_users)
    r2 = sum(user_ratings[biz2][user] for user in common_users) / len(common_users)

    numerator = sum(
        (user_ratings[biz1][user] - r1) * (user_ratings[biz2][user] - r2) for user in common_users)
    denominator1 = sum((user_ratings[biz1][user] - r1) ** 2 for user in common_users)
    denominator2 = sum((user_ratings[biz2][user] - r2) ** 2 for user in common_users)

    if denominator1 == 0 or denominator2 == 0:
        return 0
    else:
        similarity = numerator / (sqrt(denominator1) * sqrt(denominator2))
        return similarity
def item_item_cf(user, business, user_ratings_dict):
    min_common = 1
    if user not in user_biz_rdd:
        return 3.5
    if business not in biz_users_rdd:
        return biz_avg_dict[user]

    similarity_list = []
    for other_business in user_biz_rdd[user]:
        if other_business == business:
            continue
        common_users = biz_users_rdd[business] & biz_users_rdd[other_business]
        if len(common_users) < min_common:
            return biz_avg_dict[user]

        similarity = pearson_similarity(business, other_business, common_users, user_ratings_dict)
        similarity_list.append((similarity, other_business))

    similarity_list.sort(reverse=True, key=lambda x: x[0])
    similarity_list = similarity_list[:3]

    weighted_sum = 0
    denominator = 0

    for similarity, other_business in similarity_list:
        weighted_sum += similarity * user_ratings_dict[other_business][user]
        denominator += abs(similarity)
    if denominator == 0:
        return 3.5
    prediction = weighted_sum / denominator
    return prediction

yelp_val_data_rdd = sc.textFile(validation_file_path)
row_one_val = yelp_val_data_rdd.first()
yelp_val_data_rdd = yelp_val_data_rdd.filter(lambda a:a != row_one_val).map(lambda a: a.split(",")).map(lambda a: (a[0], a[1]))

prediction = []
for rev in yelp_val_data_rdd.collect():
    prediction.append(str((item_item_cf(rev[0], rev[1], user_ratings_dict))))

# Case 2

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

uz_biz = yelp_val_data_rdd.collect()

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

validation_dataset = yelp_val_data_rdd.map(extract_validation_features)

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
predicted_val_list = predicted_val.tolist()

print(f'item based: {type(prediction)}, {prediction[:10]}')
print(f'model based: {type(predicted_val_list)}, {predicted_val_list[:10]}')

alpha = 0.1

combined_preds = []
for a, b in zip(prediction, predicted_val_list):
    combined = (alpha * float(a)) + ((1 - alpha) * b)
    combined_preds.append(combined)

heading = "user_id, business_id, prediction\n"
with open(output_file_path, "w") as output_file:
    output_file.write(heading)
    for i in range(0, len(predicted_val)):
        output_file.write(f'{uz_biz[i][0]},{uz_biz[i][1]},{str(combined_preds[i])}\n')

end = time.time()
print('Duration: ', end - start)
sc.stop()
