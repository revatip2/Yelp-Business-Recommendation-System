from pyspark import SparkContext
import sys
import time
import random
import math
from math import sqrt
from itertools import combinations
from collections import defaultdict

input_file_path = sys.argv[1]
output_file_path = sys.argv[3]
validation_file_path = sys.argv[2]

sc = SparkContext('local[*]', 'cf_item')
sc.setLogLevel("ERROR")
start = time.time()

start = time.time()
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


def pearson_similarity(business1, business2, common_users, user_ratings):

    mean1 = sum(user_ratings[business1][user] for user in common_users) / len(common_users)
    mean2 = sum(user_ratings[business2][user] for user in common_users) / len(common_users)

    numerator = sum(
        (user_ratings[business1][user] - mean1) * (user_ratings[business2][user] - mean2) for user in common_users)
    den1 = sum((user_ratings[business1][user] - mean1) ** 2 for user in common_users)
    denominator2 = sum((user_ratings[business2][user] - mean2) ** 2 for user in common_users)

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
        # if not common_users:
        #     continue
        if len(common_users) < min_common:
            #continue
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

heading = "user_id, business_id, prediction\n"
with open(output_file_path, "w") as output_file:
    output_file.write(heading)
    for rev in yelp_val_data_rdd.collect():
        prediction = item_item_cf(rev[0], rev[1], user_ratings_dict)
        # print(prediction)
        output_file.write(f"{rev[0]},{rev[1]},{str(prediction)}\n")

end = time.time()
print('Duration: ',end - start)

sc.stop()
