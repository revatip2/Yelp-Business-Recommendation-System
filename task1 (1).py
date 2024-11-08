from pyspark import SparkContext
import sys
import time
import random
from itertools import combinations
from collections import defaultdict

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

sc = SparkContext('local[*]', 'jaccard')
sc.setLogLevel("ERROR")
start = time.time()

yelp_data_rdd = sc.textFile(input_file_path)
row_one = yelp_data_rdd.first()
yelp_data_rdd = yelp_data_rdd.filter(lambda a: a != row_one).map(lambda a: a.split(','))

data_rdd = yelp_data_rdd.map(lambda row: (row[1], {row[0]})).reduceByKey(lambda x, y: x.union(y))
biz_dict = dict(data_rdd.collect())

# Assign serial numbers to user_id
users_rdd = yelp_data_rdd.map(lambda row: row[0]).distinct().sortBy(lambda x: x)
users_data = users_rdd.zipWithIndex().collectAsMap()

# Build Characteristic Matrix
characteristic_matrix = {}
for business, users in data_rdd.collect():
    row = []
    for user in users_data:
        if user in users:
            row.append(1)
        else:
            row.append(0)
    characteristic_matrix[business] = row

#print('Characteristic matrix done')

random.seed(42)
n = 100
seed = 42
bins = len(characteristic_matrix)

hash_list = []
a = random.sample(range(1, bins), n)
b = random.sample(range(1, bins), n)
hash_list.append((a, b))

hash_list_broadcast = sc.broadcast(hash_list[0])
def signature_matrix_calc(user_ids):
    values_signatures = []
    for i in range(n):
        initial_value = float("inf")
        for user in user_ids:
            user_n = users_data[user]
            a, b = hash_list_broadcast.value
            hash_value = (a[i] * user_n + b[i]) % bins
            initial_value = min(initial_value, hash_value)
        values_signatures.append(int(initial_value))
    return values_signatures

signature_rdd = data_rdd.map(lambda x: (x[0], signature_matrix_calc(x[1])))
signature_matrix = signature_rdd.collectAsMap()
#print("Signature Matrix done.")

rows = 2
bands = n//rows
hash_tables = [defaultdict(list) for _ in range(bands)]
for business, signature in signature_matrix.items():
    for i in range(bands):
        band_signature = signature[i * rows: (i + 1) * rows]
        hash_code = hash(tuple(band_signature))
        hash_tables[i][hash_code].append(business)

candidate_pairs_rdd = sc.parallelize(hash_tables)\
    .flatMap(lambda band: (pair for businesses in band.values() if len(businesses) > 1 for pair in combinations(businesses, 2))) \
    .distinct()
candidate_pairs = candidate_pairs_rdd.collect()
total_candidate_pairs = candidate_pairs_rdd.count()
print('candidate length: ', total_candidate_pairs)

def jaccard_similarity(a,b):
    intersection = len(a.intersection(b))
    union = len(a)+len(b)-intersection
    return intersection/union

#print("jaccard done")

final_candidate_pairs = []

for pair in candidate_pairs:
    biz1,biz2 = pair
    similarity = jaccard_similarity(set(biz_dict[biz1]), set(biz_dict[biz2]))
    if similarity >= 0.5:
        final_candidate_pairs.append([biz1, biz2, similarity])
#print("final pairs done")

sorted_similar_pairs = [sorted(sublist[:2]) + sublist[2:] for sublist in final_candidate_pairs]
sorted_similar_pairs.sort()

#print("sorted")

with open(output_file_path, 'w') as text_file:
    text_file.write("business_id_1, business_id_2, similarity\n")
    for pair in sorted_similar_pairs:
        biz1,biz2,similarity = pair
        text_file.write(f"{biz1},{biz2},{similarity}\n")

end = time.time()
print("Duration: ", end-start)
sc.stop()