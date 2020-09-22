from pyspark import SparkContext
import xgboost as xgb
import numpy as np
import json
import sys
import time
import itertools
import math

start_time = time.time()
sc = SparkContext('local[*]', 'task2_3')
sc.setLogLevel("ERROR")
folder_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]

train_rdd_with_header = sc.textFile(folder_path + "/yelp_train.csv")
header = train_rdd_with_header.first()
train_rdd = train_rdd_with_header.filter(lambda x: x != header)

test_rdd_with_header = sc.textFile(test_file_path)
test_header = test_rdd_with_header.first()
test_rdd = test_rdd_with_header.filter(lambda x: x != test_header)


################################################################ START OF ITEM BASED ####################################################################


def find_jaccard_similarity(pair, business_users_dict):
    business_id_1 = pair[0]
    business_id_2 = pair[1]

    users_1 = business_users_dict[business_id_1]
    users_2 = business_users_dict[business_id_2]

    intersection_size = len(set(users_1).intersection(set(users_2)))
    union_size = len(set(users_1)) + len(set(users_2)) - intersection_size

    return intersection_size / union_size


def apply_hash_function(a, x, b, m):
    return ((a * x) + b) % m


def get_signature(business_users_map, user_index_map, num_of_hashes):
    business_id = business_users_map[0]
    users = business_users_map[1]
    signature = []
    primes_list = [4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549,
                   4561, 4567, 4583,
                   4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723,
                   4729, 4733, 4751,
                   4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919,
                   4931, 4933, 4937,
                   4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059,
                   5077, 5081, 5087]
    i = 0
    j = 20
    for hash_func_num in range(num_of_hashes):
        min_value = float("inf")
        for user in users:
            value = apply_hash_function(primes_list[i], user_index_map[user], primes_list[j], 690)

            if value < min_value:
                min_value = value
        signature.append(min_value)
        i += 1
        j += 1
    return business_id, signature


def find_pearson_similarity(business_id_pair):
    business_id_1 = business_id_pair[0]
    business_id_2 = business_id_pair[1]
    business1_users = business_id_users_avg_dict[business_id_1][0]
    business1_avg = business_id_users_avg_dict[business_id_1][1]
    business2_users = business_id_users_avg_dict[business_id_2][0]
    business2_avg = business_id_users_avg_dict[business_id_2][1]
    co_rated_users = business1_users.intersection(business2_users)

    co_rated_business_id_1_ratings = []
    co_rated_business_id_2_ratings = []
    for co_rated_user in co_rated_users:
        co_rated_business_id_1_ratings.append(train_data_dict[(co_rated_user, business_id_1)] - business1_avg)
        co_rated_business_id_2_ratings.append(train_data_dict[(co_rated_user, business_id_2)] - business2_avg)

    sum = 0
    for i in range(len(co_rated_business_id_1_ratings)):
        sum += (co_rated_business_id_1_ratings[i] * co_rated_business_id_2_ratings[i])

    sum_of_square_1 = 0
    for i in range(len(co_rated_business_id_1_ratings)):
        sum_of_square_1 += math.pow(co_rated_business_id_1_ratings[i], 2.0)
    root_1 = math.sqrt(sum_of_square_1)

    sum_of_square_2 = 0
    for i in range(len(co_rated_business_id_2_ratings)):
        sum_of_square_2 += math.pow(co_rated_business_id_2_ratings[i], 2.0)
    root_2 = math.sqrt(sum_of_square_2)

    if root_1 == 0 or root_2 == 0:
        return 0
    return sum / (root_1 * root_2)


def predict_rating(test_pair):
    test_user_id = test_pair[0]
    test_business_id = test_pair[1]

    existing_user = test_user_id in user_index_map
    existing_business = test_business_id in business_id_users_avg_dict

    # case 1: existing user_id and new bus_id
    if existing_user and not existing_business:
        # print("existing_user and not existing_business")
        avg_user_rating = user_business_avg_dict[test_user_id][1]
        return avg_user_rating

    # case 2: new user and existing business
    elif not existing_user and existing_business:
        # print("not existing_user and existing_business:")
        avg_bus_rating = business_id_users_avg_dict[test_business_id][1]
        return avg_bus_rating

    # case 3: new user and new business
    elif not existing_user and not existing_business:
        # print("not existing_user and not existing_business")
        return train_avg_rating

    # case 4: existing user and existing business already rated
    elif (test_user_id, test_business_id) in train_data_dict:
        return train_data_dict[(test_user_id, test_business_id)]

    # case 5: existing user and existing business not rated
    else:
        sum = 0
        deno = 0
        if test_business_id in item_neighbours_dict:
            top_N_neighbours = sorted(item_neighbours_dict[test_business_id], key=lambda x: -x[1])[:N]
            for value in top_N_neighbours:
                similar_col = value[0]
                similarity = value[1]
                if (test_user_id, similar_col) in train_data_dict:
                    similar_col_rating = train_data_dict[(test_user_id, similar_col)]
                else:
                    similar_col_rating = user_business_avg_dict[test_user_id][1]
                sum += (similar_col_rating * similarity)
                deno += math.fabs(similarity)
        if deno == 0:
            # print("predict: deno = 0, sum = ", sum)
            return user_business_avg_dict[test_user_id][1]
    return sum / deno


rdd_without_header = train_rdd
# rdd = sc.textFile("/Users/ramya/Desktop/DM/HW3/data/yelp_train.csv")
# header = rdd.first()
# rdd_without_header = rdd.filter(lambda record: record != header)
# =============================================== LSH begins =========================================================

input_rdd = rdd_without_header \
    .map(lambda line: (line.split(",")[1], [line.split(",")[0]])) \
    .reduceByKey(lambda a, b: a + b)
business_users_dict = input_rdd.collectAsMap()

list_of_users_rdd = rdd_without_header.map(
    lambda x: (x.split(",")[0], ([x.split(",")[1]], float(x.split(",")[2]), 1))) \
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])).map(
    lambda x: (x[0], [x[1][0], x[1][1] / x[1][2]]))

user_business_avg_dict = list_of_users_rdd.collectAsMap()
user_list = list_of_users_rdd.keys().collect()
user_index_map = {v: k for k, v in enumerate(user_list)}

num_of_hashes = 28
b = 14
band_start = 0

signature_rdd = input_rdd.map(
    lambda business_users_map: (get_signature(business_users_map, user_index_map, num_of_hashes)))
# print("signature_rdd.count() ", signature_rdd.count())
r = int(num_of_hashes / b)
previous_candidates_rdd = sc.emptyRDD()
for band in range(b):
    candidate_pairs_rdd = signature_rdd.map(
        lambda business_sig_map: (business_sig_map[0], business_sig_map[1][band_start:band_start + r])) \
        .map(lambda x: (tuple(x[1]), [x[0]])) \
        .reduceByKey(lambda a, b: a + b) \
        .filter(lambda z: len(z[1]) > 1) \
        .map(lambda x: itertools.combinations(sorted(x[1]), 2)) \
        .flatMap(lambda x: x)
    previous_candidates_rdd = previous_candidates_rdd.union(candidate_pairs_rdd)
    band_start = band_start + r
# print("num of candidates: ", previous_candidates_rdd.count())
distinct_candidates_rdd = previous_candidates_rdd.distinct()
similar_business_rdd = distinct_candidates_rdd.map(lambda pair: (pair[0], pair[1]))
print("total num of candidates fom LSH: ", similar_business_rdd.count())
# ===============================================================================================================================

# data matrix
# d = {(user_id, bus_id): rating}
train_data_dict = rdd_without_header.map(
    lambda line: ((line.split(",")[0], line.split(",")[1]), float(line.split(",")[2]))).collectAsMap()

train_avg_rating = rdd_without_header.map(lambda x: (1, (float(x.split(",")[2]), 1))).reduceByKey(
    lambda a, b: (a[0] + b[0], a[1] + b[1])).map(lambda x: x[1][0] / x[1][1]).collect()[0]

# stores for each business, list of users who have rated it and its average rating
# d = {bus_id: ({user3, user1, user2}, avg)}
business_id_users_avg_dict = rdd_without_header.map(
    lambda x: (x.split(",")[1], ([x.split(",")[0]], float(x.split(",")[2]), 1))) \
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])) \
    .map(lambda x: (x[0], (set(x[1][0]), x[1][1] / x[1][2]))).collectAsMap()

# print("business_id_users_avg_dict: ")
# filtered similar business_ids and their similarity
# {bus_id_1:[(bus_id_2, 0.7), (bus_id_3, 0.6)]}
item_neighbours_dict = similar_business_rdd.map(lambda x: (x[0], x[1], find_pearson_similarity(x))) \
    .filter(lambda x: x[2] >= 0.0) \
    .map(lambda x: [(x[0], [(x[1], x[2])]), (x[1], [(x[0], x[2])])]).flatMap(lambda x: x) \
    .reduceByKey(lambda a, b: a + b).collectAsMap()

# print("item_neighbours_dict: ", len(item_neighbours_dict))

N = 50

test_records_rdd = test_rdd.map(
    lambda x: ((x.split(",")[0], x.split(",")[1]), float(x.split(",")[2])))

'''test_records_rdd = test_rdd.map(
    lambda x: (x.split(",")[0], x.split(",")[1]))'''

item_based_pred_dict = test_records_rdd.map(
    lambda x: ((x[0][0], x[0][1]), predict_rating((x[0][0], x[0][1])))).collectAsMap()

'''item_based_pred_dict = test_records_rdd.map(
    lambda x: ((x[0], x[1]), predict_rating((x[0], x[1])))).collectAsMap()'''

################################################################ END OF ITEM BASED ####################################################################


train_list_of_users_rdd = train_rdd.map(lambda x: (x.split(",")[0])).distinct()
list_of_users = train_list_of_users_rdd.collect()

train_list_of_business_rdd = train_rdd.map(lambda x: (x.split(",")[1])).distinct()
list_of_business = train_list_of_business_rdd.collect()

train_user_to_number_dict = {v: k for k, v in enumerate(list_of_users)}

# print("****", train_user_to_number_dict)
train_business_to_number_dict = {v: k for k, v in enumerate(list_of_business)}
# print("&&&", train_business_to_number_dict)
user_count_start = len(train_user_to_number_dict)
bus_count_start = len(train_business_to_number_dict)

# get user details:
################################################################
userJsonRDD = sc.textFile(folder_path + "/user.json")
user_rdd_map = userJsonRDD.map(json.loads)
user_avg_rdd = user_rdd_map.map(lambda record: (record["user_id"], (record["average_stars"], record["review_count"])))

users_info_dict = user_avg_rdd.collectAsMap()

user_bus_rating_rdd = train_rdd.map(lambda x: (x.split(",")[0], (x.split(",")[1], float(x.split(",")[2]))))

# (u1, ((b1, actual_rating),(user_avg, review_count)))
intermediate_user_rdd = user_bus_rating_rdd.join(user_avg_rdd)

# ((u1,b1),(actual_rating, user_avg, user_review_count))
user_details_rdd = intermediate_user_rdd.map(lambda x: ((x[0], x[1][0][0]), (x[1][0][1], x[1][1][0], x[1][1][1])))
################################################################

# get business details
################################################################
busJsonRDD = sc.textFile(folder_path + "/business.json")
bus_rdd_map = busJsonRDD.map(json.loads)
bus_avg_rdd = bus_rdd_map.map(lambda record: (record["business_id"], (record["stars"], record["review_count"])))

bus_info_dict = bus_avg_rdd.collectAsMap()

bus_user_rating_rdd = train_rdd.map(lambda x: (x.split(",")[1], (x.split(",")[0], float(x.split(",")[2]))))

# (b1, ((u1, actual_rating),(bus_avg, bus_review_count)))
intermediate_bus_rdd = bus_user_rating_rdd.join(bus_avg_rdd)

# ((u1,b1),(actual_rating, bus_avg, bus_review_count))
bus_details_rdd = intermediate_bus_rdd.map(lambda x: ((x[1][0][0], x[0]), (x[1][0][1], x[1][1][0], x[1][1][1])))

################################################################

# train data
################################################################
# ((u1,b1),((actual_rating, user_avg, user_review_count), (actual_rating, bus_avg, bus_review_count)))
user_bus_rdd = user_details_rdd.join(bus_details_rdd)

user_details_avg_list = user_bus_rdd.map(lambda x: (1, (x[0][0], x[1][0][1], x[1][0][2], 1))) \
    .distinct() \
    .reduceByKey(lambda a, b: (a[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])) \
    .map(lambda x: (x[1][1] / x[1][3], x[1][2] / x[1][3])).collect()

train_avg_user_avg, train_avg_user_count = user_details_avg_list[0][0], user_details_avg_list[0][1]

bus_details_avg_list = user_bus_rdd.map(lambda x: (1, (x[0][1], x[1][1][1], x[1][1][2], 1))) \
    .distinct() \
    .reduceByKey(lambda a, b: (a[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])).map(
    lambda x: (x[1][1] / x[1][3], x[1][2] / x[1][3])).collect()

train_avg_bus_avg, train_avg_bus_count = bus_details_avg_list[0][0], bus_details_avg_list[0][1]

# (u1, b1, user_avg, user_review_count, bus_avg, bus_review_count, actual_rating)
train_features_and_actual_rating_rdd = user_bus_rdd.map(
    lambda x: (
        train_user_to_number_dict[x[0][0]], train_business_to_number_dict[x[0][1]], x[1][0][1], x[1][0][2], x[1][1][1],
        x[1][1][2], x[1][0][0]))

data = np.array(train_features_and_actual_rating_rdd.collect())
X_axis_train = data[:, :6]
Y_axis_train = data[:, 6:]

model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                         colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                         max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
                         n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                         silent=True, subsample=1)
model.fit(X_axis_train, Y_axis_train)
################################################################

# read test data
################################################################


test_users_list_rdd = test_rdd.map(lambda x: (x.split(",")[0])).distinct()
test_bus_list_rdd = test_rdd.map(lambda x: (x.split(",")[1])).distinct()

test_new_users_rdd = test_users_list_rdd.subtract(train_list_of_users_rdd.intersection(test_users_list_rdd))
test_new_bus_rdd = test_bus_list_rdd.subtract(train_list_of_business_rdd.intersection(test_bus_list_rdd))

new_user_to_number_dict = {v: k for k, v in enumerate(test_new_users_rdd.collect(), user_count_start)}
new_bus_to_number_dict = {v: k for k, v in enumerate(test_new_bus_rdd.collect(), bus_count_start)}


def find_feature_for_test(x):
    test_user = x[0]
    test_bus = x[1]
    # test_actual_rating = x[2]

    if test_user in train_user_to_number_dict:
        user_num = train_user_to_number_dict[test_user]
    elif test_user in new_user_to_number_dict:
        user_num = new_user_to_number_dict[test_user]
    else:
        user_num = -1

    if test_bus in train_business_to_number_dict:
        bus_num = train_business_to_number_dict[test_bus]
    elif test_user in new_bus_to_number_dict:
        bus_num = new_bus_to_number_dict[test_user]
    else:
        bus_num = -1

    if test_user in users_info_dict:
        test_user_avg, test_user_review_count = users_info_dict[test_user]
    else:
        test_user_avg, test_user_review_count = train_avg_user_avg, train_avg_user_count

    if test_bus in bus_info_dict:
        test_bus_avg, test_bus_review_count = bus_info_dict[test_bus]
    else:
        test_bus_avg, test_bus_review_count = train_avg_bus_avg, train_avg_bus_count

    return user_num, bus_num, test_user_avg, test_user_review_count, test_bus_avg, test_bus_review_count  # , test_actual_rating


user_business_pair = test_rdd.map(lambda x: [x.split(",")[0], x.split(",")[1]]).collect()

test_features_and_actual_rating_rdd = test_rdd.map(
    lambda x: (x.split(",")[0], x.split(",")[1])).map(
    lambda x: find_feature_for_test(x)).collect()

data1 = np.array(test_features_and_actual_rating_rdd)
X_axis_test = data1[:, :6]
# Y_axis_test = data1[:, 6:]
################################################################

# predict
################################################################
pred = model.predict(data=X_axis_test)

for i in range(len(user_business_pair)):
    user_business_pair[i].append(pred[i])

user_business_prediction_rdd = sc.parallelize(user_business_pair)
model_based_pred_dict = user_business_prediction_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()

################################################################ END OF MODEL BASED ####################################################################


################################################################ START OF HYBRID ####################################################################

alpha = 0.1


def find_squares_diff(perdiction_tupp):
    user_id = perdiction_tupp[0]
    business_id = perdiction_tupp[1]
    predicted_rating = perdiction_tupp[2]
    actual_rating = actual_rating_dict[(user_id, business_id)]
    return math.pow((predicted_rating - actual_rating), 2)


def hybrid_predict(x):
    return (alpha * item_based_pred_dict[x]) + ((1 - alpha) * model_based_pred_dict[x])


hybrid_predictions_rdd = test_rdd.map(lambda x: (x.split(",")[0], x.split(",")[1])).map(
    lambda x: (x[0], x[1], hybrid_predict(x)))
hybrid_predictions = hybrid_predictions_rdd.collect()

with open(output_file_path, "w") as f:
    f.write("user_id, business_id, prediction\n")
    for tupp in hybrid_predictions:
        f.write(tupp[0] + "," + tupp[1] + "," + str(tupp[2]) + "\n")

actual_rating_dict = test_records_rdd.collectAsMap()

avg_list = hybrid_predictions_rdd.map(lambda x: (1, (find_squares_diff(x), 1))).reduceByKey(
    lambda a, b: (a[0] + b[0], a[1] + b[1])).map(lambda x: x[1][0] / x[1][1]).collect()
rmse = math.sqrt(avg_list[0])
print("hybrid rmse: ", rmse)
print("Duration: ", time.time() - start_time)

################################################################ END OF HYBRID ####################################################################
