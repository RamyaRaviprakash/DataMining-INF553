from pyspark import SparkContext
import xgboost as xgb
import numpy as np
import time
import sys
import json
from sklearn.metrics import mean_squared_error

start_time = time.time()
sc = SparkContext('local[*]', 'task2_2')
sc.setLogLevel("ERROR")
folder_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]

train_rdd_with_header = sc.textFile(folder_path + "/yelp_train.csv")
header = train_rdd_with_header.first()
train_rdd = train_rdd_with_header.filter(lambda x: x != header)

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
# l = user_bus_rdd.take(1)
# print("****", l)
# print("^^^^", train_user_to_number_dict[l[0][0][0]], train_business_to_number_dict[l[0][0][1]])

train_features_and_actual_rating_rdd = user_bus_rdd.map(
    lambda x: (
        train_user_to_number_dict[x[0][0]], train_business_to_number_dict[x[0][1]], x[1][0][1], x[1][0][2], x[1][1][1],
        x[1][1][2], x[1][0][0]))

data = np.array(train_features_and_actual_rating_rdd.collect())
X_axis_train = data[:, :6]
Y_axis_train = data[:, 6:]
print("X_axis_train: ", X_axis_train)
print("Y_axis_train: ", Y_axis_train)

model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                         colsample_bytree=1, gamma=0, learning_rate=0.2, max_delta_step=0,
                         max_depth=6, min_child_weight=1, missing=None, n_estimators=100,
                         n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                         silent=True, subsample=1)
model.fit(X_axis_train, Y_axis_train)
print("model: ", model)
################################################################

# read test data
################################################################
test_rdd_with_header = sc.textFile(test_file_path)
test_header = test_rdd_with_header.first()
test_rdd = test_rdd_with_header.filter(lambda x: x != test_header)

test_users_list_rdd = test_rdd.map(lambda x: (x.split(",")[0])).distinct()
test_bus_list_rdd = test_rdd.map(lambda x: (x.split(",")[1])).distinct()

test_new_users_rdd = test_users_list_rdd.subtract(train_list_of_users_rdd.intersection(test_users_list_rdd))
test_new_bus_rdd = test_bus_list_rdd.subtract(train_list_of_business_rdd.intersection(test_bus_list_rdd))

new_user_to_number_dict = {v: k for k, v in enumerate(test_new_users_rdd.collect(), user_count_start)}
new_bus_to_number_dict = {v: k for k, v in enumerate(test_new_bus_rdd.collect(), bus_count_start)}


def find_feature_for_test(x):
    test_user = x[0]
    test_bus = x[1]
    test_actual_rating = x[2]

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

    return user_num, bus_num, test_user_avg, test_user_review_count, test_bus_avg, test_bus_review_count, test_actual_rating


user_business_pair = test_rdd.map(lambda x: [x.split(",")[0], x.split(",")[1]]).collect()

test_features_and_actual_rating_rdd = test_rdd.map(
    lambda x: (x.split(",")[0], x.split(",")[1], float(x.split(",")[2]))).map(
    lambda x: find_feature_for_test(x)).collect()

'''test_features_and_actual_rating_rdd = test_rdd.map(
    lambda x: (x.split(",")[0], x.split(",")[1])).map(
    lambda x: find_feature_for_test(x)).collect()'''

data1 = np.array(test_features_and_actual_rating_rdd)
X_axis_test = data1[:, :6]
Y_axis_test = data1[:, 6:]
print("X_axis_test: ", X_axis_test)
# print("Y_axis_test: ", Y_axis_test)

################################################################

# predict
################################################################
pred = model.predict(data=X_axis_test)
print("pred: ", pred)

for i in range(len(user_business_pair)):
    user_business_pair[i].append(pred[i])

with open(output_file_path, "w") as f:
    f.write("user_id, business_id, prediction\n")
    for line in user_business_pair:
        f.write(str(line[0]) + "," + str(line[1]) + "," + str(line[2]) + "\n")

print("Duration: ", time.time() - start_time)


rmse = np.sqrt(mean_squared_error(Y_axis_test, pred))
print("rmse: ", rmse)
################################################################
