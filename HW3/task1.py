from pyspark import SparkContext
import sys
import time
import itertools


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
    primes_list = [4421,4423,4441,4447,4451,4457,4463,4481,4483,4493,4507,4513,4517,4519,4523,4547,4549,4561,4567,4583,
                   4591,4597,4603,4621,4637,4639,4643,4649,4651,4657,4663,4673,4679,4691,4703,4721,4723,4729,4733,4751,
                   4759,4783,4787,4789,4793,4799,4801,4813,4817,4831,4861,4871,4877,4889,4903,4909,4919,4931,4933,4937,
                   4943,4951,4957,4967,4969,4973,4987,4993,4999,5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087]
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


if __name__ == "__main__":
    start_time = time.time()
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("ERROR")
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    rdd = sc.textFile(input_file_path)
    header = rdd.first()

    input_rdd = rdd.filter(lambda record: record != header) \
        .map(lambda line: (line.split(",")[1], [line.split(",")[0]])) \
        .reduceByKey(lambda a, b: a + b)
    business_users_dict = input_rdd.collectAsMap()

    list_of_users = rdd.filter(lambda record: record != header).map(lambda x: (x.split(",")[0], [x.split(",")[0]])) \
        .reduceByKey(lambda a, b: a + b).keys().collect()
    user_index_map = {v: k for k, v in enumerate(list_of_users)}

    # list_of_users = input_rdd.flatMap(lambda business_users_tuple: business_users_tuple[1]).distinct().count()
    # print(input_rdd.collect())
    num_of_hashes = 28
    b = 14
    band_start = 0

    signature_rdd = input_rdd.map(
        lambda business_users_map: (get_signature(business_users_map, user_index_map, num_of_hashes)))

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
    # print("distinct_candidates_rdd: ", distinct_candidates_rdd.count())
    lines_rdd = distinct_candidates_rdd.map(lambda pair: (pair[0], pair[1], find_jaccard_similarity(pair, business_users_dict)))\
        .sortBy(lambda x: (x[0], x[1])) \
        .filter(lambda x: x[2] >= 0.5)
    '''similar_business_rdd = lines_rdd.map(lambda x: (x[0], x[1]))'''
    # print("count: ", distinct_candidates_rdd.count())
    lines = lines_rdd.collect()

    # print("lines: ", len(lines))
    with open(output_file_path, "w") as f:
        header = "business_id_1, business_id_2, similarity\n"
        f.write(header)
        for line in lines:
            f.write(str(line[0]) + "," + str(line[1]) + "," + str(line[2]))
            f.write("\n")

    print("Duration: ", time.time()-start_time)

    '''ground_truth = sc.textFile("/Users/ramya/Desktop/DM/HW3/data/pure_jaccard_similarity.csv")
    ground_truth_rdd = ground_truth.map(lambda record: (record.split(",")[0], record.split(",")[1]))
    true_positive_rdd = ground_truth_rdd.intersection(similar_business_rdd)
    true_positive_count = true_positive_rdd.count()
    false_positive_count = similar_business_rdd.subtract(true_positive_rdd).count()
    false_negative_count = ground_truth_rdd.subtract(true_positive_rdd).count()

    print("precision: ", true_positive_count / (true_positive_count + false_positive_count))
    print("recall: ", true_positive_count / (true_positive_count + false_negative_count))'''

