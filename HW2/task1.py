from pyspark import SparkContext
import sys
import collections
import itertools
import time


def get_current_candidate_items(index, previous_frequent_items, size):
    all_candidates = set()
    true_candidates = set()
    if size == 2:
        for combination in itertools.combinations(previous_frequent_items, size):
            all_candidates.add(combination)
        true_candidates = all_candidates
    else:
        for i in range(0, len(previous_frequent_items)):
            for j in range(i + 1, len(previous_frequent_items)):
                unioned_set = set(previous_frequent_items[i]).union(set(previous_frequent_items[j]))
                if len(unioned_set) == size:
                    unioned_tuple = tuple(sorted(unioned_set))
                    all_candidates.add(unioned_tuple)
        for candidate in all_candidates:
            all_combinations = itertools.combinations(candidate, size - 1)
            for combination in all_combinations:
                if combination in previous_frequent_items:
                    true_candidates.add(candidate)
    return sorted(list(true_candidates))


def get_item_counts(current_candidate_items, baskets):
    item_counts = collections.defaultdict(int)
    for candidate_item in current_candidate_items:
        for basket in baskets:
            if set(candidate_item).issubset(basket[1]):
                item_counts[candidate_item] += 1
    return item_counts


def find_frequent_items(item_counts, sample_support_threshold):
    result = []
    for item in item_counts:
        if item_counts[item] >= sample_support_threshold:
            result.append(item)
    return sorted(result)


def apriori(index, iterator):
    baskets = list(iterator)
    # print("index: baskets: ", index, baskets)
    sample_support_threshold = (len(baskets) / total_number_of_baskets) * support
    # print("index: sample_support_threshold: ", index, sample_support_threshold)
    item_counts = collections.defaultdict(int)
    for basket in baskets:
        items = basket[1]
        for item in items:
            item_counts[item] += 1
    frequent_items = []
    single_frequent_items = find_frequent_items(item_counts, sample_support_threshold)
    # print("index: single frequent_items: ", index, frequent_items)
    for item in single_frequent_items:
        frequent_items.append((item,))
    previous_frequent_items = single_frequent_items
    size = 2
    while True:
        current_candidate_items = get_current_candidate_items(index, previous_frequent_items, size)
        item_counts = get_item_counts(current_candidate_items, baskets)
        current_frequent_items = find_frequent_items(item_counts, sample_support_threshold)
        if len(current_frequent_items) > 0:
            frequent_items += current_frequent_items
        else:
            break
        previous_frequent_items = current_frequent_items
        size += 1
    # print("index: frequent items: ", index, frequent_items)
    return frequent_items


def count_support_for_candidates(index, iterator):
    candidate_counts = collections.defaultdict(int)
    baskets = list(iterator)
    for candidate in candidates:
        for basket in baskets:
            if set(candidate).issubset(basket[1]):
                candidate_counts[candidate] += 1
    # print("candidate_counts.items(): ", candidate_counts.items())
    return candidate_counts.items()


def get_sorted_candidate_counts(candidates):
    len_to_items_dic = {}
    for candidate in candidates:
        if isinstance(candidate, str):
            if 1 not in len_to_items_dic:
                len_to_items_dic[1] = list()
                len_to_items_dic[1].append(candidate)
            else:
                len_to_items_dic[1].append(candidate)
        else:
            if len(candidate) not in len_to_items_dic:
                len_to_items_dic[len(candidate)] = list()
                len_to_items_dic[len(candidate)].append(candidate)
            else:
                len_to_items_dic[len(candidate)].append(candidate)
    # print("len_to_items_dic: ", len_to_items_dic)
    for k in len_to_items_dic:
        len_to_items_dic[k] = sorted(len_to_items_dic[k])
    output_list = sorted(len_to_items_dic.items())
    return output_list


def get_sorted_frequent_items_counts(frequent_items):
    frequent_item_dict = {}
    for item in frequent_items:
        if len(item) not in frequent_item_dict:
            frequent_item_dict[len(item)] = list()
            frequent_item_dict[len(item)].append(item)
        else:
            frequent_item_dict[len(item)].append(item)
    # print("len_to_items_dic: ", len_to_items_dic)
    for k in frequent_item_dict:
        frequent_item_dict[k] = sorted(frequent_item_dict[k])
    output_frequent_list = sorted(frequent_item_dict.items())
    return output_frequent_list


def write_to_file(output_file_path, output_list, output_frequent_list):
    with open(output_file_path, "w") as f:
        output_str = "Candidates:\n"
        for i in range(0, len(output_list)):
            list_of_tuples = output_list[i][1]
            if i == 0:
                for k in list_of_tuples:
                    output_str += "('" + k[0] + "'),"
                output_str = output_str.strip(",")
            else:
                output_str += ",".join(str(k) for k in list_of_tuples)
            output_str += "\n\n"

        output_str += "Frequent Itemsets:\n"
        for i in range(0, len(output_frequent_list)):
            list_of_tuples = output_frequent_list[i][1]
            if i == 0:
                for k in list_of_tuples:
                    output_str += "('" + k[0] + "'),"
                output_str = output_str.strip(",")

            else:
                output_str += ",".join(str(k) for k in list_of_tuples)
            output_str += "\n\n"
        output_str = output_str.strip("\n\n")
        f.write(output_str)


if __name__ == "__main__":
    start_time = time.time()
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("ERROR")
    case_num = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    input_rdd = sc.textFile(input_file_path)

    if case_num == 1:
        baskets_rdd = input_rdd.filter(lambda x: not str(x).startswith("user_id")) \
            .map(lambda line: (line.split(",")[0], [line.split(",")[1]])) \
            .reduceByKey(lambda a, b: a + b) \
            .map(lambda a: (a[0], set(a[1])))
    elif case_num == 2:
        baskets_rdd = input_rdd.filter(lambda x: not str(x).startswith("user_id")) \
            .map(lambda line: (line.split(",")[1], [line.split(",")[0]])) \
            .reduceByKey(lambda a, b: a + b) \
            .map(lambda a: (a[0], set(a[1])))

    # print("no of partitions: ", baskets_rdd.getNumPartitions())
    total_number_of_baskets = baskets_rdd.count()
    # print("total_number_of_baskets: ", total_number_of_baskets)

    candidates = baskets_rdd.mapPartitionsWithIndex(apriori) \
        .map(lambda a: (a, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .keys() \
        .collect()

    # print("candidates: ", candidates)
    # print("time taken until phase1: ", time.time() - start_time)

    output_list = get_sorted_candidate_counts(candidates)
    # print("output_list: ", output_list)

    frequent_items = baskets_rdd.mapPartitionsWithIndex(count_support_for_candidates) \
        .reduceByKey(lambda a, b: a + b) \
        .filter(lambda x: x[1] >= support) \
        .keys() \
        .collect()
    # print("frequent_items: ", frequent_items)

    output_frequent_list = get_sorted_frequent_items_counts(frequent_items)
    # print("output_frequent_list: ", output_frequent_list)

    write_to_file(output_file_path, output_list, output_frequent_list)

    print("Duration:", time.time() - start_time)
