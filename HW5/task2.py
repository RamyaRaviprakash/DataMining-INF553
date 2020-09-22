import time
import sys
import binascii
from blackbox import BlackBox
import math
from statistics import median
from statistics import mean

hash_func_list = [(4421, 4591), (4423, 4597), (4441, 4603), (4447, 4621), (4451, 4637), (4457, 4639), (4463, 4643),
                  (4481, 4649),
                  (4483, 4651), (4493, 4657), (4507, 4663), (4513, 4673), (4517, 4679), (4519, 4691), (4523, 4703),
                  (4547, 4721)]
                  # , (4549, 4723), (4561, 4729), (4567, 4733), (4583, 4751)]


def apply_hash(tup, user):
    a, b = tup
    x = int(binascii.hexlify(user.encode('utf8')), 16)
    return ((a * x) + b) % 69997


def myhashs(user):
    result = []
    for tup in hash_func_list:
        result.append(apply_hash(tup, user))
    return result


def process_batch(stream_users):
    hash_rs = [0] * len(hash_func_list)

    for user in stream_users:
        hash_values = myhashs(user)
        for i in range(len(hash_values)):
            if hash_values[i] == 0:
                continue
            binary_str = bin(hash_values[i])
            zeros_stripped_binary_str = binary_str.rstrip('0')
            num_of_trailing_zeros = len(binary_str) - len(zeros_stripped_binary_str)
            if num_of_trailing_zeros > hash_rs[i]:
                hash_rs[i] = num_of_trailing_zeros

    hash_estimates = [math.pow(2, r) for r in hash_rs]
    means_list = []
    num_of_hashes_per_group = 4
    start = 0
    while start < len(hash_estimates):
        group_mean = mean(hash_estimates[start: start + num_of_hashes_per_group])
        means_list.append(group_mean)
        start = start + num_of_hashes_per_group
    return math.floor(median(means_list))


if __name__ == "__main__":
    start_time = time.time()
    bx = BlackBox()
    input_file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_name = sys.argv[4]

    actual_list = []
    estimate_list = []
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_file_name, stream_size)
        # print("stream_users: ", stream_users)
        estimate_num_of_distinct_users = process_batch(stream_users)
        estimate_list.append(estimate_num_of_distinct_users)

        actual_num_of_distinct_users = len(set(stream_users))
        actual_list.append(actual_num_of_distinct_users)

    with open(output_file_name, "w") as f:
        f.write("Time,Ground Truth,Estimation\n")
        for i in range(len(actual_list)):
            f.write(str(i)+","+str(actual_list[i])+","+str(estimate_list[i])+"\n")

    print("Duration: ", time.time()-start_time)
    print("final result: ", sum(estimate_list)/sum(actual_list))


