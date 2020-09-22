import sys
from blackbox import BlackBox
import binascii
import time

sample_set = set()
bit_arr = [0] * 69997
hash_func_list = [(4421, 4591), (4423, 4597), (4441, 4603), (4447, 4621),(4451, 4637), (4457, 4639), (4463, 4643), (4481, 4649)]

# (4483, 4651), (4493, 4657), (4507, 4663), (4513, 4673), (4517, 4679), (4519, 4691), (4523, 4703), (4547, 4721)]
# [4421,4423,4441,4447,4451,4457,4463,4481,4483,4493,4507,4513,4517,4519,4523,4547,4549,4561,4567,4583,
# 4591,4597,4603,4621,4637,4639,4643,4649,4651,4657,4663,4673,4679,4691,4703,4721,4723,4729,4733,4751,


def apply_hash(tup, user):
    a, b = tup
    x = int(binascii.hexlify(user.encode('utf8')), 16)
    return ((a * x) + b) % 69997
    # return ((a * x) + b) % 10


def myhashs(user):
    result = []
    for tup in hash_func_list:
        result.append(apply_hash(tup, user))
    return result


def process_batch(stream_users):

    global sample_set
    global bit_arr
    TN = 0
    FP = 0

    for user in stream_users:
        list_of_indices = myhashs(user)
        flag = True
        for index in list_of_indices:
            if bit_arr[index] == 0:
                flag = False
                bit_arr[index] = 1
        if flag == False:
            TN += 1
        elif user not in sample_set:
            FP += 1
        sample_set.add(user)
    if (FP + TN) == 0:
        FPR = 0
    else:
        FPR = FP / (FP + TN)
    return FPR


if __name__ == "__main__":
    start_time = time.time()
    bx = BlackBox()
    input_file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_name = sys.argv[4]

    FPR_list = []
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_file_name, stream_size)
        # print("stream_users: ", stream_users)
        FPR = process_batch(stream_users)
        FPR_list.append(FPR)
    with open(output_file_name, "w") as f:
        f.write("Time,FPR\n")
        for i in range(len(FPR_list)):
            f.write(str(i) + "," + str(FPR_list[i]) + "\n")

    print("Duration: ", time.time()-start_time)
