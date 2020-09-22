import time
import sys
from blackbox import BlackBox
import random

sample = []
user_seq_num = 0


def sample_stream(stream_users):
    global sample
    global user_seq_num
    if len(sample) == 0:
        sample = stream_users
        user_seq_num = stream_size
    else:
        for user in stream_users:
            user_seq_num += 1
            r = random.randint(0, 100000) % user_seq_num
            if r < stream_size:  # selected
                index_to_replace = random.randint(0, 100000) % stream_size
                sample[index_to_replace] = user


if __name__ == "__main__":
    start_time = time.time()
    bx = BlackBox()
    input_file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_name = sys.argv[4]
    random.seed(553)

    output_list = []
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_file_name, stream_size)
        sample_stream(stream_users)
        output_list.append([str(user_seq_num), sample[0], sample[20], sample[40], sample[60], sample[80]])

    with open(output_file_name, "w") as f:
        f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
        for tup in output_list:
            f.write(",".join(tup) + "\n")
    print("Duration: ", time.time()-start_time)
