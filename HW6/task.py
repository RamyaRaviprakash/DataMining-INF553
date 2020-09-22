import numpy as np
import sys
import time
from sklearn.cluster import KMeans
#from sklearn.metrics.cluster import normalized_mutual_info_score


def get_cluster_dict(batch_list, cluster_indices_list):
    clusters_dict = {}
    for i in range(len(batch_list)):
        ind = cluster_indices_list[i]
        value = batch_list[i]
        if ind in clusters_dict:
            clusters_dict[ind].append(value)
        else:
            clusters_dict[ind] = list()
            clusters_dict[ind].append(value)
    return clusters_dict


def update_rs(clusters_dict):
    RS.clear()
    for i in range(len(clusters_dict)):
        if len(clusters_dict[i]) == 1:
            RS.append(clusters_dict[i][0])


def update_cs(clusters_dict):
    global CS
    global CS_N
    global CS_SUM
    global CS_SUM_SQ
    global CS_VAR
    global CS_STD_DEV
    global CS_CENTROID
    for i in range(len(clusters_dict)):
        if len(clusters_dict[i]) != 1:
            CS.append(clusters_dict[i])
            CS_N = np.append(CS_N, np.array([[len(clusters_dict[i])]]), axis=0)

            n = np.array([0] * num_of_dimensions)
            p = np.array([0] * num_of_dimensions)
            for x in clusters_dict[i]:
                n = n + np.array(x)
                p = p + np.array([y * y for y in x])
            CS_SUM = np.append(CS_SUM, np.array([n]), axis=0)
            CS_SUM_SQ = np.append(CS_SUM_SQ, np.array([p]), axis=0)
    if CS_N.size != 0:
        CS_VAR = (CS_SUM_SQ / CS_N) - np.square(CS_SUM / CS_N)
        CS_STD_DEV = np.sqrt(CS_VAR)
        CS_CENTROID = CS_SUM / CS_N


def check_if_mergable(centroid_i, centroid_j, std_dev_i, std_dev_j):
    numerator = np.square(centroid_i - centroid_j)
    denominator = np.square(std_dev_i) * np.square(std_dev_j)
    div = numerator / denominator
    MD = np.sqrt(np.sum(div))

    return MD < (2 * np.sqrt(num_of_dimensions))


def merge_two_cs_cluster(i, j):
    global CS
    global CS_N
    global CS_SUM
    global CS_SUM_SQ
    global CS_VAR
    global CS_STD_DEV
    global CS_CENTROID

    CS[i] = CS[i] + CS[j]
    CS_N[i] = CS_N[i] + CS_N[j]
    CS_SUM[i] = CS_SUM[i] + CS_SUM[j]
    CS_SUM_SQ[i] = CS_SUM_SQ[i] + CS_SUM_SQ[j]

    del CS[j]
    CS_N = np.delete(CS_N, j, 0)
    CS_SUM = np.delete(CS_SUM, j, 0)
    CS_SUM_SQ = np.delete(CS_SUM_SQ, j, 0)

    CS_VAR = (CS_SUM_SQ / CS_N) - np.square(CS_SUM / CS_N)
    CS_STD_DEV = np.sqrt(CS_VAR)
    CS_CENTROID = CS_SUM / CS_N


def merge_cs_cluster_to_ds_cluster(i, j):
    global DS_N
    global DS_SUM
    global DS_SUM_SQ
    global DS_VAR
    global DS_STD_DEV
    global DS_CENTROID
    global point_to_cluster_list
    global points_to_index_dict
    global CS_VAR
    global CS_STD_DEV
    global CS_CENTROID

    for x in CS[i]:
        point_to_cluster_list.append((points_to_index_dict[tuple(x)], j))

    DS_N[j] = DS_N[j] + CS_N[i]
    DS_SUM[j] = DS_SUM[j] + CS_SUM[i]
    DS_SUM_SQ[j] = DS_SUM_SQ[j] + CS_SUM_SQ[i]

    DS_VAR = (DS_SUM_SQ / DS_N) - np.square((DS_SUM / DS_N))
    # print("DS_VAR ", DS_VAR)
    DS_STD_DEV = np.sqrt(DS_VAR)
    DS_CENTROID = DS_SUM / DS_N


# def print_cs(m, CS):
#     print("m: ", m)
#     for x in CS:
#         for y in x:
#             print(points_to_index_dict[tuple(y)])


if __name__ == "__main__":
    start_time = time.time()
    input_file_path = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file_path = sys.argv[3]

    raw_data = np.loadtxt(input_file_path, delimiter=",", dtype='float')
    points = raw_data[:, 0:1]
    # print("points: ", point[0])
    data = raw_data[:, 2:]
    num_of_dimensions = len(data[0])
    # print("num_of_dimensions: ", num_of_dimensions)
    # {(d1, d2, d3...dn): index}
    points_to_index_dict = {}
    raw_data_list = raw_data.tolist()
    for p in raw_data_list:
        points_to_index_dict[tuple(p[2:])] = int(p[0])
    # print("points_to_index_dict ", points_to_index_dict)
    # print(data)
    np.random.shuffle(data)
    # print("point: value: ", points[0], points_dict[list(points[0])[0]])

    # print(data[0:20])
    # print(len(data))
    batches = []
    start = 0
    load = int(0.2 * len(data))
    end = load
    for i in range(5):
        if i == 4:
            batches.append(data[start:])
        else:
            batches.append(data[start:end])
        start = end
        end = start + load

    num_of_points_in_ds = [0] * 5
    num_of_clusters_in_cs = [0] * 5
    num_of_points_in_cs = [0] * 5
    num_of_points_in_rs = [0] * 5

    RS = []
    CS = []
    CS_N = np.empty((0, 1), int)
    CS_SUM = np.empty((0, num_of_dimensions), float)
    CS_SUM_SQ = np.empty((0, num_of_dimensions), float)

    CS_VAR = None
    CS_STD_DEV = None
    CS_CENTROID = None

    five_times_n_cluster = 5 * n_cluster

    cluster_indices = KMeans(n_clusters=five_times_n_cluster) \
        .fit_predict(batches[0])
    batch_list = batches[0].tolist()
    cluster_indices_list = cluster_indices.tolist()
    clusters_dict = get_cluster_dict(batch_list, cluster_indices_list)
    update_rs(clusters_dict)
    # print(len(RS))

    # print("RS: ", RS)
    # remove rs from initial set of points
    for x in RS:
        batch_list.remove(x)

    # step 4 [(pt index, ds cluster number)]
    point_to_cluster_list = []
    DS_N = np.array([[0]] * n_cluster)
    DS_SUM = np.array([[0.0] * num_of_dimensions] * n_cluster)
    DS_SUM_SQ = np.array([[0.0] * num_of_dimensions] * n_cluster)

    ds_cluster = KMeans(n_clusters=n_cluster) \
        .fit_predict(np.array(batch_list))
    ds_cluster_list = ds_cluster.tolist()

    # step 5
    for i in range(len(batch_list)):
        cluster_num = ds_cluster_list[i]
        point = batch_list[i]
        point_to_cluster_list.append((points_to_index_dict[tuple(point)], cluster_num))
        DS_N[cluster_num] += 1
        DS_SUM[cluster_num] += np.array(point)
        DS_SUM_SQ[cluster_num] += np.array([x * x for x in point])

    # step 6
    # print("len(point_to_cluster_list)", len(point_to_cluster_list))
    # print("len(set(point_to_cluster_list))", len(set(point_to_cluster_list)))
    result = KMeans(n_clusters=min(len(RS), five_times_n_cluster)) \
        .fit_predict(np.array(RS))
    clusters_dict = get_cluster_dict(RS, result.tolist())
    # print("clusters_dict: ", clusters_dict)

    update_rs(clusters_dict)
    update_cs(clusters_dict)
    # print_cs(1, CS)

    # update output
    num_of_points_in_ds[0] = int(np.sum(DS_N))
    num_of_clusters_in_cs[0] = len(CS_N)
    num_of_points_in_cs[0] = int(np.sum(CS_N))
    num_of_points_in_rs[0] = len(RS)

    # print(num_of_points_in_ds)
    # print(num_of_clusters_in_cs)
    # print(num_of_points_in_cs)
    # print(num_of_points_in_rs)

    DS_VAR = (DS_SUM_SQ / DS_N) - np.square(DS_SUM / DS_N)
    DS_STD_DEV = np.sqrt(DS_VAR)
    DS_CENTROID = DS_SUM / DS_N

    # step 7 - 12
    for i in range(1, len(batches)):
        # print("i: ", i)
        batch = batches[i]
        batch_list = batch.tolist()
        # print("len(batch_list)", len(batch_list))
        for p in batch_list:
            # print("I'm here 1")
            X = np.array([p] * n_cluster)
            ans1 = np.square((X - DS_CENTROID) / DS_STD_DEV)
            pt_to_cluster_distances = np.sqrt(np.sum(ans1, axis=1, keepdims=True))
            # print("pt_to_cluster_distances ", pt_to_cluster_distances)
            min_cluster_distance = np.min(pt_to_cluster_distances)

            # step 8 - add to DS
            if min_cluster_distance < (2 * np.sqrt(num_of_dimensions)):  # add point to DS
                nearest_cluster_index = np.where(pt_to_cluster_distances == min_cluster_distance)[0].tolist()[0]
                point_to_cluster_list.append((points_to_index_dict[tuple(p)], nearest_cluster_index))
                DS_N[nearest_cluster_index] += 1
                DS_SUM[nearest_cluster_index] += np.array(p)
                DS_SUM_SQ[nearest_cluster_index] += np.array([x * x for x in p])

                DS_VAR = (DS_SUM_SQ / DS_N) - np.square(DS_SUM / DS_N)
                DS_STD_DEV = np.sqrt(DS_VAR)
                DS_CENTROID = DS_SUM / DS_N

            # step 9 - add to CS
            elif CS_N.size != 0:  # at least one CS cluster exists
                num_of_cs_clusters = CS_N.size
                Y = np.array([p] * num_of_cs_clusters)
                ans1 = np.square((Y - CS_CENTROID) / CS_STD_DEV)
                pt_to_cluster_distances = np.sqrt(np.sum(ans1, axis=1, keepdims=True))
                min_cluster_distance = np.min(pt_to_cluster_distances)

                if min_cluster_distance < (2 * np.sqrt(num_of_dimensions)):
                    nearest_cluster_index = np.where(pt_to_cluster_distances == min_cluster_distance)[0].tolist()[0]
                    CS[nearest_cluster_index].append(p)
                    CS_N[nearest_cluster_index] += 1
                    CS_SUM[nearest_cluster_index] += np.array(p)
                    CS_SUM_SQ[nearest_cluster_index] += np.array([x * x for x in p])

                    CS_VAR = (CS_SUM_SQ / CS_N) - np.square(CS_SUM / CS_N)
                    CS_STD_DEV = np.sqrt(CS_VAR)
                    CS_CENTROID = CS_SUM / CS_N
                else:
                    RS.append(p)

            # step 10 - add to RS
            else:
                RS.append(p)
        # print("total1 ds, cs, rs ", np.sum(DS_N), np.sum(CS_N), len(RS))
        # print_cs(2, CS)
        # print("I'm here 2")
        # step 11
        result = KMeans(n_clusters=min(len(RS), five_times_n_cluster)) \
            .fit_predict(np.array(RS))
        clusters_dict = get_cluster_dict(RS, result.tolist())
        update_rs(clusters_dict)
        update_cs(clusters_dict)
        # print_cs(3, CS)
        # print("I'm here 3")
        # step 12 - merge CS clusters
        # print("before merging, CS_N.size: ", CS_N.size)
        # print("total2 ds, cs, rs ", np.sum(DS_N), np.sum(CS_N), len(RS))
        # print("CS_N.size ", CS_N.size)
        if CS_N.size > 1:
            k = 0
            n = len(CS_N)
            while k < n:
                # print("k: ", k)
                j = 0
                while j < n:
                    mergable = False
                    # print("j: ", j)
                    # print("n: ", n)
                    if k != j:
                        # print("k not equal to j")
                        mergable = check_if_mergable(CS_CENTROID[k], CS_CENTROID[j], CS_STD_DEV[k], CS_STD_DEV[j])
                        if mergable:
                            # print("merging and breaking")
                            merge_two_cs_cluster(k, j)  # modify existing CS_N, CS_SUM...
                            break
                    # print("incrementing j")
                    j += 1
                    # print("j value incremented to", j)
                if mergable:
                    # print("resetting k", k, j)
                    k = 0
                    n = len(CS_N)
                else:
                    # print("incrementing k", k)
                    k += 1
        # print("after merging, CS_N.size: ", CS_N.size)
        # print_cs(4, CS)
        # print("total3 ds, cs, rs ", np.sum(DS_N), np.sum(CS_N), len(RS))
        # print("len(point_to_cluster_list)", len(point_to_cluster_list))
        # print("len(set(point_to_cluster_list))", len(set(point_to_cluster_list)))

        if i == 4:  # last batch - merge CS cluster into DS cluster
            cs_cluster_indices_merged_to_ds = []
            for k in range(len(CS_N)):
                for j in range(len(DS_N)):
                    mergable = check_if_mergable(CS_CENTROID[k], DS_CENTROID[j], CS_STD_DEV[k], DS_STD_DEV[j])
                    if mergable:
                        merge_cs_cluster_to_ds_cluster(k, j)
                        cs_cluster_indices_merged_to_ds.append(k)
                        break

            for cs_index in cs_cluster_indices_merged_to_ds:
                # print("removing cs cluster which got merged to ds")
                del CS[cs_index]
                CS_N = np.delete(CS_N, cs_index, 0)
                CS_SUM = np.delete(CS_SUM, cs_index, 0)
                CS_SUM_SQ = np.delete(CS_SUM_SQ, cs_index, 0)
                CS_VAR = np.delete(CS_VAR, cs_index, 0)
                CS_STD_DEV = np.delete(CS_STD_DEV, cs_index, 0)
                CS_CENTROID = np.delete(CS_CENTROID, cs_index, 0)

        # update output
        num_of_points_in_ds[i] = int(np.sum(DS_N))
        num_of_clusters_in_cs[i] = len(CS_N)
        num_of_points_in_cs[i] = int(np.sum(CS_N))
        num_of_points_in_rs[i] = len(RS)

        # print(num_of_points_in_ds)
        # print(num_of_clusters_in_cs)
        # print(num_of_points_in_cs)
        # print(num_of_points_in_rs)

    for cluster in CS:
        for p in cluster:
            point_to_cluster_list.append((points_to_index_dict[tuple(p)], -1))

    for p in RS:
        point_to_cluster_list.append((points_to_index_dict[tuple(p)], -1))
    point_to_cluster_list.sort(key=lambda x: x[0])
    # print("point_to_cluster_list ", point_to_cluster_list)
    with open(output_file_path, "w") as f:
        f.write("The intermediate results:\n")
        for i in range(5):
            str_output = "Round " + str(i+1) + ": " + str(num_of_points_in_ds[i]) + "," + str(num_of_clusters_in_cs[i]) \
                         + "," + str(num_of_points_in_cs[i]) + "," + str(num_of_points_in_rs[i]) + "\n"
            f.write(str_output)
        str_output_2 = "\nThe clustering results:\n"
        for x in point_to_cluster_list:
            str_output_2 = str_output_2 + str(x[0]) + "," + str(x[1]) + "\n"
        f.write(str_output_2)

    print("Duration: ", time.time()-start_time)
    #print("point_to_cluster_list len ", len(point_to_cluster_list))
    #my_labels = [x[1] for x in point_to_cluster_list]
    #r = raw_data[:, 1:2].tolist()
    #ground_truth = [x[0] for x in r]
    #print("normalized_mutual_info_score: ", normalized_mutual_info_score(my_labels, ground_truth))
