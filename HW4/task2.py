from pyspark import SparkContext
import sys
import time
import collections
import copy


def bfs_traversal(start_node, adj_list):
    bfs_queue = collections.deque([start_node])
    visited = set()
    visited.add(start_node)
    while bfs_queue:
        parent = bfs_queue.popleft()
        children = adj_list[parent]
        for child in children:
            if child not in visited:
                bfs_queue.append(child)
                visited.add(child)
    return visited


def calculate_modularity(components):
    total_sum = 0
    for comp in components:
        comp_sum = 0
        for i in comp:
            sum = 0
            for j in comp:
                aij = 1 if i in adjacency_list[j] else 0
                sum += (aij - ((len(adjacency_list[i]) * len(adjacency_list[j])) / (2 * m)))
            comp_sum += sum
        total_sum += comp_sum
    modularity = total_sum / (2 * m)
    return modularity


# given a graph (list of nodes) find all connected components
def find_connected_components(list_of_nodes, copy_adjacency_list):
    all_connected_components = []
    remaining_nodes = set(list_of_nodes)
    while len(remaining_nodes) != 0:
        start_node = remaining_nodes.pop()
        connected_component = bfs_traversal(start_node, copy_adjacency_list)
        all_connected_components.append(connected_component)
        remaining_nodes = remaining_nodes - connected_component
        # remaining_nodes = [i for i in remaining_nodes if i not in connected_component]
    return all_connected_components


def find_edge_credits(root, all_nodes, adj_list):
    node_credit = {i: 1 for i in all_nodes}
    edge_credit = {}
    no_of_shortest_path = {i: 0 for i in all_nodes}
    no_of_shortest_path[root] = 1
    level = {root: 1}
    popped = set()
    parent_dict = {i: list() for i in all_nodes}
    bfs_queue = collections.deque([(root, 1)])
    visited = [root]
    visited_set = set()
    visited_set.add(root)

    while bfs_queue:
        parent, parent_level = bfs_queue.popleft()
        popped.add(parent)
        children = adj_list[parent]
        for child in children:
            if child not in popped:
                if child in visited_set and level[child] != parent_level:
                    parent_dict[child].append(parent)
                    no_of_shortest_path[child] += no_of_shortest_path[parent]
                elif child not in visited_set:
                    parent_dict[child].append(parent)
                    no_of_shortest_path[child] += no_of_shortest_path[parent]
            if child not in visited_set:
                bfs_queue.append((child, parent_level + 1))
                visited.append(child)
                visited_set.add(child)
                level[child] = parent_level + 1

    bfs_reverse = visited[::-1]
    for node in bfs_reverse:
        parents = parent_dict[node]
        total_no_of_shortest_path_to_node = 0
        for parent in parents:
            total_no_of_shortest_path_to_node += no_of_shortest_path[parent]
        for parent in parents:
            edge = tuple(sorted((node, parent)))
            edge_credit[edge] = (no_of_shortest_path[parent] / total_no_of_shortest_path_to_node) * node_credit[node]
            node_credit[parent] += edge_credit[edge]
    return edge_credit.items()


if __name__ == "__main__":
    start_time = time.time()
    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel("ERROR")
    input_file_path = sys.argv[1]
    btw_output_file_path = sys.argv[2]
    community_output_file_path = sys.argv[3]

    rdd = sc.textFile(input_file_path)
    edge_rdd = rdd.map(lambda line: (line.split(" ")[0], line.split(" ")[1])).map(lambda x: tuple(sorted(x)))

    distinct_nodes_rdd = edge_rdd.flatMap(lambda x: x).distinct()

    # original graph
    adjacency_list = edge_rdd.map(lambda x: [(x[0], [x[1]]), (x[1], [x[0]])]) \
        .flatMap(lambda x: x).reduceByKey(lambda a, b: a + b).collectAsMap()

    all_nodes_list = distinct_nodes_rdd.collect()
    edge_btw = distinct_nodes_rdd.map(
        lambda x: find_edge_credits(x, all_nodes_list, adjacency_list)) \
        .flatMap(lambda x: x).reduceByKey(lambda a, b: a + b).map(lambda x: (x[0], x[1] / 2)).sortBy(
        lambda x: (-x[1], x[0][0], x[0][1])).collect()

    with open(btw_output_file_path, "w") as f:
        for edge in edge_btw:
            f.write(str(edge[0]) + ", " + str(edge[1]))
            f.write("\n")

    # original graph
    m = edge_rdd.count()

    # original graph
    # degrees = edge_rdd.map(lambda x: [(x[0], [x[1]]), (x[1], [x[0]])]) \
    #     .flatMap(lambda x: x).reduceByKey(lambda a, b: a + b).map(lambda x: (x[0], len(x[1]))).collectAsMap()

    copy_adjacency_list = copy.deepcopy(adjacency_list)

    # original graph modularity
    cc = find_connected_components(all_nodes_list, copy_adjacency_list)
    max_modularity = calculate_modularity(cc)
    max_modularity_components = cc

    # remove highest btw edge from original graph
    highest_btw_edge = edge_btw[0][0]
    copy_adjacency_list[highest_btw_edge[0]].remove(highest_btw_edge[1])
    copy_adjacency_list[highest_btw_edge[1]].remove(highest_btw_edge[0])

    n = m-1
    while n > 0:
        # print("n: ", n)
        # modularity
        cc = find_connected_components(all_nodes_list, copy_adjacency_list)
        # print("cc: ", cc)
        modularity = calculate_modularity(cc)
        # print("modularity: ", modularity)
        if modularity > max_modularity:
            max_modularity = modularity
            max_modularity_components = cc

        # between-ess
        edge_btw = distinct_nodes_rdd.map(
            lambda x: find_edge_credits(x, all_nodes_list, copy_adjacency_list)) \
            .flatMap(lambda x: x).reduceByKey(lambda a, b: a + b).map(lambda x: (x[0], x[1] / 2)).sortBy(
            lambda x: (-x[1], x[0][0])).first()
        # print("edge_btw: ", edge_btw)
        edge = edge_btw[0]

        # remove highest btw edge
        copy_adjacency_list[edge[0]].remove(edge[1])
        copy_adjacency_list[edge[1]].remove(edge[0])
        n = n - 1

    for i in range(len(max_modularity_components)):
        max_modularity_components[i] = sorted(max_modularity_components[i])

    sorted_max_modularity_components = sorted(max_modularity_components, key=lambda x: (len(x), x[0]))

    with open(community_output_file_path, "w") as f1:
        output_str = ""
        for community in sorted_max_modularity_components:
            output_str += ", ".join("'"+ x +"'" for x in community)
            output_str = output_str.rstrip(", ")
            output_str += "\n"
        f1.write(output_str)

    print("modularity: ", max_modularity)
    # print("max_modularity_components: ", max_modularity_components)
    print("no of communities: ", len(max_modularity_components))
    print("Duration: ", time.time() - start_time)
