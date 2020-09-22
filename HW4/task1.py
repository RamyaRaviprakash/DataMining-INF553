from graphframes import *
from pyspark.sql import SparkSession
from pyspark import SparkContext
import os
import sys
import time

start_time = time.time()
os.environ["PYSPARK_SUBMIT_ARGS"] = ( "--packages graphframes:graphframes:0.6.0-spark2.3- s_2.11")

sc = SparkContext('local[*]', 'task1')
sc.setLogLevel("ERROR")
spark = SparkSession(sc)

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

rdd = sc.textFile(input_file_path)

distinct_nodes_rdd = rdd.map(lambda line: (line.split(" ")[0], line.split(" ")[1])).flatMap(lambda x:x).distinct().map(lambda x: (x, x))
vertices = distinct_nodes_rdd.toDF(["id","name"])

edges = rdd.map(lambda line: (line.split(" ")[0], line.split(" ")[1])).map(lambda x: [(x[0], x[1]), (x[1], x[0])]).flatMap(lambda x:x).toDF(["src", "dst"])

graph = GraphFrame(vertices, edges)

result = graph.labelPropagation(maxIter=5)
arr = result.select("id", "label").collect()
# print(arr)

label_map = {}

for x in arr:
    id = x[0]
    label = x[1]
    if label not in label_map:
        label_map[label] = list()
        label_map[label].append(id)
    else:
        label_map[label].append(id)

for k in label_map:
    label_map[k] = sorted(label_map[k])

sorted_label_list = sorted(label_map, key=lambda k: (len(label_map[k]), label_map[k][0]))

with open(output_file_path, "w") as f:
    for label in sorted_label_list:
        f.write(", ".join("'" + x + "'" for x in label_map[label]))
        f.write("\n")

print("Duartion: ", time.time()-start_time)
