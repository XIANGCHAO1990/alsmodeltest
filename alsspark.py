#!/usr/bin/python
#coding=utf-8

from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import Rating,ALS

Path = "file:/home/coolingshooter/workspace/ALSspark/"

spark = SparkSession.builder.appName('alsspark').enableHiveSupport().config("spark.some.config.option","some-value").master("local[*]").getOrCreate()
sc = spark.sparkContext
rawUserData = sc.textFile(Path+"data/ml-100k/u.data")
rawRatings = rawUserData.map(lambda line: line.split("\t")[:3])
ratingsRDD = rawRatings.map(lambda x: (x[0],x[1],x[2]))
numUsers = ratingsRDD.map(lambda x:x[0]).distinct().count()
numMovies = ratingsRDD.map(lambda x:x[1]).distinct().count()

model = ALS.train(ratingsRDD, 10, 10, 0.01)
recommendP = model.recommendProducts(196,5)
model.predict(100, 1141)
model.recommendUsers(product=200,num=5)

itemRDD = sc.textFile(Path+"data/ml-100k/u.item")
movieTitle = itemRDD.map(lambda line: line.split("|")).map(lambda a:(float(a[0]),a[1])).collectAsMap()

for p in recommendP:
    print("对用户"+str(p[0])+"推荐电影"+str(movieTitle[p[1]])+"推荐评分:",p[2])





















