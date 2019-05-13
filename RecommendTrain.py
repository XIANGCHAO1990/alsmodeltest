#! /usr/bin/python
#coding=utf-8

from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS,Rating

def SaveModel(sc):
    try:
        model.save(sc,Path+"ALSmodel")
        print("已存储Model在ALSmodel")
    except Exception:
        print('Model已经存在，请先删除再存储')
        
def PrepareData(sc):
    rawUserData = sc.textFile(Path+"data/ml-100k/u.data")
    rawRatings = rawUserData.map(lambda line: line.split("\t")[:3])
    ratingsRDD = rawRatings.map(lambda x: (x[0],x[1],x[2]))
    return ratingsRDD

if __name__ == '__main__':
    Path = "file:/home/coolingshooter/workspace/ALSspark/"
    spark = SparkSession.builder.appName('RT').enableHiveSupport().config("spark.some.config.option","some-value").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    print("===========数据准备阶段===========")
    ratingsRDD = PrepareData(sc)
    print("===========训练阶段==============")
    print("开始ALS训练，参数rank=5,iterations=20,lambda=0.1")
    model = ALS.train(ratingsRDD,5,20,0.1)
    print("===========存储Model============")
    SaveModel(sc)