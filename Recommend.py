#! /usr/bin/python
#coding=utf-8

from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS,Rating,MatrixFactorizationModel
import sys

def SaveModel(sc):
    try:
        model.save(sc,Path+"ALSmodel")
        print("已存储Model在ALSmodel")
    except Exception:
        print('Model已经存在，请先删除再存储')
        
def PrepareData(sc):
    print("开始读取电影id与名称字典...")
    itemRDD = sc.textFile(Path+"data/ml-100k/u.item")
    movieTitle = itemRDD.map(lambda line: line.split("|")).map(lambda a:(float(a[0]),a[1])).collectAsMap()
    return movieTitle

def loadModel(sc):
    try:
        model = MatrixFactorizationModel.load(sc,Path+"ALSmodel")
        print("载入ALSModel模型")
    except Exception:
        print("找不到ALSModel模型，请先训练")
    return model

def RecommendMovies(model,movieTitle,inputUserID):
    RecommendMovie = model.recommendProducts(inputUserID, 10)
    print("针对用户id"+str(inputUserID)+'推荐下列电影:')
    for rmd in RecommendMovie:
        print("推荐电影{0}推荐评分{1}".format(movieTitle[rmd[1]],rmd[2]))

def RecommendUsers(model,movieTitle,inputMovieID):
    RecommendUser = model.recommendUsers(inputMovieID, 10)
    print("针对电影id{0}电影名:{1}推荐下列用户:".format(inputMovieID,movieTitle[inputMovieID]))
    for rmd in RecommendUser:
        print("推荐用户{0}推荐评分{1}".format(rmd[0],rmd[2]))
        
def Recommend(model):
    if sys.argv[1] == '--U':
        RecommendMovies(model,movieTitle,int(sys.argv[2]))
    if sys.argv[1] == '--M':
        RecommendUsers(model,movieTitle,int(sys.argv[2]))

if __name__ == '__main__':
    Path = "file:/home/coolingshooter/workspace/ALSspark/"   #你的项目路径
    if len(sys.argv) != 3:
        print("请输入2个参数")
        sys.exit(-1)
    spark = SparkSession.builder.appName('RT').enableHiveSupport().config("spark.some.config.option","some-value").master("local[*]").getOrCreate()
    sc = spark.sparkContext
    print("===========数据准备阶段===========")
    movieTitle = PrepareData(sc)
    print("===========载入模型==============")
    model = loadModel(sc)
    print("===========进行推荐============")
    Recommend(model)