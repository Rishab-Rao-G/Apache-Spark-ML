from sklearn.metrics import classification_report, confusion_matrix
import sklearn
from pyspark.mllib.evaluation import MulticlassMetrics
from collections import Counter
from pyspark.sql.functions import col
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext
import numpy as np
from pyspark.sql.types import FloatType
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

#Create a Spark Context
sc = SparkContext()

#Enable a spark session 
spark = SparkSession \
    .builder \
    .appName("SVM in Spark") \
    .getOrCreate()

#Read thhe data from HDFS 
Data = spark.read.format("csv").option("header", "true").load('hdfs:///user/hduser/ECG.csv')

#Convert all the column into floating type
Data = Data.select(*(col(c).cast("float").alias(c) for c in Data.columns))

#Rename the last coulm to labels
Data = Data.withColumnRenamed("0_y", "label")

#Select all the feature columns (201 columns) and
#store it a vector named features
Features_assem = VectorAssembler(inputCols=[c for c in Data.columns if c not in {'label'}], outputCol="features")

#Apply the transformation to the data
Data = Features_assem.transform(Data)

#Select the features and labels columns for classification
Final_Data = Data.select(col("label"), col("features"))

#Split the Data into train and test data in the ratio 70:30 
Train_Data, Test_Data = Final_Data.randomSplit([0.8,0.2])

#Create a linear model with 25 iterations
#Fit one vs all classifier-to enable multiclass classification
lsvc = LinearSVC(maxIter=35, regParam=0.001)
ovr = OneVsRest(classifier=lsvc)

#Fit the model on the training data
model = ovr.fit(Train_Data)

#Test the model on test data
predictions = model.transform(Test_Data)

#Select the features and labels from the predicted data for evaluation
y_true = predictions.select(['label']).collect()
y_pred = predictions.select(['prediction']).collect()

#Print the classification report
print(classification_report(y_true, y_pred))

#Create a confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)
print(type(conf_mat))

#Extract tn, tp, fn, fp from confusion matrix
fp = conf_mat.sum(axis=0) - np.diag(conf_mat)
fn = conf_mat.sum(axis=1) - np.diag(conf_mat)
tp = np.diag(conf_mat)
tn = conf_mat.sum() - (fp + fn + tp)

fp = fp.astype(float)
fn = fn.astype(float)
tp = tp.astype(float)
tn = tn.astype(float)
"""
print(fp)
print(fn)
print(tn)
print(tp)
"""
#Calculate the accuracy specificity and sensitivity
acc = (tp+tn)/(tp+tn+fp+fn)
acc = np.average(acc)
SE = tp/(tp+fn)
SP = tn/(fp+tn)
SE = np.average(SE)
SP = np.average(SP)

print(" SE: %.4f | SP: %4f | acc: %.4f" %(SE, SP, acc))


