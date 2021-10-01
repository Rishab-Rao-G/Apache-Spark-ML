from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql.functions import rand
import sklearn
from pyspark.mllib.evaluation import MulticlassMetrics
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
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
import pyspark.sql.functions as F

#Create a Spark Context
sc = SparkContext()

#Enable a spark session 
spark = SparkSession \
    .builder \
    .appName("RF in Spark") \
    .getOrCreate()

#Read thhe data from HDFS 
Data = spark.read.format("csv").option("header", "true").load('hdfs:///user/hduser/ECG.csv')

#Rename the last coulm to labels
Data = Data.withColumnRenamed("0_y", "label")

#Convert all the column into floating type
Data = Data.select(*(col(c).cast("float").alias(c) for c in Data.columns))

#Select all the feature columns (201 columns) and
#store it a vector named features
Features_assem = VectorAssembler(inputCols=[c for c in Data.columns if c not in {'label'}], outputCol="features")

#Apply the transformation to the data
Data = Features_assem.transform(Data)

#Select the features and labels columns for classification
Final_Data = Data.select(col("label"), col("features"))

#Show the Final Data table
Final_Data.show()

#Split the Data into train and test data in the ratio 70:30 
Train_Data, Test_Data = Final_Data.randomSplit([0.7,0.3])

#Create a random forest model 
rf = RandomForestClassifier(labelCol="label", 
                            featuresCol="features", 
                            numTrees = 200, 
                            maxDepth = 5, 
                            maxBins = 30)

#Fit the model on the training data
rf_model = rf.fit(Train_Data)

#Test the trained model on the test dataset
predictions = rf_model.transform(Test_Data)

#Select the features and labels from the predicted data for evaluation
y_true = predictions.select(['label']).collect()
y_pred = predictions.select(['prediction']).collect()

#Print the classification report
print(classification_report(y_true, y_pred))

#Create a confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)

#Extract tn, tp, fn, fp from confusion matrix
fp = conf_mat.sum(axis=0) - np.diag(conf_mat)
fn = conf_mat.sum(axis=1) - np.diag(conf_mat)
tp = np.diag(conf_mat)
tn = conf_mat.sum() - (fp + fn + tp)

fp = fp.astype(float)
fn = fn.astype(float)
tp = tp.astype(float)
tn = tn.astype(float)

#Calculate the specificity and sensitivity
SE = tp/(tp+fn)
SP = tn/(fp+tn)
SE = np.average(SE)
SP = np.average(SP)

print(" SE: %.4f | SP: %4f" %(SE, SP))