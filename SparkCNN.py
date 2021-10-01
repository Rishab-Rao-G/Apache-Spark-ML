from __future__ import print_function
from sklearn.metrics import confusion_matrix
import pyspark.sql.functions as F
from matplotlib import pyplot
from pyspark.mllib.evaluation import MulticlassMetrics
from collections import Counter
from pyspark.sql.functions import col
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext
from sklearn.metrics import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from pyspark.sql.types import FloatType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from tensorflow.keras import optimizers, regularizers
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform

#Create a Spark Context
sc = SparkContext()

#Create a function to read the data and output train dataset and test dataset
def data():
    spark = SparkSession \
        .builder \
        .appName("CNN in Spark") \
        .getOrCreate()

    Data = spark.read.format("csv").option("header", "true").load('hdfs:///user/hduser/ECG.csv')

    Data = Data.select(*(col(c).cast("float").alias(c) for c in Data.columns))
    Data = Data.withColumnRenamed("0_y", "label")
    X = np.array(Data.select(*(col(c) for c in Data.columns if c not in {'label'})).collect())
    Y = np.array(Data.select('label').collect())
    Y = Y.flatten()

    class_num = 5
    r_seed = 1229
    data_len = len(X)
    np.random.seed(r_seed)
    idx = list(range(data_len))
    np.random.shuffle(idx)

    train_len = int(data_len*0.8)
    test_len = data_len-train_len

    X_train = X[idx][:train_len]
    X_test = X[idx][train_len:]
    Y_train = Y[idx][:train_len]
    Y_test = Y[idx][train_len:]


    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=class_num)
    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=class_num)

    return X_train, Y_train, X_test, Y_test


#Function to create a Deep Neural Network model using 4 layers
def model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(512, input_shape=(201,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    #Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    #Train the model
    result = model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              epochs=5,
              verbose=2,
              validation_split=0.2)
    
    #Find the best accuracy for each epoch
    validation_acc = np.amax(result.history['val_accuracy']) 
    print('Best validation acc of epoch:', validation_acc)
    
    #Return the best accuracy and status of the model
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

#Using HyperOpt function, the DNN can be executed parallely
best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials())
X_train, Y_train, X_test, Y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))

#Select the best configuration of the model and test it on the test data
y_pred = best_model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = list(Y_test)
y_true = np.argmax(y_true, axis=1)

#print the confusion matrix
conf = confusion_matrix(y_true, y_pred).ravel()

#Reshape the single array confusion matrix into nxn matrix
conf = np.reshape(conf,(5,5))

print(type(conf))
#Extract tn, tp, fn, fp from the confusion matrix
fp = conf.sum(axis=0) - np.diag(conf)
fn = conf.sum(axis=1) - np.diag(conf)
tp = np.diag(conf)
tn = conf.sum() - (fp + fn + tp)

fp = fp.astype(float)
fn = fn.astype(float)
tp = tp.astype(float)
tn = tn.astype(float)

#Calculate the sensitivity and specificity
SE = tp/(tp+fn)
SP = tn/(fp+tn)
SE = np.average(SE)
SP = np.average(SP)

print(" SE: %.4f | SP: %4f" %(SE, SP))
