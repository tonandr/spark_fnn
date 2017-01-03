'''
    Analysis of learning models' evaluation
    @author: Inwoo Chung (gutomitai@gmail.com)
    @since: Sep. 17, 2016 
'''

'''
    Analysis about SparkNeuralNetwork.
'''

import math

# Calculate judging accuracy for classification.
def calJudgingAccuracyC(yResults, jResult):
    
    # Count matched numbers.
    length = yResults.colLength()
    count = 0.0
    
    for i in range(length):
        print i, yResults.getVal(1, i + 1), jResult.getVal(1, i + 1)
        
        if (yResults.getVal(1, i + 1) == jResult.getVal(1, i + 1)):
            count = count + 1.0
            
    accuracy = count / float(length)
    
    print str(accuracy * 100.0) + " %"
    return accuracy

# Calculate judging accuracy for regression.
def calJudgingAccuracyR(yResults, jResult):
    
    # Count matched numbers.
    length = yResults.colLength()
    diffSum = 0.0
    
    for i in range(length):
        print i, yResults.getVal(1, i + 1), jResult.getVal(1, i + 1)
        
        diffSum = diffSum + math.fabs(yResults.getVal(1, i + 1) - jResult.getVal(1, i + 1))
            
    accuracy = diffSum / float(length)
    
    print str(accuracy)
    return accuracy
    
# Parse handwritten numbers data into matrixes.
def parseHandwrittenNumData(gateway, dataFilePath, loadFlag = False):
    if (loadFlag):
        X = gateway.jvm.maum.dm.Matrix.loadMatrix("HWN_X.ser")
        Y = gateway.jvm.maum.dm.Matrix.loadMatrix("HWN_Y.ser")
        yResults = gateway.jvm.maum.dm.Matrix.loadMatrix("HWN_yResults.ser")
    
        return X, Y, yResults
    
    with open(dataFilePath, 'rb') as file:
        csvReader = csv.reader(file)
        
        # Create X, Y java matrix instances.
        X = gateway.jvm.maum.dm.Matrix(400, 5000, 0.0)
        Y = gateway.jvm.maum.dm.Matrix(10, 5000, 0.0)
        yResults = gateway.jvm.maum.dm.Matrix(1, 5000, 0.0)
        
        # Assign values at X, Y.
        cols = 1
        
        for line in csvReader:
            print str(cols) + ", " + str(int(line[400]))
            for i in range(len(line) - 1):
                X.setVal(i + 1, cols, float(line[i]))
            
            if (int(line[400]) == 10):
                Y.setVal(1, cols, 1.0)
                yResults.setVal(1, cols, 0.0)
            else:
                Y.setVal(int(line[400]) + 1, cols, 1.0)
                yResults.setVal(1, cols, float(line[400]))

            cols = cols + 1
        
        # Save.
        X.saveMatrix("HWN_X.ser")
        Y.saveMatrix("HWN_Y.ser")
        yResults.saveMatrix("HWN_yResults.ser")
            
        return X, Y, yResults

# Parse handwritten numbers data into matrixes for regression.
def parseHandwrittenNumDataReg(gateway, dataFilePath, loadFlag = False):
    if (loadFlag):
        X = gateway.jvm.maum.dm.Matrix.loadMatrix("HWN_X_reg.ser")
        Y = gateway.jvm.maum.dm.Matrix.loadMatrix("HWN_Y_reg.ser")
        yResults = gateway.jvm.maum.dm.Matrix.loadMatrix("HWN_yResults_reg.ser")
    
        return X, Y, yResults
    
    with open(dataFilePath, 'rb') as file:
        csvReader = csv.reader(file)
        
        # Create X, Y java matrix instances.
        X = gateway.jvm.maum.dm.Matrix(400, 5000, 0.0)
        Y = gateway.jvm.maum.dm.Matrix(1, 5000, 0.0)
        yResults = gateway.jvm.maum.dm.Matrix(1, 5000, 0.0)
        
        # Assign values at X, Y.
        cols = 1
        
        for line in csvReader:
            print str(cols) + ", " + str(int(line[400]))
            for i in range(len(line) - 1):
                X.setVal(i + 1, cols, float(line[i]))
            
            if (int(line[400]) == 10):
                Y.setVal(1, cols, 0.0)
                yResults.setVal(1, cols, 0.0)
            else:
                Y.setVal(1, cols, float(line[400]))
                yResults.setVal(1, cols, float(line[400]))

            cols = cols + 1
        
        # Save.
        X.saveMatrix("HWN_X_reg.ser")
        Y.saveMatrix("HWN_Y_reg.ser")
        yResults.saveMatrix("HWN_yResults_reg.ser")
            
        return X, Y, yResults

# Convert a Py4jArray.
def convPy4jArray(raw_vs):
    vs = list()
    
    for raw_v in raw_vs:
        vs.append(raw_v)
    
    return vs
    
import csv
from py4j.java_gateway import *


# For classification
gateway = JavaGateway(auto_field=True, auto_convert=True)
java_import(gateway.jvm, "org.apache.spark.SparkConf")
conf = gateway.jvm.SparkConf().setMaster("local[*]").setAppName("SparkNN")
sc = gateway.entry_point.getSparkContext(conf)
X, Y, yResults = parseHandwrittenNumData(gateway, "ex4data1.csv", loadFlag=True)
numLayers = 3; numActs = gateway.new_array(gateway.jvm.int, numLayers)
numActs[0] = 400; numActs[1] = 25; numActs[2] = 10
nonlinearCGOptimizer = gateway.entry_point.getNonlinearCGOptimizer();
clusterComputingMode = 0;
acceleratingComputingMode = 0;
hwn_nn = gateway.entry_point.getNeuralNetworkClassification(clusterComputingMode, acceleratingComputingMode, numLayers, numActs, nonlinearCGOptimizer)
la = 0.0; batchMode = 0; numSamplesForMiniBatch = 1; numRepeat = 1; numIter = 20; isGradientChecking = False
JEstimationFlag = False; JEstimationRatio = 1.0
tResult = hwn_nn.train(sc, X, Y, la, batchMode, numSamplesForMiniBatch, numRepeat, numIter, isGradientChecking, JEstimationFlag, JEstimationRatio)
costVals = convPy4jArray(tResult.costVals)
jResults = hwn_nn.judge(X, 0.0)
accuracy = calJudgingAccuracyC(yResults, jResults)

# For regression.
gateway = JavaGateway(auto_field=True, auto_convert=True)
java_import(gateway.jvm, "org.apache.spark.SparkConf")
conf = gateway.jvm.SparkConf().setMaster("local[*]").setAppName("SparkNN")
sc = gateway.entry_point.getSparkContext(conf)
X, Y, yResults = parseHandwrittenNumDataReg(gateway, "ex4data1.csv", loadFlag=True)
numLayers = 3; numActs = gateway.new_array(gateway.jvm.int, numLayers)
numActs[0] = 400; numActs[1] = 25; numActs[2] = 1
nonlinearCGOptimizer = gateway.entry_point.getNonlinearCGOptimizer();
clusterComputingMode = 0;
acceleratingComputingMode = 0;
hwn_nn = gateway.entry_point.getNeuralNetworkRegression(clusterComputingMode, acceleratingComputingMode, numLayers, numActs, nonlinearCGOptimizer)
la = 0.0; batchMode = 0; numSamplesForMiniBatch = 1; numRepeat = 1; numIter = 100; isGradientChecking = False
JEstimationFlag = False; JEstimationRatio = 1.0
tResult = hwn_nn.train(sc, X, Y, la, batchMode, numSamplesForMiniBatch, numRepeat, numIter, isGradientChecking, JEstimationFlag, JEstimationRatio)
costVals = convPy4jArray(tResult.costVals)
jResults = hwn_nn.predict(X)
accuracy = calJudgingAccuracyR(yResults, jResults)

# For classification
gateway = JavaGateway(auto_field=True, auto_convert=True)
java_import(gateway.jvm, "org.apache.spark.SparkConf")
conf = gateway.jvm.SparkConf().setMaster("local[*]").setAppName("SparkNN")
sc = gateway.entry_point.getSparkContext(conf)
X, Y, yResults = parseHandwrittenNumData(gateway, "ex4data1.csv", loadFlag=True)
numLayers = 3; numActs = gateway.new_array(gateway.jvm.int, numLayers)
numActs[0] = 400; numActs[1] = 25; numActs[2] = 10
LBFGSOptimizer = gateway.entry_point.getLBFGSOptimizer();
clusterComputingMode = 1;
acceleratingComputingMode = 0;
hwn_nn = gateway.entry_point.getNeuralNetworkClassification(clusterComputingMode, acceleratingComputingMode, numLayers, numActs, LBFGSOptimizer)
la = 0.0; batchMode = 0; numSamplesForMiniBatch = 1; numRepeat = 1; numIter = 2; isGradientChecking = False
JEstimationFlag = False; JEstimationRatio = 1.0
tResult = hwn_nn.train(sc, X, Y, la, batchMode, numSamplesForMiniBatch, numRepeat, numIter, isGradientChecking, JEstimationFlag, JEstimationRatio)
costVals = convPy4jArray(tResult.costVals)
jResults = hwn_nn.judge(X, 0.0)
accuracy = calJudgingAccuracyC(yResults, jResults)

# For regression.
gateway = JavaGateway(auto_field=True, auto_convert=True)
java_import(gateway.jvm, "org.apache.spark.SparkConf")
conf = gateway.jvm.SparkConf().setMaster("local[*]").setAppName("SparkNN")
sc = gateway.entry_point.getSparkContext(conf)
X, Y, yResults = parseHandwrittenNumDataReg(gateway, "ex4data1.csv", loadFlag=True)
numLayers = 3; numActs = gateway.new_array(gateway.jvm.int, numLayers)
numActs[0] = 400; numActs[1] = 25; numActs[2] = 1
LBFGSOptimizer = gateway.entry_point.getLBFGSOptimizer();
clusterComputingMode = 0;
acceleratingComputingMode = 0;
hwn_nn = gateway.entry_point.getNeuralNetworkRegression(clusterComputingMode, acceleratingComputingMode, numLayers, numActs, LBFGSOptimizer)
la = 0.0; batchMode = 0; numSamplesForMiniBatch = 1; numRepeat = 1; numIter = 50; isGradientChecking = False
JEstimationFlag = False; JEstimationRatio = 1.0
tResult = hwn_nn.train(sc, X, Y, la, batchMode, numSamplesForMiniBatch, numRepeat, numIter, isGradientChecking, JEstimationFlag, JEstimationRatio)
costVals = convPy4jArray(tResult.costVals)
jResults = hwn_nn.predict(X)
accuracy = calJudgingAccuracyR(yResults, jResults)
