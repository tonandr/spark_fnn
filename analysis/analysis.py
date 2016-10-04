'''
    Analysis of learning models' evaluation
    @author: Inwoo Chung (gutomitai@gmail.com)
    @since: Sep. 17, 2016 
'''

import csv

'''
    Analysis about SparkNeuralNetwork.
'''

from py4j.java_gateway import *
gateway = JavaGateway(auto_field=True, auto_convert=True)
java_import(gateway.jvm, "org.apache.spark.SparkConf")
conf = gateway.jvm.SparkConf().setMaster("local[*]").setAppName("SparkNN")
sc = gateway.entry_point.getSparkContext(conf)
X, Y = analysis.parseHandwrittenNumData(gateway, "ex4data1.csv", loadFlag=True)
numLayers = 3; numActs = gateway.new_array(gateway.jvm.int, numLayers)
numActs[0] = 400; numActs[1] = 25; numActs[2] = 10
opt = gateway.jvm.maum.dm.SparkNeuralNetwork.BGD(0.0005, 1000)
hwn_nn = gateway.entry_point.getSparkNeuralNetwork(numLayers, numActs, opt)
result = hwn_nn.train(sc, X, Y, 0.0, 0.0, False); r = convPy4jArray(result.costVals); ar = list()

# Parse handwritten numbers data into matrixes.
def parseHandwrittenNumData(gateway, dataFilePath, loadFlag = False):
    if (loadFlag):
        X = gateway.jvm.maum.dm.Matrix.loadMatrix("HWN_X.ser")
        Y = gateway.jvm.maum.dm.Matrix.loadMatrix("HWN_Y.ser")
    
        return X, Y
    
    with open(dataFilePath, 'rb') as file:
        csvReader = csv.reader(file)
        
        # Create X, Y java matrix instances.
        X = gateway.jvm.maum.dm.Matrix(400, 5000, 0.0)
        Y = gateway.jvm.maum.dm.Matrix(10, 5000, 0.0)
        
        # Assign values at X, Y.
        cols = 1
        
        for line in csvReader:
            print str(cols) + ", " + str(int(line[400]))
            for i in range(len(line) - 1):
                X.setVal(i + 1, cols, float(line[i]))
            
            Y.setVal(int(line[400]), cols, 1.0)
            
            cols = cols + 1
        
        # Save.
        X.saveMatrix("HWN_X.ser")
        Y.saveMatrix("HWN_Y.ser")
            
        return X, Y
