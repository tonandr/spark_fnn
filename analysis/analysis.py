'''
    Analysis of learning models' evaluation
    @author: Inwoo Chung (gutomitai@gmail.com)
    @since: Sep. 17, 2016 
'''

import csv

'''
    Analysis about SparkNeuralNetwork.
'''

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
