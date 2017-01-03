/*
 * Copyright 2015, 2016 (C) Inwoo Chung (gutomitai@gmail.com)
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * 		http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package maum.dm;

import maum.dm.optimizer.Optimizer;

/**
 * Neural network for regression.
 * 
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Dec. 26, 2016
 */
public class NeuralNetworkRegression extends AbstractNeuralNetwork {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6640917303731514764L;

	public NeuralNetworkRegression(int clusterComputingMode, int acceleratingComputingMode,
			int numLayers, int[] numActs, Optimizer optimizer) {
		super(REGRESSION_TYPE, clusterComputingMode, acceleratingComputingMode, numLayers, numActs, optimizer);
	}
	
	/**
	 * Predict.
	 * @param Matrix of input values for prediction.
	 * @return Predicted result matrix.
	 */
	public Matrix predict(Matrix X) {
		
		// Check exception.
		// Null.
		if (X == null) 
			throw new NullPointerException();
		
		// X dimension.
		if (X.colLength() < 1 || X.rowLength() != numActs[0]) 
			throw new IllegalArgumentException();
		
		return feedForwardR(X);
	}
}
