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
 * Neural network for classification.
 * 
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Dec. 26, 2016
 */
public class NeuralNetworkClassification extends AbstractNeuralNetwork {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5878265523592968408L;

	public NeuralNetworkClassification(int clusterComputingMode, int acceleratingComputingMode,
			int numLayers, int[] numActs, Optimizer optimizer) {
		super(CLASSIFICATION_TYPE, clusterComputingMode, acceleratingComputingMode, numLayers, numActs, optimizer);
	}

	/**
	 * Predict.
	 * @param Matrix of input values for prediction.
	 * @return Predicted probability matrix.
	 */
	public Matrix predictProb(Matrix X) {
		
		// Check exception.
		// Null.
		if (X == null) 
			throw new NullPointerException();
		
		// X dimension.
		if (X.colLength() < 1 || X.rowLength() != numActs[0]) 
			throw new IllegalArgumentException();
		
		return feedForwardC(X);
	}
	
	/**
	 * Judge.
	 * @param Matrix of input values for judgment.
	 * @return Judgment matrix.
	 */
	public Matrix judge(Matrix X, double TH) {
		
		// Check exception.
		// Null.
		if (X == null) 
			throw new NullPointerException();
		
		// X dimension.
		if (X.colLength() < 1 || X.rowLength() != numActs[0]) 
			throw new IllegalArgumentException();
		
		// Threshold.
		if (TH > 1.0 || TH < 0.0) 
			throw new IllegalArgumentException();
		
		// Predict probability.
		Matrix predictedProb = feedForwardC(X);
		
		// Judge.
		Matrix judgment = new Matrix(1, X.colLength(), 0.0);
		
		if (numActs[numActs.length - 1] == 1) {
			for (int col = 1; col <= predictedProb.colLength(); col++) {
				if (predictedProb.getVal(1, col) >= TH) 
					judgment.setVal(1, col, 1.0);
			}
		} else {
			for (int col = 1; col <= predictedProb.colLength(); col++) {
				
				// Get an index with maximum probability.
				int maxIndex = 0;
				double maxVal = 0.0;
				
				for (int row = 1; row <= predictedProb.rowLength(); row++) {
					if (predictedProb.getVal(row, col) > maxVal) {
						maxIndex = row;
						maxVal = predictedProb.getVal(row, col);
					}
				}
				
				if (predictedProb.getVal(maxIndex, col) >= TH) // Check it for f measure. 
					judgment.setVal(1, col, maxIndex - 1);
			}
		}
		
		return judgment;
	}
}
