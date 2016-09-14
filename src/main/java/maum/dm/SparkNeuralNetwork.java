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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.JavaRDD;

/**
 * Neural network model using Apache Spark.
 * 
 * <p>
 * A default vector is assumed to a vertical array.
 * </p>
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Oct. 21, 2015
 * 
 * <p> Revision </p>
 * <ul>
 * 	<li> -Mar. 22, 2016 </br>
 * 		The name of this class is changed into SparkNeuralNetwork 
 * 		in order to distinguish it from the general abstract neural network class.
 *  </li>
 *  <li>-Mar. 27, 2016 </br>
 *  	The delta term of a vector is changed into the character delta term of a matrix.
 *  </li>
 *  <li>-Sep. 11, 2016 </br>
 *  	Back propagation is debugged.
 *  </li>
 * 	<li>-Sep. 13, 2016 </br>
 * 		Spark context is provided to use it globally.
 * 	</li>
 *  </ul>
 */
public class SparkNeuralNetwork implements Serializable {

	// Debug flag.
	boolean DEBUG = true;

	// Constants for the Neural Network model.
	final static int NUM_ITER = 400;
	final static double LAMBDA = 1.0; //?
	final static double RATE = 0.01; //?
	
	// Spark context.
	private JavaSparkContext sc;
	
	// Neural network architecture.
	private int numLayers;
	private int[] numActs;
	
	// Neural network parameter map.
	private Map<Integer, Matrix> thetas = new HashMap<Integer, Matrix>();
	
	// Cost function relevances.
	private double cost;
	private Map<Integer, Matrix> thetaGradients = new HashMap<Integer, Matrix>();
	
	/** Cost function result. */
	public final static class CostFunctionResult {
		
		/** Cost function value. */
		public double J;
		
		/** Theta gradient matrix. */
		public Map<Integer, Matrix> thetaGrads = new HashMap<Integer, Matrix>();
	}
	
	/**
	 * Constructor.
	 * @param numLayers Number of layers of NN.
	 * @param numActs Array of the number of activations for each layer.
	 */
	public SparkNeuralNetwork(JavaSparkContext sc, int numLayers, int[] numActs) {
		
		// Check exception.
		// Null.
		if (sc == null || numActs == null) 
			throw new NullPointerException();
		
		// Parameters.
		if (numLayers < 2 || numActs.length != numLayers)
			throw new IllegalArgumentException();
		
		for (int n : numActs) 
			if (n < 1) 
				throw new IllegalArgumentException();
		
		this.sc = sc; 		
		this.numLayers = numLayers;
		this.numActs = numActs.clone();
		
		// Create and initialize the theta map for each layer.
		createInitThetaMap();
	}
	
	/**
	 * Train the Neural Network model.
	 * @param X Factor matrix.
	 * @param Y Result matrix.
	 */
	public void train(Matrix X, Matrix Y) {

		// Check exception.
		// Null.
		if (X == null || Y == null) 
			throw new NullPointerException();

		boolean result = (X.rowLength() != (numActs[1 - 1]))
				|| (Y.rowLength() != numActs[numActs.length - 1])
				|| (X.colLength() != Y.colLength());

		if (result)
			throw new IllegalArgumentException();
		
		// Calculate parameters to minimize the Neural Network cost function.
		calParams(X, Y, NUM_ITER, RATE, LAMBDA);
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
		
		return feedForward(X);
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
		Matrix predictedProb = feedForward(X);
		
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
					judgment.setVal(maxIndex, col, maxIndex);
			}
		}
		
		return judgment;
	}
	
	/**
	 * Feedforward.
	 * @param X Matrix of input values.
	 * @return Feedforward probability matrix.
	 */
	protected Matrix feedForward(Matrix X) {
		int numSamples = X.colLength();
		
		// Feedforward.
		// Get activation matrixes for each layer.
		Map<Integer, Matrix> actMatrixes = new HashMap<Integer, Matrix>();
		
		// Get a bias term added X.
		Matrix firstBias = new Matrix(1, X.colLength(), 1.0);
		Matrix firstBiasAddedActMatrix = firstBias.verticalAdd(X);
		
		// Get an activation matrix.
		// Second.
		Matrix secondActMatrix = sigmoid(thetas.get(1).multiply(firstBiasAddedActMatrix));
		actMatrixes.put(1, secondActMatrix);
		
		// From the third activation.
		for (int i = 2; i <= numLayers - 1; i++) {
			
			// Get a bias term added activation.
			Matrix bias = new Matrix(1, actMatrixes.get(i - 1).colLength(), 1.0);
			Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(i - 1));
			
			// Get an activation matrix.
			Matrix actMatrix = sigmoid(thetas.get(i).multiply(biasAddedActMatrix));
			actMatrixes.put(i, actMatrix);
		}
		
		// Get a final activation matrix.
		Matrix finalActMatrix = actMatrixes.get(numLayers);
		
		return finalActMatrix;
	}
	
	// Calculate parameters to minimize the Neural Network cost function.
	private void calParams(Matrix X, Matrix Y, int numIter, double rate, double lambda) {
		
		// Conduct Gradient Descent.
		for (int i = 0; i < numIter; i++) {
			
			// Calculate the cost function and theta gradient.
			CostFunctionResult r = calNNCostFunction(X, Y, rate, lambda);

			if (DEBUG) {
				String result = String.format("NumIter: %d, CostVal: %f", i, r.J);
				printMsgatSameLine(result);
			}

			// Update each theta.
			for (int j = 1; j <= numLayers - 1; j++) {
				Matrix updateTheta 
					= thetas.get(j).minus(Matrix.constArithmeticalMultiply(rate, r.thetaGrads.get(j))); //?
			}
		}
	}

	// Calculate the Neural Network cost function and theta gradient.
	private CostFunctionResult calNNCostFunction(Matrix X, Matrix Y, double rate, double lambda) {
		
		// Calculate the Neural Network cost function.
		final int numSamples = X.colLength();
		CostFunctionResult result = new CostFunctionResult();
		
		// Feedforward.
		// Get activation matrixes for each layer.
		Map<Integer, Matrix> actMatrixes = new HashMap<Integer, Matrix>();

		// The first activation.
		actMatrixes.put(1, X);
		
		// Get a bias term added X.
		Matrix firstBias = new Matrix(1, X.colLength(), 1.0);
		Matrix firstBiasAddedActMatrix = firstBias.verticalAdd(X);
		
		// Get activation matrixes.
		// Second.
		Matrix secondActMatrix = sigmoid(thetas.get(1).multiply(firstBiasAddedActMatrix));
		actMatrixes.put(2, secondActMatrix);
		
		// From the second activation.
		for (int i = 3; i <= numLayers; i++) {
			
			// Get a bias term added activation.
			Matrix bias = new Matrix(1, actMatrixes.get(i - 1).colLength(), 1.0);
			Matrix biasAddedActMatrix = bias.verticalAdd(actMatrixes.get(i - 1));
			
			// Get an activation matrix.
			Matrix actMatrix = sigmoid(thetas.get(i - 1).multiply(biasAddedActMatrix));
			actMatrixes.put(i, actMatrix);
		}
		
		// Get a final activation matrix.
		Matrix finalActMatrix = actMatrixes.get(numLayers); 
		
		// Cost matrix (m x m).
		Matrix cost = 
				Matrix.constArithmeticalDivide
				(Matrix.constArithmeticalMultiply(-1.0, Y.transpose()).multiply(log(finalActMatrix))
						.minus(Matrix.constArithmeticalMinus(1.0, Y).transpose()
						.multiply(log(Matrix.constArithmeticalMinus(1.0, finalActMatrix)))), numSamples);
		
		// Cost function.
		double J = 0.0;
		
		for (int i = 1; i <= numSamples; i++) {
			J += cost.getVal(i, i);
		}
		
		// Regularized cost function.
		double sum = 0.0;
		
		for (int key : thetas.keySet()) {
			Matrix theta = thetas.get(key);
			int[] range = {1, theta.rowLength(), 2, theta.colLength()};
			
			sum += theta.getSubMatrix(range).arithmeticalPower(2.0).sum();
		}
		
		J += lambda / (2.0 * numSamples) * sum;
	
		result.J = J;
		
		// Backpropagation.
		// Calculate character deltas for each layer.	
		// Get samples of the first activation vector.
		List<Matrix> samples = new ArrayList<Matrix>();
		
		for (int i = 1; i <= numSamples; i++) {
			int[] range = {1, firstBiasAddedActMatrix.rowLength(), i, i};
			Matrix a1 = firstBiasAddedActMatrix.getSubMatrix(range);
			a1.index = i;
			samples.add(a1);
		}
				
		JavaRDD<Matrix> a1s = sc.parallelize(samples); // Check the index of each element?
		
		final Matrix YForP = Y; 
		
		// Conduct MapReduce.
		// Map.
		JavaRDD<Map<Integer, Matrix>> cDeltasCollection 
			= a1s.map(new Function<Matrix, Map<Integer, Matrix>>() {

			public Map<Integer, Matrix> call(Matrix a1) throws Exception {
				
				// Feedforward.
				Map<Integer, Matrix> zs = new HashMap<Integer, Matrix>();
				Map<Integer, Matrix> as = new HashMap<Integer, Matrix>();
				as.put(1, a1);
				
				for (int i = 2; i <= numLayers - 1; i++) {
					Matrix z = thetas.get(i - 1).multiply(as.get(i - 1));
					Matrix a =  new Matrix(1, 1, 1.0).verticalAdd(sigmoid(z));
					
					zs.put(i, z);
					as.put(i, a);
				}
				
				Matrix z = thetas.get(numLayers - 1).multiply(as.get(numLayers - 1));
				Matrix a = sigmoid(z);
				
				zs.put(numLayers, z);
				as.put(numLayers, a);
				
				// Calculate delta vectors for each layer.
				Map<Integer, Matrix> deltas = new HashMap<Integer, Matrix>();
				int[] range = {1, YForP.rowLength(), a1.index, a1.index};
				
				deltas.put(numLayers, as.get(numLayers).minus(YForP.getSubMatrix(range)));
				
				for (int i = numLayers - 1; i >= 2; i--) {					
					Matrix biasAddeSG = new Matrix(1, 1, 1.0).verticalAdd(sigmoidGradient(zs.get(i)));
					Matrix preDelta = thetas.get(i).transpose().multiply(deltas.get(i + 1))
							.arithmeticalMultiply(biasAddeSG);
					
					int[] iRange = {2, preDelta.rowLength(), 1, 1};					
					Matrix delta = preDelta.getSubMatrix(iRange); 
					
					deltas.put(i, delta);	
				}
				
				// Accumulate the gradient.
				// Calculate character deltas for each layer.
				Map<Integer, Matrix> cDeltas = new HashMap<Integer, Matrix>();
				
				// Initialize character deltas.
				for (int i = 1; i <= numLayers - 1; i++) {
					cDeltas.put(i, new Matrix(numActs[i], numActs[i - 1] + 1, 0.0));
				}
				
				for (int k = numLayers - 1; k >= 1; k--) {
					cDeltas.get(k).cumulativePlus(deltas.get(k + 1).multiply(as.get(k).transpose()));
				}
				
				return cDeltas;
			}
		});
		
		// Reduce.
		Map<Integer, Matrix> cTotalSumDeltas = cDeltasCollection.reduce(new Function2<Map<Integer,Matrix>
			, Map<Integer,Matrix>
			, Map<Integer,Matrix>>() {
			
			public Map<Integer, Matrix> call(Map<Integer, Matrix> cDeltas1
					, Map<Integer, Matrix> cDeltas2) throws Exception {
				Map<Integer, Matrix> cDeltasPart = new HashMap<Integer, Matrix>();
				
				for (int index : cDeltas1.keySet()) {
					Matrix sumDelta = cDeltas1.get(index).plus(cDeltas2.get(index));
					cDeltasPart.put(index, sumDelta);
				}
				
				return cDeltasPart;
			}
		});
		
		// Obtain the regularized gradient.		
		for (int i = 1; i <= numLayers - 1; i++) {
			int[] range = {1, cTotalSumDeltas.get(i).rowLength(), 2, cTotalSumDeltas.get(i).colLength()};
			Matrix preThetaGrad = Matrix.constArithmeticalDivide(cTotalSumDeltas.get(i).getSubMatrix(range), (double)numSamples)
					.plus(Matrix.constArithmeticalMultiply(lambda / numSamples
							, thetas.get(i).getSubMatrix(range)));
			
			int[] range2 = {1, cTotalSumDeltas.get(i).rowLength(), 1, 1};  
			Matrix thetaGrad = Matrix.constArithmeticalDivide(cTotalSumDeltas.get(i).getSubMatrix(range2), (double)numSamples)
					.horizontalAdd(preThetaGrad);
			
			result.thetaGrads.put(i, thetaGrad);
		}
		
		return result;
	}
	
	// Exponential log.
	private Matrix log(Matrix m) {
		
		// Create a matrix having log function values for the input matrix.
		Matrix lm = new Matrix(m.rowLength(), m.colLength(), 0.0);
		
		for (int rows = 1; rows <= m.rowLength(); rows++) {
			for (int cols = 1; cols <= m.colLength(); cols++) {
				lm.setVal(rows, cols
						, Math.log(m.getVal(rows, cols)));
			}
		}
		
		return lm;
	}
	
	// Sigmoid function.
	private static Matrix sigmoid(Matrix m) {
		
		// Create a matrix having sigmoid function values for the input matrix.
		Matrix sm = new Matrix(m.rowLength(), m.colLength(), 0.0);
		
		for (int rows = 1; rows <= m.rowLength(); rows++) {
			for (int cols = 1; cols <= m.colLength(); cols++) {
				sm.setVal(rows, cols
						, 1.0 /(1.0 + Math.pow(Math.E, -1.0 * m.getVal(rows, cols))));
			}
		}
		
		return sm;
	}
	
	// Sigmoid gradient function.
	private static Matrix sigmoidGradient(Matrix m) {
		
		// Create a sigmoid gradient matrix.
		Matrix one = new Matrix(m.rowLength(), m.colLength(), 1.0);
		Matrix sdm = sigmoid(m).arithmeticalMultiply(one.minus(m));
		
		return sdm;
	}
	
	// Create and initialize the theta map for each layer.
	private void createInitThetaMap() {
		Random rand = new Random();
		double e = 0.12;
		
		thetas.clear();
		
		for (int i = 1; i <= numLayers - 1; i++) {
			
			// Create a theta.
			Matrix theta = new Matrix(numActs[i + 1 - 1], numActs[i - 1] + 1, 0.0); 
			
			// Initialize the created theta.
			for (int rows = 1; rows <= theta.rowLength(); rows++ ) {
				for (int cols = 1; cols <= theta.colLength(); cols++) {
					theta.setVal(rows, cols, rand.nextDouble() * 2.0 * e - e);
				}
			}
			
			thetas.put(i, theta);	
		}
	}
	
	// Print a message.
	public static void printMsg(String msg) {
		System.out.println(msg);
	}

	// Print a message at a same line.
	public static void printMsgatSameLine(String msg) {
		System.out.print("\r" + msg);
	}
}
