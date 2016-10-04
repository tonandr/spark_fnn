/*
 * Copyright 2016 (C) Inwoo Chung (gutomitai@gmail.com)
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

import java.util.List;

import org.apache.commons.math3.exception.NullArgumentException;

/**
 * Abstract recurrent neural network.
 *
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Apr. 12, 2016
 */
public abstract class AbstractRecurrentNeuralNetwork {

	/** RNN architecture. */
	public static class RNNArch {
		
		// Sequence training direction.
		public static int TRAINING_DIRECTION_UNI = 0;
		public static int TRAINING_DIRECTION_BI = 1;
		
		// Gate kind constant.
		public static int GATE_GRU = 0;
		
		/** Sequence training direction. */
		public int seqTrainingDirection;
		
		/** Gate kind. */
		public int gateKind;
		
		/** Number of layers. */
		public int numLayers;
		
		/** Activation number of each layer. */
		public int[] numActs;
		
		/** Depth of sequence. */
		public int depthSequence;
		
		/** Constructor. */
		public RNNArch(int seqTrainingDirection
				, int gateKind
				, int numLayers
				, int[] numActs
				, int depthSequence) {
			
			// Check exception.
			// Null.
			if (numActs == null)
				throw new NullArgumentException();
			
			// Dimension.
			if (numLayers < 3 
					|| numActs.length < 3
					|| numLayers != numActs.length
					|| depthSequence < 1)
				throw new IllegalArgumentException();
			
			for (int n : numActs) {
				if (n < 1)
					throw new IllegalArgumentException();
			}
			
			// Values.
			if (seqTrainingDirection < TRAINING_DIRECTION_UNI || seqTrainingDirection > TRAINING_DIRECTION_BI
					|| gateKind < GATE_GRU || gateKind > GATE_GRU) 
				throw new IllegalArgumentException();
			
			this.seqTrainingDirection = seqTrainingDirection;
			this.gateKind = gateKind;
			this.numLayers = numLayers;
			this.numActs = numActs;
			this.depthSequence = depthSequence;
		}
	}
	
	/** RNN parameters. */
	public static class RNNParams {
		
		// Optimization kind constant.
		public static int OPTIMIZATION_BGD = 0;
		public static int OPTIMIZATION_MBGD = 1;
		public static int OPTIMIZATION_SGD = 2;
		
		/** Optimization kind. */
		public int optimizationKind;
		
		/** Learning rate. */
		public double learningRate;
		
		/** Regularization value. */
		public double regularizationVal;
		
		public RNNParams(int optimizationKind
				, double learningRate
				, double regularizationVal) {
			
			// Check exception.
			// Values.
			if (optimizationKind < OPTIMIZATION_BGD || optimizationKind > OPTIMIZATION_SGD 
					|| learningRate <= 0.0
					|| regularizationVal < 0)
				throw new IllegalArgumentException();
			
			this.optimizationKind = optimizationKind;
			this.learningRate = learningRate;
			this.regularizationVal = regularizationVal;
		}
	}
	
	/** Training samples of X, Y for sequence. */
	public static class TrainingXYForSequence {
		
		/** X. */
		public Matrix X;
		
		/** Y. */
		public Matrix Y;
		
		/** Constructor. */
		public TrainingXYForSequence(Matrix X, Matrix Y) {
			
			// Check exception.
			// Null.
			if (X == null || Y == null) 
				throw new NullArgumentException();
			
			// Dimension.
			if (X.colLength() != Y.colLength())
				throw new IllegalArgumentException();
			
			this.X = X;
			this.Y = Y;
		}
	}
	
	public static class GateUnit {
		
	}
	
	// RNN architecture instance.
	protected RNNArch arch;
	
	// RNN paramters instance.
	protected RNNParams params;
	
	/**
	 * Constructor.
	 * @param arch RNN architecture.
	 * @param params RNN parameters.
	 */
	public AbstractRecurrentNeuralNetwork(RNNArch arch, RNNParams params) {
		
		// Check exception.
		// Null.
		if (arch == null || params == null) 
			throw new NullArgumentException();
		
		this.arch = arch;
		this.params = params;
		
		
	}
	
	/**
	 * Train.
	 * @param samples List of training X, Y for sequence.
 	 */
	public void train(List<TrainingXYForSequence> samples) {
		
		// Check exception.
		// Null.
		if (samples == null) 
			throw new NullArgumentException();
		
		
		
	}
	
}
