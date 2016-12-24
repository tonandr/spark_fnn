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
package maum.dm.optimizer;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.JavaSparkContext;

import maum.dm.CostFunctionResult;
import maum.dm.Matrix;
import maum.dm.SparkNeuralNetwork.TrainingResult;
import maum.dm.Utility;

/**
 * Gradient descent optimizer.
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Dec. 25, 2016
 */
public class GradientDescentOptimizer extends Optimizer {

	// Debug flag.
	public boolean DEBUG = true;
	
	/**
	 * Minimize.
	 * @param iCostFunc
	 * @param sc
	 * @param X
	 * @param Y
	 * @param thetas
	 * @param lambda
	 * @param isGradientChecking
	 * @param JEstimationFlag
	 * @param JEstimationRatio
	 * @param numRepeat
	 * @return
	 */
	public Map<Integer, Matrix> minimize(ICostFunction iCostFunc
			, JavaSparkContext sc
			, Matrix X
			, Matrix Y
			, Map<Integer, Matrix> preThetas
			, double rate
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio
			, int numRepeat) {

		// Check exception.

		// Minimize.
		// Copy thetas.
		Map<Integer, Matrix> thetas = Utility.roll(Utility.unroll(preThetas), preThetas);
		
		int numLayers = thetas.size() + 1; //?
		
		for (int i = 0; i < numRepeat; i++) {
			
			// Calculate the cost function and theta gradient.
			CostFunctionResult r = iCostFunc.costFunction(sc
					, X
					, Y
					, thetas
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio);

			if (DEBUG) {
				String result = String.format("NumRepeat: %d, CostVal: %f \n", i, r.J);
				printMsgatSameLine(result);
			}

			// Update each theta.
			for (int j = 1; j <= numLayers - 1; j++) {
				Matrix updateTheta 
					= thetas.get(j).minus(Matrix.constArithmeticalMultiply(rate, r.thetaGrads.get(j)));
				thetas.replace(j, updateTheta);
			}
		}
		
		return thetas;
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
