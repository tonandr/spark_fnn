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

import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.spark.api.java.JavaSparkContext;

import maum.dm.CostFunctionResult;
import maum.dm.Matrix;
import maum.dm.Utility;

/**
 * Limited-memory BFGS optimizer.
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Dec. 30, 2016
 */
public class LBFGSOptimizer extends Optimizer {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -578037482937009349L;

	/**
	 * Minimize.
	 * @param iCostFunc
	 * @param sc
	 * @param clusterComputingMode
	 * @param acceleratingComputingMode
	 * @param X
	 * @param Y
	 * @param thetas
	 * @param lambda
	 * @param isGradientChecking
	 * @param JEstimationFlag
	 * @param JEstimationRatio
	 * @param costFunctionResults
	 * @return 
	 */
	public Map<Integer, Matrix> minimize(ICostFunction iCostFunc
			, JavaSparkContext sc
			, int clusterComputingMode
			, int acceleratingComputingMode
			, Matrix X
			, Matrix Y
			, Map<Integer, Matrix> thetas
			, int maxIter
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio
			, List<CostFunctionResult> costFunctionResults) {
		
		// Input parameters are assumed to be valid.
		
		// Conduct optimization.		
		// Calculate the initial gradient and inverse Hessian.
		Matrix T0 = Utility.unroll(thetas);
		CostFunctionResult r1 = classRegType == 0 ? iCostFunc.costFunctionC(sc
				, clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, Utility.roll(T0, thetas)
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio) 
				: iCostFunc.costFunctionR(sc
						, clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, Utility.roll(T0, thetas)
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio);
		Matrix G0 = Utility.unroll(r1.thetaGrads);
		Matrix H0 = Matrix.getIdentity(G0.rowLength());
		Matrix P0 = Matrix.constArithmeticalMultiply(-1.0, H0.multiply(G0));
		
		// Calculate an optimized step size.
		double alpha = backtrackingLineSearch(T0
				, G0
				, P0
				, iCostFunc
				, sc
				, clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, thetas
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio);
		
		// Update the theta.
		Matrix T1 = T0.plus(Matrix.constArithmeticalMultiply(alpha, P0));
		
		CostFunctionResult r2 = classRegType == 0 ? iCostFunc.costFunctionC(sc
				, clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, Utility.roll(T1, thetas)
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio) 
				: iCostFunc.costFunctionR(sc
						, clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, Utility.roll(T1, thetas)
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio);
		
		Matrix G1 = Utility.unroll(r2.thetaGrads);
		costFunctionResults.add(r2);
		
		// Check exception. (Bug??)	
		if (Double.isNaN(r2.J)) {
			return Utility.roll(T0, thetas);
		}
		
		// Store values for inverse Hessian updating.
		LinkedList<Matrix> ss = new LinkedList<Matrix>();
		LinkedList<Matrix> ys = new LinkedList<Matrix>();
		
		Matrix s = T1.minus(T0);
		Matrix y = G1.minus(G0);
		
		ss.add(s);
		ys.add(y);
		
		int count = 1;
		
		while (count <= maxIter) {
			
			// Calculate the next theta, gradient and inverse Hessian.
			T0 = T1;
			
			r1 = classRegType == 0 ? iCostFunc.costFunctionC(sc
					, clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, Utility.roll(T0, thetas)
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio) 
					: iCostFunc.costFunctionR(sc
							, clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, Utility.roll(T0, thetas)
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio);
			G0 = Utility.unroll(r1.thetaGrads);
			
			// Calculate a search direction.
			P0 = calSearchDirection(ss, ys, G0);
			
			// Calculate an optimized step size.
			alpha = backtrackingLineSearch(T0
					, G0
					, P0
					, iCostFunc
					, sc
					, clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, thetas
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio);
			
			// Update the theta.
			T1 = T0.plus(Matrix.constArithmeticalMultiply(alpha, P0));
			
			r2 = classRegType == 0 ? iCostFunc.costFunctionC(sc
					, clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, Utility.roll(T1, thetas)
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio) 
					: iCostFunc.costFunctionR(sc
							, clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, Utility.roll(T1, thetas)
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio);
			
			G1 = Utility.unroll(r2.thetaGrads);
			costFunctionResults.add(r2);
			
			// Check exception. (Bug??)	
			if (Double.isNaN(r2.J)) {
				return Utility.roll(T0, thetas);
			}
			
			// Store values for inverse Hessian updating.
			if (ss.size() == 30) {
				ss.removeFirst();
				ys.removeFirst();
				ss.add(T1.minus(T0));
				ys.add(G1.minus(G0));
			} else {
				ss.add(T1.minus(T0));
				ys.add(G1.minus(G0));
			}

			count++;
		}
		
		return Utility.roll(T1, thetas);
	}
	
	// Calculate a search direction.
	private Matrix calSearchDirection(LinkedList<Matrix> ss, LinkedList<Matrix> ys, Matrix G0) {
		
		// Input parameters are assumed to be valid.
		
		// Calculate P.
		Matrix Q = G0.clone();
		LinkedList<Double> as = new LinkedList<Double>();
		
		for (int i = ss.size() - 1; i >= 0; i--) {
			double rho = 1.0 / ys.get(i).transpose().multiply(ss.get(i)).getVal(1, 1); // Inner product.
			double a = rho * ss.get(i).transpose().multiply(Q).getVal(1, 1);
			as.add(a);
			Q = Q.minus(Matrix.constArithmeticalMultiply(a, ys.get(i)));
		}
		
		double w = 1.0; //ys.getLast().transpose().multiply(ys.getLast()).getVal(1, 1) 
				// / ys.getLast().transpose().multiply(ss.getLast()).getVal(1, 1);
		Matrix H0 = Matrix.constArithmeticalMultiply(w, Matrix.getIdentity(G0.rowLength())); //?
		Matrix R = H0.multiply(Q);
		
		for (int i = 0 ; i < ss.size(); i++) {
			double rho = 1.0 / ys.get(i).transpose().multiply(ss.get(i)).getVal(1, 1);
			double b = Matrix.constArithmeticalMultiply(rho, ys.get(i).transpose().multiply(R)).getVal(1, 1);
			R = R.plus(Matrix.constArithmeticalMultiply(ss.get(i), as.get(as.size() - i - 1) - b));
		}
		
		return Matrix.constArithmeticalMultiply(-1.0, R);
	}
	
	// Conduct backtracking line search.
	private double backtrackingLineSearch(Matrix T
			, Matrix G
			, Matrix P
			, ICostFunction iCostFunc
			, JavaSparkContext sc
			, int clusterComputingMode
			, int acceleratingComputingMode
			, Matrix X
			, Matrix Y
			, Map<Integer, Matrix> thetas
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio) {
		
		// Input parameters are assumed to be valid.
		
		// Calculate an optimized step size.
		double alpha = 1.0;
		double c = 0.5;
		double gamma = 0.5;
		
		CostFunctionResult r1 = classRegType == 0 ? iCostFunc.costFunctionC(sc
				, clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, Utility.roll(T, thetas)
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio) 
				: iCostFunc.costFunctionR(sc
						, clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, Utility.roll(T, thetas)
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio);
		
		Matrix T1 = T.plus(Matrix.constArithmeticalMultiply(alpha, P));
		
		CostFunctionResult r2 = classRegType == 0 ? iCostFunc.costFunctionC(sc
				, clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, Utility.roll(T1, thetas)
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio) 
				: iCostFunc.costFunctionR(sc
						, clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, Utility.roll(T1, thetas)
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio);
		
		Matrix M = P.transpose().multiply(G);
		double m = M.getVal(1, 1);
		
		// Check the Armijo-Goldstein condition.
		while (r1.J - r2.J < -1.0 * alpha * c * m) {
			alpha = gamma * alpha;
			
			T1 = T.plus(Matrix.constArithmeticalMultiply(alpha, P));
			
			r2 = classRegType == 0 ? iCostFunc.costFunctionC(sc
					, clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, Utility.roll(T1, thetas)
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio) 
					: iCostFunc.costFunctionR(sc
							, clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, Utility.roll(T1, thetas)
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio);
			
			M = P.transpose().multiply(G);
			m = M.getVal(1, 1);
		} // Close condition?
		
		return alpha;
	}
}
