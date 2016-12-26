/**  
  Minimize a continuous differentialble multivariate function. Starting point
  is given by "X" (D by 1), and the function named in the string "f", must
  return a function value and a vector of partial derivatives. The Polack-
  Ribiere flavour of conjugate gradients is used to compute search directions,
  and a line search using quadratic and cubic polynomial approximations and the
  Wolfe-Powell stopping criteria is used together with the slope ratio method
  for guessing initial step sizes. Additionally a bunch of checks are made to
  make sure that exploration is taking place and that extrapolation will not
  be unboundedly large. The "length" gives the length of the run: if it is
  positive, it gives the maximum number of line searches, if negative its
  absolute gives the maximum allowed number of function evaluations. You can
  (optionally) give "length" a second component, which will indicate the
  reduction in function value to be expected in the first line-search (defaults
  to 1.0). The function returns when either its length is up, or if no further
  progress can be made (ie, we are at a minimum, or so close that due to
  numerical problems, we cannot get any closer). If the function terminates
  within a few iterations, it could be an indication that the function value
  and derivatives are not consistent (ie, there may be a bug in the
  implementation of your "f" function). The function returns the found
  solution "X", a vector of function values "fX" indicating the progress made
  and "i" the number of iterations (line searches or function evaluations,
  depending on the sign of "length") used.
 
  Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
 
  See also: checkgrad 
 
  Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
 
 
  (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
  
  Permission is granted for anyone to copy, use, or modify these
  programs and accompanying documents for purposes of research or
  education, provided this copyright notice is retained, and note is
  made of any changes that have been made.
  
  These programs and documents are distributed without any warranty,
  express or implied.  As the programs were written for research
  purposes only, they have not been tested to the degree that would be
  advisable in any important application.  All use of these programs is
  entirely at the user's own risk.
 
  [ml-class] Changes Made:
  1) Function name and argument specifications
  2) Output display
  
  ----------------------------------------------------------------------------
  fmincg (octave) developed by Carl Edward Rasmussen is converted into 
  NonlinearCGOptimizer (java) by Inwoo Chung since Dec. 23, 2016.
*/
 
package maum.dm.optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.spark.api.java.JavaSparkContext;

import maum.dm.CostFunctionResult;
import maum.dm.Matrix;
import maum.dm.Utility;

/**
 * Nonlinaer conjugate gradient optimizer extended from fmincg developed 
 * by Carl Edward Rasmussen.
 *  
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Dec. 23, 2016
 */
public class NonlinearCGOptimizer extends Optimizer {	
		
	// Constants.
	public static double RHO = 0.01;
	public static double SIG = 0.5;
	public static double INT = 0.1;
	public static double EXT = 3.0;
	public static int MAX = 20;
	public static double RATIO = 100;
	
	public int maxIter = 100;
	
	public Map<Integer, Matrix> fmincg(ICostFunction iCostFunc
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
		
		// Check exception.
		
		// Optimize the cost function.
		List<CostFunctionResult> costFunctionResults = new ArrayList<CostFunctionResult>();
		
		int length = maxIter;
		
		Matrix T = Utility.unroll(thetas);
		int i = 0;
		int is_failed = 0;
		
		CostFunctionResult r1 = classRegType == 0 ? iCostFunc.costFunctionC(sc
				, clusterComputingMode
				, acceleratingComputingMode
				, X
				, Y
				, thetas
				, lambda
				, isGradientChecking
				, JEstimationFlag
				, JEstimationRatio) 
				: iCostFunc.costFunctionR(sc
						, clusterComputingMode
						, acceleratingComputingMode
						, X
						, Y
						, thetas
						, lambda
						, isGradientChecking
						, JEstimationFlag
						, JEstimationRatio);
		double f1 = r1.J;
		Matrix df1 = Utility.unroll(r1.thetaGrads);
		
		i += length < 0 ? 1 : 0;
		
		Matrix s = Matrix.constArithmeticalMultiply(-1.0, df1);
		double d1 = -1.0 * Matrix.innerProduct(s.unrolledVector(), s.unrolledVector());
		double z1 = 1.0 / (1.0 - d1);
		
		int success = 0;
		
		double f2 = 0.0;
		Matrix df2 = null;
		double z2 = 0.0;			
		double d2 = 0.0;
		double f3 = 0.0;
		double d3 = 0.0;
		double z3 = 0.0;
		
		Matrix T0 = null;
		double f0 = 0;
		Matrix df0 = null;
		
		while (i < Math.abs(length)) {
			i += length > 0 ? 1 : 0;
			
			T0 = T.clone();
			f0 = f1;
			df0 = df1.clone();
			
			T = T.plus(Matrix.constArithmeticalMultiply(z1, s));
			
			CostFunctionResult r2 = classRegType == 0 ? iCostFunc.costFunctionC(sc
					, clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, thetas
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio) 
					: iCostFunc.costFunctionR(sc
							, clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, thetas
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio);
			
			f2 = r2.J;
			df2 = Utility.unroll(r2.thetaGrads);
			z2 = 0.0;
			
			i += length < 0 ? 1 : 0;
			
			d2 = Matrix.innerProduct(df2.unrolledVector(), s.unrolledVector());
			f3 = f1;
			d3 = d1;
			z3 = -1.0 * z1;
			
			int M = 0;
			
			if (length > 0) {
				M = MAX;
			} else {
				M = Math.min(MAX, -1 * length - i);
			}
			
			double limit = -1.0;
			
			while (true) {
				while ((f2 > (f1 + z1 * RHO * d1) || (d2 > -1.0 * SIG * d1)) && (M > 0)) {
					limit = z1;
					
					if (f2 > f1) {
						z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 -f3);
					} else {
						double A = 6.0 * (f2 - f3) / z3 + 3.0 * (d2 + d3);
						double B = 3.0 * (f3 - f2) -z3 * (d3 + 2 * d2);
						z2 = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A;
					}
					
					if (Double.isNaN(z2) || Double.isInfinite(z2)) {
						z2 = z3 / 2.0;
					}
					
					z2 = Math.max(Math.min(z2, INT * z3), (1.0 - INT) * z3);
					z1 = z1 + z2;
					T = T.plus(Matrix.constArithmeticalMultiply(z2, s));
					
					CostFunctionResult r3 = classRegType == 0 ? iCostFunc.costFunctionC(sc
							, clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, thetas
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio) 
							: iCostFunc.costFunctionR(sc
									, clusterComputingMode
									, acceleratingComputingMode
									, X
									, Y
									, thetas
									, lambda
									, isGradientChecking
									, JEstimationFlag
									, JEstimationRatio);
					
					f2 = r3.J;
					df2 = Utility.unroll(r3.thetaGrads);
					
					M = M - 1;
					
					i += length < 0 ? 1 : 0;
					
					d2 = Matrix.innerProduct(df2.unrolledVector(), s.unrolledVector());
					z3 = z3 - z2;
				}
				
				if (f2 > (f1 + z1 * RHO * d1) || d2 > -1.0 *SIG * d1) 
					break;
				else if (d2 > SIG * d1) {
					success = 1;
					break;
				} else if (M == 0) 
					break;
			}
			
			double A = 6.0 * (f2 - f3) / z3 + 3.0 * (d2 + d3);
			double B = 3.0 * (f3 - f2) -z3 * (d3 + 2 * d2);
			z2 = -1.0 * d2 * z3 * z3 /(B + Math.sqrt(B * B - A * d2 * z3 * z3));
			
			if (Double.isNaN(z2) || Double.isInfinite(z2) || z2 < 0.0) {
				if (limit < -0.5) {
					z2 = z1 * (EXT - 1.0);
				} else {
					z2 = (limit - z1) / 2.0;
				}
			} else if (limit > -0.5 && ((z2 + z1) > limit)) {
				z2 = (limit - z1) / 2.0;
			} else if (limit < -0.5 && ((z2 + z1) > (z1 * EXT))) {
				z2 = z1 * (EXT - 1.0);
			} else if (z2 < -1.0 * z3 * INT) {
				z2 = -1.0 * z3 * INT;
			} else if (limit > -0.5 && (z2 < (limit - z1) * (1.0 - INT))) {
				z2 = (limit - z1) * (1.0 - INT);
			}
			
			f3 = f2;
			d3 = d2;
			z3 = -1.0 * z2;
			z1 = z1 + z2;
			T = T.plus(Matrix.constArithmeticalMultiply(z2, s));
			
			CostFunctionResult r4 = classRegType == 0 ? iCostFunc.costFunctionC(sc
					, clusterComputingMode
					, acceleratingComputingMode
					, X
					, Y
					, thetas
					, lambda
					, isGradientChecking
					, JEstimationFlag
					, JEstimationRatio) 
					: iCostFunc.costFunctionR(sc
							, clusterComputingMode
							, acceleratingComputingMode
							, X
							, Y
							, thetas
							, lambda
							, isGradientChecking
							, JEstimationFlag
							, JEstimationRatio);
			
			f2 = r4.J;
			df2 = Utility.unroll(r4.thetaGrads);
			
			M = M - 1;
			
			i += length < 0 ? 1 : 0;
			
			d2 = Matrix.innerProduct(df2.unrolledVector(), s.unrolledVector());
		}
		
		if (success == 1) {
			f1 = f2;
			s = Matrix.constArithmeticalMultiply((Matrix.innerProduct(df2.unrolledVector(), df2.unrolledVector()) 
					- Matrix.innerProduct(df1.unrolledVector(), df2.unrolledVector())) 
					/ (Matrix.innerProduct(df1.unrolledVector(), df1.unrolledVector())), s).minus(df2);
			Matrix tmp = df1.clone();
			df1 = df2.clone();
			df2 = tmp;
			d2 = Matrix.innerProduct(df1.unrolledVector(), s.unrolledVector());
			
			if (d2 > 0) {
				s = Matrix.constArithmeticalMultiply(-1.0, df1);
				d2 = -1.0 * Matrix.innerProduct(s.unrolledVector(), s.unrolledVector());
			}
			
			z1 = z1 * Math.min(RATIO, d1 / (d2 - Double.MIN_VALUE));
			d1 = d2;
			is_failed = 0;
		} else {
			T = T0.clone();
			f1 = f0;
			df1 = df0.clone();
			
			if (!(is_failed == 0 || i > Math.abs(length))) {
				Matrix tmp = df1.clone();
				df1 = df2.clone();
				df2 = tmp;
				s = Matrix.constArithmeticalMultiply(-1.0, df1);
				d1 = -1.0 * Matrix.innerProduct(s.unrolledVector(), s.unrolledVector());
				z1 = 1.0 / (1.0 - d1);
				is_failed = 1;
			}
		}
		
		return Utility.roll(T, thetas);
	}
}
