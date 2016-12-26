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

import java.util.Map;

import org.apache.spark.api.java.JavaSparkContext;

import maum.dm.CostFunctionResult;
import maum.dm.Matrix;

/**
 * Cost function.
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Dec. 23, 2016
 */
public interface ICostFunction {
	public CostFunctionResult costFunctionC(JavaSparkContext sc
			, int clusterComputingMode
			, int acceratingComputingMode
			, Matrix X
			, Matrix Y
			, final Map<Integer, Matrix> thetas
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio);
	
	public CostFunctionResult costFunctionR(JavaSparkContext sc
			, int clusterComputingMode
			, int acceratingComputingMode
			, Matrix X
			, Matrix Y
			, final Map<Integer, Matrix> thetas
			, double lambda
			, boolean isGradientChecking
			, boolean JEstimationFlag
			, double JEstimationRatio);
}
