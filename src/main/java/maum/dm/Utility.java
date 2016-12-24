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

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.exception.NullArgumentException;

/**
 * Utility.
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Dec. 24, 2016
 */
public class Utility {
	
	/**
	 * Unroll a matrix map.
	 * @param matrixMap a matrix map.
	 * @return Unrolled matrix.
	 */
	public static Matrix unroll(Map<Integer, Matrix> matrixMap) {
		
		// Check exception.
		if (matrixMap == null)
			throw new NullArgumentException();
		
		if (matrixMap.isEmpty())
			throw new IllegalArgumentException();
		
		// Unroll.
		int count = 0;
		Matrix unrolledM = null;
		
		for (int i : matrixMap.keySet()) {
			if (count == 0) {
				double[] unrolledVec = matrixMap.get(i).unrolledVector();
				unrolledM = new Matrix(unrolledVec.length, 1, unrolledVec);
				count++;
			} else {
				double[] unrolledVec = matrixMap.get(i).unrolledVector();
				unrolledM.verticalAdd(new Matrix(unrolledVec.length, 1, unrolledVec));
			}
		}
		
		return unrolledM;
	}
	
	public static Map<Integer, Matrix> roll(Matrix unrolledM
			, Map<Integer, Matrix> matrixMapModel) {
		
		// Check exception.
		if (unrolledM == null || matrixMapModel == null) 
			throw new NullArgumentException();
		
		// Input parameters are assumed to be valid.
		
		// Roll.
		Map<Integer, Matrix> matrixMap = new HashMap<Integer, Matrix>();
		int rowIndexStart = 1;
		
		for (int i : matrixMapModel.keySet()) {
			int[] range = {rowIndexStart
			               , rowIndexStart + matrixMapModel.get(i).rowLength() 
			               * matrixMapModel.get(i).colLength() - 1
			               , 1
			               , 1};
			double[] pVec = unrolledM.getSubMatrix(range).unrolledVector();
			Matrix pM = new Matrix(matrixMapModel.get(i).rowLength()
					, matrixMapModel.get(i).colLength(), pVec);
			
			matrixMap.put(i, pM);
			rowIndexStart = rowIndexStart + matrixMapModel.get(i).rowLength() 
					* matrixMapModel.get(i).colLength();
		}
		
		return matrixMap;
	}
}
