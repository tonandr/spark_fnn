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
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.exception.NullArgumentException;

/**
 * Utility.
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Dec. 24, 2016
 */
public class Utility implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 733845246331692308L;

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
				unrolledM = unrolledM.verticalAdd(new Matrix(unrolledVec.length, 1, unrolledVec));
			}
		}
		
		return unrolledM;
	}
	
	/**
	 * Unroll a Matrix.
	 * @param unrolledM
	 * @param matrixMapModel
	 * @return
	 */
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
	
	/**
	 * Encode NN params. to Base64 string.
	 * @param nn
	 * @return
	 */
	public static String encodeNNParamsToBase64(AbstractNeuralNetwork nn) {
		
		// Check exception.
		if (nn == null) 
			throw new NullPointerException();
		
		// Unroll a theta matrix map.
		double[] unrolledTheta = unroll(nn.thetas).unrolledVector();
		
		// Convert the unrolled theta into a byte array.
		byte[] unrolledByteTheta = new byte[unrolledTheta.length * 8];
		
		for (int i = 0; i < unrolledTheta.length; i++) {
			
			// Convert a double value into a 8 byte long format value.
			long lv = Double.doubleToLongBits(unrolledTheta[i]);
			
			for (int j = 0; j < 8; j++) {
				unrolledByteTheta[j + i * 8] = (byte)((lv >> ((7 - j) * 8)) & 0xff);
			}
		}
		
		// Encode the unrolledByteTheta into the base64 string.
		String base64Str = Base64.getEncoder().encodeToString(unrolledByteTheta);
		
		return base64Str;
	}
	
	/**
	 * Decode Base64 string into a theta map of NN.
	 * @param nn
	 * @return
	 */
	public static void decodeNNParamsToBase64(String encodedBase64NNParams, AbstractNeuralNetwork nn) {
		
		// Check exception.
		if (nn == null && encodedBase64NNParams == null) 
			throw new NullPointerException();
		
		// Input parameters are assumed to be valid.
		// An NN theta model must be configured.
		
		// Decode the encoded base64 NN params. string into a byte array.
		byte[] unrolledByteTheta = Base64.getDecoder().decode(encodedBase64NNParams);
		
		// Convert the byte array into an unrolled theta.
		double[] unrolledTheta = new double[unrolledByteTheta.length / 8];
				
		for (int i = 0; i < unrolledTheta.length; i++) {
			
			// Convert 8 bytes into a 8 byte long format value.
			long lv = 0;
			
			for (int j = 0; j < 8; j++) {
				long temp = ((long)(unrolledByteTheta[j + i * 8])) << ((7 - j) * 8);
				lv = lv | temp;
			}
			
			unrolledTheta[i] = Double.longBitsToDouble(lv);
		}
		
		// Convert the unrolled theta into a NN theta map.
		nn.thetas = roll(new Matrix(unrolledTheta.length, 1, unrolledTheta), nn.thetas);
	}
}
