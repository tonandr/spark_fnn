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

/**
 * <p> Matrix. </p>
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Oct. 20, 2015
 * 
 * <p> Revision </p>
 * <ul>
 * 	<li>
 * 		-Sep. 15, 2016 </br>
 * 			clone is added.
 * </li>
 * 	<li>-Sep. 16, 2016 </br>
 * 		CumulativePlus is debugged.
 * 	</li>
 * </ul>
 */
public class Matrix implements Serializable {
	
	/** Matrix number for parallel processing of Apache Spark. */
	public int index;
	
	// Matrix values.
	protected double[][] m;

	/**
	 * Constructor to create a homogeneous matrix.
	 * @param rows Number of rows.
	 * @param cols Number of columns.
	 * @param val Homogeneous value.
	 */
	public Matrix(int rows, int cols, double val) {
		
		// Create a matrix with a homogeneous value.
		createHomoMatrix(rows, cols, val);
	}
	
	/**
	 * Constructor to create a matrix from a vector.
	 * @param rows Number of rows.
	 * @param cols Number of columns.
	 * @param v Unrolled values' vector.
	 */
	public Matrix(int rows, int cols, double[] v) {
		
		// Create a matrix.
		createMatrixFromVector(rows, cols, v);
	}
	
	/**
	 * Create a homogeneous matrix.
	 * @param rows Number of rows.
	 * @param cols Number of columns.
	 * @param val Homogeneous value.
	 */
	public void createHomoMatrix(int rows, int cols, double val) {
		
		// Check exception.
		// Parameters.
		if (rows < 1 || cols < 1) {
			throw new IllegalArgumentException();
		}
		
		// Create a matrix.
		m = new double[rows][];
		
		for (int i = 0; i < rows; i++) {
			m[i] = new double[cols];
			
			for (int j = 0; j < cols; j++) {
				m[i][j] = val;
			}
		}
	}
	
	/**
	 * Create a matrix from a vector.
	 * @param rows Number of rows.
	 * @param cols Number of columns.
	 * @param v Unrolled values' vector.
	 */
	public void createMatrixFromVector(int rows, int cols, double[] v) {
	
		// Check exception.
		// Parameters.
		if (rows < 1 || cols < 1 || v == null) {
			throw new IllegalArgumentException();
		}
		
		// Valid transformation from a vector to a matrix.
		if (rows * cols != v.length) {
			throw new IllegalArgumentException();
		}
		
		// Create a matrix.
		m = new double[rows][];
		int vIndex = 0;
		
		for (int i = 0; i < rows; i++) {
			m[i] = new double[cols];
			
			for (int j = 0; j < cols; j++) {
				m[i][j] = v[vIndex++];
			}
		}
	}
	
	/**
	 * Get a value of the matrix.
	 * @param rows One-based row index.
	 * @param cols One-based column index.
	 * @return Matrix element value.
	 */
	public double getVal(int rows, int cols) { // One-based index.
		
		// Check exception.
		checkMatrix();
		
		// Parameters.
		if (rows < 1 || rows > m.length || cols < 1 || cols > m[0].length) 
			throw new IllegalArgumentException();
		
		return m[rows - 1][cols - 1];
	}
	
	/**
	 * Set a value of the matrix.
	 * @param rows One-based row index.
	 * @param cols One-based column index.
	 */
	public void setVal(int rows, int cols, double val) { // One-based index.
		
		// Check exception.
		checkMatrix();
		
		// Parameters.
		if (rows < 1 || rows > m.length || cols < 1 || cols > m[0].length) 
			throw new IllegalArgumentException();
		
		m[rows - 1][cols - 1] = val;
	}
	
	/**
	 * Row length of the matrix.
	 * @return Row length.
	 */
	public int rowLength() {
		
		// Check exception.
		checkMatrix();
		
		return m.length;
	}
	
	/**
	 * Column length of the matrix.
	 * @return Column length.
	 */
	public int colLength() {
		
		// Check exception.
		checkMatrix();
				
		return m[0].length;
	}
	
	/**
	 * Get an unrolled vector.
	 * @return Unrolled vector.
	 */
	public double[] unrolledVector() {
		
		// Check exception.
		checkMatrix();
		
		// Create a unrolled vector.
		double[] v = new double[rowLength() * colLength()];
		int vIndex = 0;
		
		for (int i = 0; i < rowLength(); i++) {			
			for (int j = 0; j < colLength(); j++) {
				v[vIndex++] = m[i][j];
			}
		}
		
		return v;
	}
	
	/**
	 * Column vector.
	 * 
	 * @param cols One-based column index.
	 * @return Column vector.
	 */
	public double[] colVector(int cols) {
		
		// Check exception.
		checkMatrix();
		
		// Parameters.
		if (cols < 1 || cols > colLength()) 
			throw new IllegalArgumentException();
		
		// Get a column vector.
		double[] colVec = new double[rowLength()];
		
		for (int i = 0; i < rowLength(); i++) {
			colVec[i] = m[i][cols - 1];
		}
		
		return colVec;
	}
	
	/**
	 * Row vector.
	 * 
	 * @param rows One-based row index.
	 * @return Row vector.
	 */
	public double[] rowVector(int rows) {
		
		// Check exception.
		checkMatrix();
		
		// Parameters.
		if (rows < 1 || rows > rowLength()) 
			throw new IllegalArgumentException();
		
		// Get a row vector.	
		return m[rows - 1].clone();
	}
	
	// Matrix operation.
	/**
	 * Plus operation.
	 * @param om Other matrix.
	 * @return Summation matrix.
	 */
	public Matrix plus(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (rowLength() != om.rowLength() || colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create a summation matrix.
		double[] sv = new double[v.length];
		
		for (int i = 0; i < v.length; i++) {
			sv[i] = v[i] + ov[i];
		}
			
		return new Matrix(rowLength(), colLength(), sv);
	}
	
	/**
	 * Cumulative plus operation.
	 * @param om Other matrix.
	 */
	public void cumulativePlus(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (rowLength() != om.rowLength() || colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Cumulative plus.
		for (int rows = 1; rows <= rowLength(); rows++) {
			for (int cols = 1; cols <= colLength(); cols++) {
				m[rows - 1][cols - 1] = m[rows - 1][cols - 1] + om.getVal(rows, cols);
			}
		}
	}
	
	/**
	 * Minus operation.
	 * @param om Other matrix.
	 * @return Subtracted matrix.
	 */
	public Matrix minus(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (rowLength() != om.rowLength() || colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create a subtracted matrix.
		double[] sv = new double[v.length];
		
		for (int i = 0; i < v.length; i++) {
			sv[i] = v[i] - ov[i];
		}
			
		return new Matrix(rowLength(), colLength(), sv);
	}
	
	/**
	 * Conduct inner product for two vectors.
	 * @param v1 First vector.
	 * @param v2 Second vector.
	 * @return Inner product value.
	 */
	public static double innerProduct(double[] v1, double[] v2) {
		
		// Check exception.
		// Null.
		if (v1 == null || v2 == null) 
			throw new NullPointerException();
		
		// Parameters.
		if (v1.length < 1 || v2.length < 1 || v1.length != v2.length) 
			throw new IllegalArgumentException();
		
		// Conduct inner product for two vectors.
		double sum = 0.0;
		
		for (int i = 0; i < v1.length; i++) {
			sum += v1[i] * v2[i];
		}
		
		return sum;
	}
	
	/**
	 * Multiply operation.
	 * @param om Operand matrix.
	 * @return Multiplied matrix.
	 */
	public Matrix multiply(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (colLength() != om.rowLength()) {
			throw new IllegalArgumentException();
		}
		
		// Create a multiplied matrix.
		Matrix mm = new Matrix(rowLength(), om.colLength(), 0.0);
		
		for (int i = 0; i < rowLength(); i++) {
			for (int j = 0; j < om.colLength(); j++) {
				mm.setVal(i + 1, j + 1, Matrix.innerProduct(rowVector(i + 1), om.colVector(j + 1)));
			}
		}
		
		return mm;
	}
	
	/**
	 * Transpose operation.
	 * @return Transpose matrix.
	 */
	public Matrix transpose() {
		
		// Check exception.
		checkMatrix();
		
		// Create a transpose matrix.
		Matrix tm = new Matrix(colLength(), rowLength(), 0.0);
		
		for (int i = 0; i < rowLength(); i++) {
			for (int j = 0; j < colLength(); j++) {
				tm.setVal(j + 1, i + 1, m[i][j]);
			}
		}
		
		return tm;
	}
	
	/**
	 * Multiply both matrixes arithmetically.
	 * @param om Operand matrix
	 * @return Arithmetically multiplied matrix.
	 */
	public Matrix arithmeticalMultiply(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (rowLength() != om.rowLength() || colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create an arithmetically multiplied matrix.
		double[] amv = new double[v.length];
		
		for (int i = 0; i < v.length; i++) {
			amv[i] = v[i] * ov[i];
		}
			
		return new Matrix(rowLength(), colLength(), amv);
	}
	
	/**
	 * Divide this matrix by an operand matrix arithmetically.
	 * @param om Operand matrix
	 * @return Arithmetically divided matrix.
	 */
	public Matrix arithmeticalDivide(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
		
		// Check exception.
		checkMatrix();
		
		// Dimension.
		if (rowLength() != om.rowLength() || colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create an arithematically divided matrix.
		double[] adv = new double[v.length];
		
		for (int i = 0; i < v.length; i++) {
			adv[i] = v[i] / ov[i];
		}
			
		return new Matrix(rowLength(), colLength(), adv);
	}
	
	/**
	 * Vertically add.
	 * @param om Operand matrix.
	 * @return Vertically added matrix.
	 */
	public Matrix verticalAdd(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
				
		// Check exception.
		checkMatrix();
				
		// Dimension.
		if (colLength() != om.colLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create a vertically added matrix.
		double[] vav = new double[v.length + ov.length];
		
		for (int i = 0; i < v.length; i++) {
			vav[i] = v[i];
		}
		
		for (int i = v.length; i < v.length + ov.length; i++) {
			vav[i] = ov[i - v.length];
		}
		
		return new Matrix(rowLength() + om.rowLength(), colLength(), vav);
	}

	/**
	 * Horizontally add.
	 * @param om Operand matrix.
	 * @return Horizontally added matrix.
	 */
	public Matrix horizontalAdd(Matrix om) {
		
		// The operand matrix is assumed to be valid. 
				
		// Check exception.
		checkMatrix();
				
		// Dimension.
		if (rowLength() != om.rowLength()) {
			throw new IllegalArgumentException();
		}
		
		// Get both unrolled vectors.
		double[] v = unrolledVector();
		double[] ov = om.unrolledVector();
		
		// Create a horizontally added matrix.
		double[] hav = new double[v.length + ov.length];
		
		for (int i = 0; i < rowLength(); i++) {
			for (int k = 0; k < colLength(); k++) {
				hav[i * (colLength() + om.colLength()) + k] = v[i * colLength() + k];
			}
			
			for (int l = 0; l < om.colLength(); l++) {
				hav[i * (colLength() + om.colLength()) + colLength() + l] 
						= ov[i * om.colLength() + l];
			}
		}
		
		return new Matrix(rowLength(), colLength() + om.colLength(), hav);
	}
	
	/**
	 * Arithmetical power.
	 * @param p Operand matrix.
	 * @return Arithmetically powered matrix.
	 */
	public Matrix arithmeticalPower(double p) {
		
		// Check exception.
		checkMatrix();
		
		// Create a arithmetically powered matrix.
		double[] v = unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] = Math.pow(v[i], p);
		}
		
		return new Matrix(rowLength(), colLength(), v);
	}
	
	/**
	 * Matrix + constant value.
	 * @param m Operand matrix.
	 * @param val Constant value.
	 * @return Matrix of a matrix + a constant value. 
	 */
	public static Matrix constArithmeticalPlus(Matrix m, double val) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a matrix + a constant value.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] += val;
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Constant value + matrix.
	 * @param val Constant value.
	 * @param m Operand matrix.
	 * @return Matrix of a constant value + a matrix. 
	 */
	public static Matrix constArithmeticalPlus(double val, Matrix m) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a constant value + a matrix.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] = val + v[i];
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Matrix - constant value.
	 * @param m Operand matrix.
	 * @param val Constant value.
	 * @return Matrix of a matrix - a constant value. 
	 */
	public static Matrix constArithmeticalMinus(Matrix m, double val) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a matrix - a constant value.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] -= val;
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Constant value - matrix.
	 * @param val Constant value.
	 * @param m Operand matrix.
	 * @return Matrix of a constant value - a matrix. 
	 */
	public static Matrix constArithmeticalMinus(double val, Matrix m) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a constant value - a matrix.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] = val - v[i];
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Matrix * constant value.
	 * @param m Operand matrix.
	 * @param val Constant value.
	 * @return Matrix of a matrix * a constant value. 
	 */
	public static Matrix constArithmeticalMultiply(Matrix m, double val) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a matrix * a constant value.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] *= val;
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Constant value * matrix.
	 * @param val Constant value.
	 * @param m Operand matrix.
	 * @return Matrix of a constant value * a matrix. 
	 */
	public static Matrix constArithmeticalMultiply(double val, Matrix m) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a constant value * a matrix.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] = val * v[i];
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Matrix / constant value.
	 * @param m Operand matrix.
	 * @param val Constant value.
	 * @return Matrix of a matrix / a constant value. 
	 */
	public static Matrix constArithmeticalDivide(Matrix m, double val) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a matrix / a constant value.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] /= val;
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Constant value / matrix.
	 * @param val Constant value.
	 * @param m Operand matrix.
	 * @return Matrix of a constant value / a matrix. 
	 */
	public static Matrix constArithmeticalDivide(double val, Matrix m) {
		
		// Check exception.
		m.checkMatrix();
		
		// Create the matrix of a constant value / a matrix.
		double[] v = m.unrolledVector();
		
		for (int i = 0; i < v.length; i++) {
			v[i] = val / v[i];
		}
		
		return new Matrix(m.rowLength(), m.colLength(), v);
	}
	
	/**
	 * Get a sub matrix.
	 * @param range row1:row2, col1:col2 -> {row1, row2, col1, col2} array.
	 * @return Sub matrix.
	 */
	public Matrix getSubMatrix(int[] range) {
		
		// Check exception.
		checkMatrix();
		
		// Null.
		if (range == null) 
			throw new NullPointerException();
		
		// Parameter.
		if (range.length < 4 
				| range[0] < 1 | range[0] > rowLength() 
				| range[1] < range[0]
				| range[1] < 1 | range[1] > rowLength()
				| range[2] < 1 | range[2] > colLength() 
				| range[3] < range[2]
				| range[3] < 1 | range[3] > colLength())
			throw new IllegalArgumentException();
		
		// Create a sub matrix.
		Matrix subMatrix = new Matrix(range[1] - range[0] + 1, range[3] - range[2] + 1, 0.0);
		
		for (int i = range[0]; i <= range[1]; i++) {
			for (int j = range[2]; j <= range[3]; j++) {
				subMatrix.setVal(i - range[0] + 1, j - range[2] + 1
						, getVal(i, j));
			}
		}
						
		return subMatrix;
	}
	
	/**
	 * Clone.
	 * @return Copyied matrix.
	 */
	public Matrix clone() {
		
		// Check exception.
		checkMatrix();
				
		// Create a clone matrix.
		Matrix cloneMatrix = new Matrix(rowLength(), colLength(), 0.0);
		
		for (int i = 1; i <= rowLength(); i++) {
			for (int j = 1; j <= colLength(); j++) {
				cloneMatrix.setVal(i, j, getVal(i, j));
			}
		}
						
		return cloneMatrix;
	}
	
	/**
	 * Sum.
	 * @return Summed value of all values of this matrix.
	 */
	public double sum() {
		
		// Check exception.
		checkMatrix();
		
		// Sum all values of this matrix.
		double[] v = unrolledVector();
		double sum = 0.0;
		
		for (int i = 0; i < v.length; i++) {
			sum += v[i];
		}
	
		return sum;
	}
	
	// Check the matrix state.
	public void checkMatrix() {
		
		// Null.
		if (m == null) 
			throw new IllegalStateException();
	}
}
