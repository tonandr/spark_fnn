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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.NullArgumentException;

/**
 * Abstract tree decision.
 * 
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Apr. 27, 2016
 * 
 * Revision
 * 	
 */
public abstract class AbstractTreeDecision {

	// Constants for the tree decision model.
	public final static int FEATURE_TYPE_CLASS = 0;
	public final static int FEATURE_TYPE_COUNTABLE = 1;
	
	/** Feature information. */
	public static class FeatureInfo {
		public String featureName;
		public int featureType;
		public double[] cFeatureRange = new double[2]; // It must be assigned.
		public double range; // It must be assigned.
		
		// Class feature: Map<Integer, String>, Countable feature: Map<Integer, double[]>.
		public Map<Integer, ?> valIndexMap;
	}
	
	/** Tree Node */
	public static class TreeNode {
		public final static int NODE_TREE_ELEMENT_TYPE = 0;
		public final static int LEAF_TREE_ELEMENT_TYPE = 1;
				
		public int treeElementType;
		public String featureName;
		public int featureType;
		public String featureClassVal;
		public double[] featureContVal = new double[2]; // Range: [ );
				
		public Map<Integer, TreeNode> nodeMap = new HashMap<Integer, TreeNode>(); // Itself reference is valid?
		public Map<Integer, Double> probDist = new HashMap<Integer, Double>();
		
		public boolean isMatchedForClassVal(String classVal) {
			
			// Check exception.
			// TODO
			
			return classVal.compareTo(featureName) == 0 ? true : false;
		}
		
		public boolean isMatchedForContVal(double[] contVal) {
			
			// Check exception.
			// TODO
			
			return ((contVal[0] == featureContVal[0]) 
					&& (contVal[1] == featureContVal[1])) ? true : false;
		}
	}
	
	/** Parent tree node. */
	public TreeNode pTreeNode = new TreeNode();
	
	/** Constructor. */
	protected void AbstractTreeDecision(List<FeatureInfo> featureInfos
			, Matrix factorM
			, Matrix targetM) {
		
		// Check exception.
		// Null.
		if (featureInfos == null || factorM == null || targetM == null) 
			throw new NullArgumentException("Faild to instantiate TreeDecision "
					+ "because one among input arguments is null.");
		
		// Class set assignment for each class feature.
		for (FeatureInfo v : featureInfos) {
			if (v.featureType == FEATURE_TYPE_CLASS) {
				if (v.valIndexMap == null) 
					throw new IllegalArgumentException("Failed to instantiate TreeDecision "
							+ "because one among the class feature information list for class set assignment"
							+ "isn't satisfied."); 
			}
		}
		
		// Assign divided ranges for each countable feature.
		assignDividedRanges(featureInfos);
	}
	
	// Assign divided ranges for each countable feature.
	private void assignDividedRanges(List<FeatureInfo> featureInfos) {
		
		// Check feature type and assign divided ranges.
		for (FeatureInfo v : featureInfos) {
			if (v.featureType == FEATURE_TYPE_COUNTABLE) {
				
			}
		}
	}
	
	/** Train. */
	public void train(Matrix factorM, Matrix targetM) {
		
	}
}
