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

import java.io.Serializable;

/**
 * Optimizer
 * 
 * @author Inwoo Chung (gutomitai@gmail.com)
 * @since Dec. 23, 2016
 */
public abstract class Optimizer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7090539796033653772L;
	public int classRegType = 0;
}
