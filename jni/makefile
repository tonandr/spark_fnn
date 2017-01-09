#
# Copyright 2016 (C) Inwoo Chung (gutomitai@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 		http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Title: Matrix native functions for CUDA.
# Author: Inwoo Chung (gutomitai@gmail.com)
# Since: Jan. 10, 2017
#

# Define a variable for classpath.
CLASS_PATH = /Users/gutoo/topcoder/fishing_judging/dm/target/classes

# Define a virtual path for .class in the bin directory.
#vpath %.class $(CLASS_PATH)

# $* matches the target filename without the extension.
maum_dm_Matrix.h : maum.dm.Matrix.class
		javah -classpath $(CLASS_PATH) $*
