# This repository consists of the code for simulation and real data application in the manuscript entitled "Sparse Estimation of Historical Functional Linear Models with a Nested Group Bridge Approach". 

The folder "hist_mod_Mfuns" contains the necessary MATLAB functions. The most important ones are introduced below. 

1. The zipped folder fdaM inside the folder "hist_mod_Mfuns" contains the package developed by Prof. Ramsay for functional data analysis (check more at http://www.psych.mcgill.ca/misc/fda/software.html); Please unzipped this folder inside the folder "hist_mod_Mfuns" first.
	
2. File NodeIndexation.m creates the indices of the nodes;
	
3. File ParalleloGrid.m creates the coordinates of the nodes corresponding to their indices;
	
4. File BetaEvalFD.m evaluates a regression function for a given basis coefficient vector at specified time values;
	
5. File DesignMatrixFD.m constructs the design matrix;
	
6. File bhat_gbridge.m implements the nested group bridge approach, which returns the estimated basis coefficients, and degrees of freedom;
	
7. File calc_delta.m returns the estimated delta.

The folder "speech" contains the speech production experiment data (EMG.dat and LipACC.dat) and the code for analysis (analysis.m), as well as the bootstrap confidence interval by resampling the residual.

To replicate the simulations,
	1.	Setup the path properly and run simu_scX.m, for X=1, 2, 3;
	2.	Data simu_x.dat contains the functional covariate to generate the data.

Further explanations can be found in the annotated files.
