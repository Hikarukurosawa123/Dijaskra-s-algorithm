This file contains the code for the classification of epilepsy patients and non-epilepsy individuals based on electroencephalography data. 
Given the difficulty of distinguishing between the two population in the time domain, the frequency information were obtained using fast fourier transform. 
Some of the features that were experimented include:
1. relative power of the frequency bands (alpha, beta, delta, theta, and gamma band values)
2. product of beta and delta power
3. product of delta and theta power
4. product of beta and theta power

As a result, a training accuracy of 80% and testing accuracy of 55% was obtained. Further optimization through exploration of features, hyperparameters, and regularization methods is necessary to improve the testing results. 

![Figure 2023-09-21 135926](https://github.com/Hikarukurosawa123/hikaruk.github.io/assets/94869114/97726069-ee38-4d32-91b3-2f5c3a7292ea)
