This folder contains the code for the analysis of cortex-muscle coupling during dynamic balance.

We investigated the cortical (brain activity) recruitment during postural tasks by measuring the connectivity in the frequency domain between brain signals using electroencephalography (EEG) and muscle signals using electromyography (EMG).

We hypothesized that there would be less cortical involvement with reactive tasks in comparison to voluntary tasks (more consciousness in postural control). 
To test our hypothesis, a lean and release test was conducted. In this test, the participant was asked to lean forward while being supported by a cable with a manual cable release mechanism (shown below). 

<img width="342" alt="image" src="https://github.com/Hikarukurosawa123/hikaruk.github.io/assets/94869114/5277f214-429d-43d1-93ca-7a5164d98672">

Then, the participant executed a single reactive stepping motion once the cable was released. 
Two conditions were compared. 
1) Non-anticipated reactive stepping (cable release without counting down)
2) Anticipated reactive stepping (cable release with a counting down of 3)
We hypothesized that condition 2 would involve more cortical control since the participant is more conscious of the stepping motion that they would be taking post cable release.

Cortical activity using EEG and lower limb muscle activity using EMG of Tibialis Anterior (TA), Medial Gastrocnemius (MG), and Soleus (SOL) muscles was recorded. 

To quantify this recruitment, we utilized two metrics. 
1) Cortico-muscular coherence (CMC) - phase based metric (if two signals display similar phase value, it would result in high connectivity)
2) Instantaneous Amplitude Correlation (IAC) - amplitude based metric (if two signals display coupled oscillation change in amplitude, it would result in high connectivity)
Conventionally, only CMC have been used to quantify the synchronicity between brain and muscles. However, the shortfall of CMC was that it required stationarity of signals (i.e. the spectral profile of the signal needed to remain the same over the entire time course).
Therefore, we incorporated a novel metric, IAC, to capture the cortex-muscle coupling even in fast changing signals.

While there were no difference CMC between the conditions, IAC accurately revealed connectivity in the beta band (12-30 Hz) only for the anticipated stepping.
This suggested that anticipated stepping involved more cortical recruitment in the postural task despite the kinematics of both tasks being almost identical. 
This also suggetsed that IAC may be an effective alternative to CMC to quantify cortex-muscle synchronicity in non-stationary signals. 

Details of the results are available in the Poster titled "Research Day Poster". 

Also, the matlab file titled "dynamic_code_demo" can be run to compute the graphs. Please note that the "code" and "dynamic_supplementary_func" folders must be added to the directory in matlab beforehand. 

![image](https://github.com/Hikarukurosawa123/hikaruk.github.io/assets/94869114/48c3e643-1e36-4785-90a2-cce24c5a5956)

