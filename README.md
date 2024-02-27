# Project: Conformal prediction
Author: Michel Lutz \
Date: WS 2023/24 \
Finished: 15.01.2024

![image](mondrian.jpg)

## Requirements
- Pyhon 3
- R

Most of the code here is available in Jupiter Notebooks and can therefore be easily executed and modified. 
Some scripts import more monolithic Python code for faster execution. R scripts were only supportive for smaller visualisations
visualisations and loading the PERMAD dataset and has not been used for conformal prediction.
 

## Contribution
This thesis is primarily an extensive literature search for the conformal prediction framework and contains papers on this topic up to and including 2023. 
The main part of this thesis is the written elaboration.
The attached presentation provides a quick introduction to the topic.
- Presentation: /presentation/Conformal_Prediction_Presentation.pdf
- Report: /paper/Conformal_Prediction_final.pdf

This code was created as part of the Master Project "Medical Systems Biology Project". \
The source code contains, among others:
- Code for various conformal methods:
  - Re-implementation of different methods
  - Regression
  - Classification
  - Different adaptive methods
  - Code for MAPIE usage
  - Wrapper Torch to SciPy
- Code for Conformal Evaluation
- Code for Visualisation

- Conformal Change Point Detection - Simple Jumper implementation
- Data Handling PERMAD dataset
- Generation of an artificial data set
- Real-Life data set "BEANS"

Unfortunately, the CTM approach for handling the PERMA data set does not provide any useful insights.

## Final remark
In my opinion, the best way to familiarise yourself with the topic is the enclosed presentation. 
Here you will find a slide that shows good starting points for newcomers to the field of Conformal Prediction. 
\
\
Conformal Prediction is fascinating. Have Fun!


