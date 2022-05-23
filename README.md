# SmartHomeEnergy
Detection of Manipulated Pricing in Smart Energy CPS Scheduling
`TrainingData.txt`: Training set. 10,000 predictive guideline price curves with lables

`TestingData.txt`: 100 predictive guideline price curves without labels.

`Classification.py`: Predict whether the predicting guide price is normal or abnormal.

`Schedule.py`: The scheduling model is implemented for task scheduling based on guide prices and scheduling requirements (i.e. Ready time, Deadline, maximum planned energy per hour, energy demand). A bar graph is saved in the `charts` directory for each set of guide prices in the inputs file, which corresponds to the calculated scheduling plan. The indicators of the price curves (the prices predicted in classification.py labeled as 0, i.e. the line numbers of Abnormal Price) are used to name the chart files.

Commandline for classification: Execute the following command for classification,the
out file will be saved in the path ”./dataset/TestingResults.txt”:

        python classification.py

Commandline for scheduling: Execute the following command for scheduling. The
scheduling filename is “input.xslx”:

        python schedule.py

`Testingresults.txt`: 100 rows x 25 columns, output file of ’Classification.py ’. The last
row is the prediction label, where 1 means normal and 0 means abnormal.
