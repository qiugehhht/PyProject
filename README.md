# PyProject

Analysis PGG dataset

processing.ipynb is the main file to analyze the PPG data (modeling, visualization)

preprocess.ipynb is to prove that I continually work on this project (commited by 13 Jan)

apiApp/Data_explain.pptx is the ppt to explain the data and my way of processing

apiApp/Data_expalin.pdf is the pdf version of Data_explain.pptx

apiApp is the API for display and prediction, after running app.py you can open google chrome and input "localhost:5000/index" to view the table of processed data, and input "localhost:5000/decision" to get the best estimator of decision tree. "localhost:5000/plt" can help you to get a heatmap of dataset
 
You can also input "localhost:5000/random" to get best random forest estimator. You can input "localhost:5000/random_tree_pre" to define your own hyperparameters and get your model accuracy by submit the form. After that, you can input your variable to make prediction of target in "localhost:5000/pre_random" (this page will automatically showup ).
