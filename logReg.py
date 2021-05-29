import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


#Machine Learning Logic for soultion
def machineLearningLogic():
	currentDir = os.getcwd()
	currentFileCSV = currentDir +"//" + "ddp.csv"
	df = pd.read_csv(currentFileCSV)

	X_all = df[['NaCl-concentration(mM)','pH']].values.tolist()

	Y_all = df['Aggregation'].values.tolist()


	X_train,X_test,y_train,y_test = train_test_split(X_all,Y_all,test_size=0.4,random_state=0)



	logistic_regression = LogisticRegression()
	logistic_regression.fit(X_train,y_train)
	y_pred=logistic_regression.predict(X_test)

	# Use score method to get accuracy of model
	score = logistic_regression.score(X_test, y_test)
	print("the score is: " + str(score*100))

	print(f1_score(y_test, y_pred, average="macro"))
	print(precision_score(y_test, y_pred, average="macro"))
	print(recall_score(y_test, y_pred, average="macro")) 
	print(accuracy_score(y_test, y_pred)) 



	#confusion matrix
	cm = confusion_matrix(y_test, y_pred)
	print(cm)

	return




machineLearningLogic()

