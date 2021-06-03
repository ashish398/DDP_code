#TO CALCULATE THE PERFORMANCE OF THE MODEL
import os
#MACHINE LEARNING LIBRARY IMPORTS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


#MACHINE LEARNING LOGIC
def machineLearningLogic():
	#IMPORTING FILE
	currentDir = os.getcwd()
	currentFileCSV = currentDir +"//" + "ddp_database.csv"
	df = pd.read_csv(currentFileCSV)

	#CREATING THE FEATURE VECTOR,RESULT VECTOR AND INDICES VECTOR 
	X = df[['Name','NaCl-concentration(mM)','pH']].values.tolist()
	X_all = df[['NaCl-concentration(mM)','pH']].values.tolist()
	y_all = df['Aggregation'].values.tolist()
	indices = range(len(X_all))

	#SPLITING THE DATASET INTO TEST AND TRAIN DATA
	X_train,X_test,y_train,y_test,indices_train,indices_test = train_test_split(X_all,y_all,indices,test_size=0.4,random_state=0)


	#CREATING THE MACHINE LEARNING MODEL
	logistic_regression = LogisticRegression()
	logistic_regression.fit(X_train,y_train)
	y_pred=logistic_regression.predict(X_test)

	#LOGIC TO PRINT X_test, y_test and y_pred 
	for i in range(len(indices_test)):
		idx = indices_test[i]
		print(X[idx],y_test[i],y_pred[i])

	#CALCULATING THE PERFORMANCE OF THE MODEL
	print("F1 score is: ",f1_score(y_test, y_pred, average="macro"))
	print("Precision is: ",precision_score(y_test, y_pred, average="macro"))
	print("Recall is: ",recall_score(y_test, y_pred, average="macro")) 
	print("Accuracy is: ",accuracy_score(y_test, y_pred)) 

	#LOGIC FOR GRAPHS
	#graph to visualise predicted data
	x_axis = range(len(y_pred))
	plt.scatter(x_axis, y_pred, label= "stars", color= "green", marker= "*")
	x_threshold = np.linspace(0,15,500)
	y = [0.5]*500
	plt.plot(x_threshold, y, '-g', ls='--' , label = "threshold = 0.5")
	plt.title("Test data")
	plt.xlabel("points")
	plt.ylabel("aggregate")
	plt.show()

  	#graph to visualise all the data
	#y_dataset = df.iloc[:, 9]
	# aggregated = df.loc[y_dataset==1]
	# not_aggregated = df.loc[y_dataset==0]
	# plt.scatter(aggregated.iloc[:, 6], aggregated.iloc[:, 6], s=30, label='aggregated')
	# plt.scatter(not_aggregated.iloc[:, 7], not_aggregated.iloc[:, 7], s=30, label='Not aggregated')
	# plt.legend()
	# plt.show()

	#graph of concentration vs data
	# x_axis = df[['NaCl-concentration(mM)']].values.tolist()
	# plt.scatter(x_axis, y_all, label= "stars", color= "green", marker= "*")
	# plt.title("conc. graph")
	# plt.xlabel("NaCl-concentration")
	# plt.ylabel("aggregation")
	# plt.show()
	
	#CONFUSION MATRIX
	#cm = confusion_matrix(y_test, y_pred)
	#print(cm)

	#LOGIC FOR RANDOM FOREST
	#from sklearn.ensemble import RandomForestRegressor
	#model = RandomForestRegressor(random_state=0).fit(x_train, y_train)

	return




machineLearningLogic()

