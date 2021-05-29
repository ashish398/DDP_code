import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os

def idxlogic(name):
	#protein : [startidx,endidx]
	d = {'1': [0,8],'2': [8,16],'3': [16,20],'4': [20,23],'5': [23,26],'6': [26,29] }
	return d[name]


#Machine Learning Logic for soultion
def machineLearningLogic(startidx,endidx,x):
	currentDir = os.getcwd()
	currentFileCSV = currentDir +"//" + "ddp.csv"
	df = pd.read_csv(currentFileCSV)

	X_all = df[['NaCl-concentration(mM)','pH']].values.tolist()
	X = X_all[:][startidx:endidx]

	Y_all = df['Aggregation'].values.tolist()
	Y= Y_all[:][startidx:endidx]


	X_train,X_test,y_train,y_test = train_test_split(X_all,Y_all,test_size=0.3,random_state=0)



	logistic_regression = LogisticRegression()
	logistic_regression.fit(X_train,y_train)
	y_pred=logistic_regression.predict(X_test)
	# Use score method to get accuracy of model
	score = logistic_regression.score(X_test, y_test)
	print("the score is: " + str(score*100))
	#score = LogisticRegression.score(X_test, y_test)

	#scores = cross_val_score(logistic_regression, X_train_imputed, y_train, cv=10)
	#print("Cross-Validation Accuracy Scores", scores)
	
	
	#score = accuracy_score(y_train, y_pred)
	#print("Test Accuracy Score", score)
	#print(y_pred.tolist())
	#print(y_test)
	y_ans = logistic_regression.predict(x)
	return y_ans




#Display Logic
print("Protein Aggregation predictor")
print("1.Lysozyme; 2.Chymotrypsinogen; 3.Chymotrypsin ; 4.Lactoferrin ; 5.Catalase ; 6.Concanavalin-A")
name = input("enter the number of protein from above: ")
print("There are 2 required inputs")
print("")
#x1 = float(input("enter SASA value: ")) #6447
#x2 = float(input("enter Solvation-energy value: ")) #-1823
x3 = float(input("enter NaCl-concentration value: ")) #40-500
x4 = float(input("enter pH value: ")) #4.5 - 3
x = [[x3,x4]]
startidx, endidx = idxlogic(name)
print(startidx,endidx)
print("hey")
ans = machineLearningLogic(startidx,endidx,x)


print("")
print('...')
if ans==1:
	print("No, protein will not aggregate")
else:
	print("Yes, protein will aggregagte")
print("...")


