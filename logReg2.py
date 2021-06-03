#TO CALCULATE IF PROTEIN WILL AGGREGATE IN INPUT CONDITIONS
import os

#MACHINE LEARNING IMPORTS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


#DICTIONARY CONTATING START AND END INDEX OF THE FEATURES OF TRAINED PROTEINS.
#THIS FUNCTION TAKE PROTEIN NAME AS INPUT AND GIVES START AND END INDICES OF THAT PROTEIN'S FEATURES.
def idxlogic(name):
	#protein : [startidx,endidx]
	d = {'1': [0,8],'2': [8,16],'3': [16,20],'4': [20,23],'5': [23,26],'6': [26,29] }
	return d[name]


#MACHINE LEARNING LOGIC
#IT GETS STARTIDX, ENDIDX FROM THE ABOVE FUNCTION AND X FROM THE INPUT
def machineLearningLogic(startidx,endidx,x):
	#IMPORTING FILE
	currentDir = os.getcwd()
	currentFileCSV = currentDir +"//" + "ddp_database.csv"
	df = pd.read_csv(currentFileCSV)

	#CREATING THE FEATURE VECTOR AND RESULT VECTOR 
	X_all = df[['NaCl-concentration(mM)','pH']].values.tolist()
	X = X_all[:][startidx:endidx]
	Y_all = df['Aggregation'].values.tolist()
	Y= Y_all[:][startidx:endidx]

	#SPLITING THE DATASET INTO TEST AND TRAIN DATA
	X_train,X_test,y_train,y_test = train_test_split(X_all,Y_all,test_size=0.3,random_state=0)


	#CREATING THE MACHINE LEARNING MODEL
	logistic_regression = LogisticRegression()
	logistic_regression.fit(X_train,y_train)

	#y_pred=logistic_regression.predict(X_test)
	#score = LogisticRegression.score(X_test, y_test)
	#print(y_test)

	#CALCULATING THE PREDICTION FOR THE INPUT
	y_ans = logistic_regression.predict(x)
	return y_ans




#INPUT LOGIC
print("Protein Aggregation predictor")
print("1.Lysozyme; 2.Chymotrypsinogen; 3.Chymotrypsin ; 4.Lactoferrin ; 5.Catalase ; 6.Concanavalin-A")
name = input("enter the number of protein from above: ")
print("There are 2 required inputs")
print("")
#x1 = float(input("enter SASA value: ")) #6447
#x2 = float(input("enter Solvation-energy value: ")) #-1823
x3 = float(input("enter NaCl-concentration value(Eg 50): ")) #40-500
x4 = float(input("enter pH value(Eg 5): ")) #4.5 - 3
x = [[x3,x4]]
startidx, endidx = idxlogic(name)
ans = machineLearningLogic(startidx,endidx,x)

#OUTPUT LOGIC
print("")
print('...')
if ans==1:
	print("Yes, protein will aggregate")
else:
	print("No, protein will not aggregagte")
print("...")


