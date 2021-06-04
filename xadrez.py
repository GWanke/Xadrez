import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

def read_input():
	df = pd.read_csv('games.csv')
	#indexR = df[df['rated']==0].index
	#df.drop(indexR,inplace = True)
	#df[colunasScaler] = df[colunasScaler].apply(ss.fit_transform)
	return df

def analiseExploratoria(df):
	#df.hist(bins=50,figsize=(15,10))
	#print(df.info())
	#print(df.nunique())
	#Corr = df.corr()
	#plt.figure(figsize=(15,10))
	#sns.countplot(x='winner', data=df)
	#plt.show()
	#sns.heatmap(df.corr()[['winner']], annot=True)
	#sns.pairplot(df, hue = 'winner', vars = ['white_rating','black_rating','dif_rating'])
	sns.histplot(df['dif_rating'])
	plt.tight_layout()
	plt.show()

def preProcessamento_e_split(df):
	colunasSemUso = ['id','white_id','black_id','last_move_at','created_at','moves']
	df.drop(colunasSemUso, axis=1,inplace=True)
	df['dif_rating'] = df['white_rating'] - df['black_rating']
	colunasObject = ['victory_status','winner','increment_code','opening_eco','opening_name']
	le = LabelEncoder()
	df[colunasObject] = df[colunasObject].apply(le.fit_transform)
	'''Metodo responsavel pelo pre processamento do dataframe, bem como o split em treino e teste.'''
	x = df[df.columns.difference(['winner'])]
	y = df['winner']
	X_train, X_test, y_train, y_test = train_test_split(
                 x, y,
                 test_size = 0.30, random_state = 101)
	'''Upsampling, pois o y possui poucas entradas com "draw".'''
	sm = SMOTE(random_state = 101)
	X_train,y_train = sm.fit_resample(X_train,y_train)
	'''Usando um scaler para normalizar os valores para [0,1]'''
	mm = MinMaxScaler()
	X_train = mm.fit_transform(X_train)
	X_test = mm.fit_transform(X_test)
	return X_train, X_test, y_train, y_test

def gridSVM(X_train, X_test, y_train):
	'''Metodo responsavel por hiperparametrizar para o modelo SVM'''
	param_grid_svm = [
					{'C': [1, 10, 20], 'gamma': [0.01,0.001,'scale','auto'], 'kernel': ['linear','rbf','sigmoid']},
					{'C': [1, 10, 20], 'degree': [2,3], 'gamma': [0.01,0.001,'scale','auto'], 'kernel': ['poly']}
	]
	grid = GridSearchCV(SVC(), param_grid_svm, verbose = 3,scoring = 'accuracy')
	grid.fit(X_train, y_train)
	print(f"O melhor parametro para o SVM encontrado foi de: {grid.best_params_}")
	return grid.predict(X_test)

def gridKNN(X_train, X_test, y_train):
	'''Metodo responsavel por hiperparametrizar para o modelo kNN'''
	param_grid_knn = [
					{'n_neighbors': [1,5, 10,20,40,1000], 'weights': ['distance','uniform'],'leaf_size': [1,5,10,20,40,1000], 
					 'p':[1,2]},
	]
	grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, verbose = 3, scoring = 'accuracy')
	grid.fit(X_train, y_train)
	print(f"O melhor parametro para o kNN encontrado foi de: {grid.best_params_}")
	return grid.predict(X_test)	

def gridRandomForest(X_train, X_test, y_train):
	param_grid_rf = [
				{'n_estimators':[100,200],'max_features':['auto','sqrt'],'max_depth':[10,50,100],
				 'min_samples_split':[2,5,10],'min_samples_leaf': [1, 2, 4],'bootstrap':[True,False]}
	]
	grid = GridSearchCV(RandomForestClassifier(), param_grid_rf, verbose = 3, scoring = 'accuracy')
	grid.fit(X_train, y_train)
	print(f"O melhor parametro para o SVM encontrado foi de: {grid.best_params_}")
	return grid.predict(X_test)
def main():
	'''2 == white
	   0 == black
	   1 == drawn'''
	df = read_input()
	#analiseExploratoria(df)
	X_train, X_test, y_train, y_test = preProcessamento_e_split(df)
	#rfPred = gridRandomForest(X_train, X_test, y_train)
	#svmPred = gridSVM(X_train, X_test, y_train)
	knnPred = gridKNN(X_train, X_test, y_train)
	print(classification_report(y_test, knnPred))

if __name__ == '__main__':
	main()