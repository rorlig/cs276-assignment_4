# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/Users/kam/.spyder2/.temp.py
"""
import sklearn.ensemble
import pandas as pd

class RandomForestModel:

	def __init__(self,_testFilePath):
		self.testFilePath = _testFilePath

	def buildModel(self):
		data = pd.read_csv(self.testFilePath)
		data_x = data[:-1].as_matrix()
		data_y=data[-1:].as_matrix()
		model=RandomForestRegressor(n_estimators=1000,random_state=1001)
		model.fit(data_x,data_y)
		return model

	def predict(self,model,data):
		return model.predict(data)


def main(path,testDataPath):
	test=pd.read_csv(testDataPath).as_matrix()
	rf = RandomForestModel(path)
	model=rf.buildModel()
	rf.predict(model,test)

if __name__=='__main__':
	main()



