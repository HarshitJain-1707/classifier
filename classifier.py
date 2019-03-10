from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import csv

def load_Data():
	data=[]	
	target=[]
	with open("Iris.csv","r") as csvfile:
		csvreader = csv.reader(csvfile)
		fields = csvreader.next()
		for row in csvreader:
			data.append(row[0:5])
			target.append(row[5])
	return data,target


def applyKFold(data,target):
	acc=0	
	kf = KFold(n_splits=10,shuffle=True)
	for train_indices,test_indices in kf.split(data):
		train_data=[]
		train_target=[]
		test_data=[]
		test_target=[]
		for index in train_indices:
			train_data.append(data[index])
			train_target.append(target[index])
		for index in test_indices:
			test_data.append(data[index])
			test_target.append(target[index])
		clf = DecisionTreeClassifier(criterion="entropy")
		clf.fit(train_data,train_target)
 		acc=acc+clf.score(test_data,test_target)
	acc=acc*10
	return acc



#The program starts here
data,target = load_Data()
acc=applyKFold(data,target)
print(acc)

