from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import csv
from sklearn.naive_bayes import GaussianNB

def load_Data(dataset):
	data=[]	
	target=[]
	dataset = "dataset/" + dataset+ ".csv"
	with open(dataset,"r") as csvfile:
		csvreader = csv.reader(csvfile)
		fields = csvreader.next()
		for row in csvreader:
			lst=[]
			for i in range(len(row)-1):
				lst.append(eval(row[i]))
			data.append(lst)
			target.append(row[len(row)-1])
	return data,target


def removenoise(data,target):
	gnb = GaussianNB()
	gnb.fit(data,target)
	predicted = gnb.predict(data)
	ln = len(target)
	filtered_data=[]
	filtered_target=[]
	for i in range(ln):
		if target[i] == predicted[i]:
			filtered_data.append(data[i])
			filtered_target.append(target[i])
	#print(len(target),len(filtered_target))
	return filtered_data,filtered_target


def applyKFold(data,target,remove_noise):
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
		if remove_noise == True:
			train_data,train_target=removenoise(train_data,train_target)
		clf.fit(train_data,train_target)
 		acc=acc+clf.score(test_data,test_target)
	acc=acc*10
	return acc


#The program starts here
datasets=["haberman","bupa","iris","breast","ecoli","cleveland","pima","hepatitis","thyroid"]

for dataset in datasets:
	data,target = load_Data(dataset)
	acc=applyKFold(data,target,remove_noise=False)
	print(dataset," with noise ",acc)
	acc=applyKFold(data,target,remove_noise=True)
	print(dataset," without noise ",acc)

