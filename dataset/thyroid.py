

import csv



lines=[]

with open("thyroid.csv","r") as csvreader:
	reader = csv.reader(csvreader)
	for row in reader:
		cur = row[1:]
		cur.append(row[0])
		lines.append(cur)


with open("update_thyroid.csv","w") as csvwriter:
	writer = csv.writer(csvwriter)
	writer.writerows(lines)

