import numpy as np
import pandas as pd
import random
from pandas import *
import numpy
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
header = ['item_id', 'user_id', 'summary', 'rat2','rating','time','sum1','sum2']
df = pd.read_csv('0921694001508774030_0.csv', sep=',', names=header, index_col=False, low_memory=False)

#print(df)
def getData(M):
	rows = len(M)
	cols = len(M[0])
	for row in M:
		randoms = random.sample(range(35),12)
		for i in randoms:
			row[i] = random.randint(1,5)
	return M
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0] 
#print(df.item_id.unique())
user_dict = {}
item_dict = {}

k = 0
for i in df.item_id.unique():
	if i not in item_dict:
		item_dict[i.lstrip()] = k
		k+=1
	


k = 0
for i in df.user_id.unique():
	if i not in user_dict:
		user_dict[i.lstrip()] = k
		k+=1

#print(user_dict)	
#print(item_dict)
print('Number of users = ' + str(n_users) + ' | Number of food items = ' + str(n_items))


train_data, test_data = cv.train_test_split(df, test_size=0.25)

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
	train_data_matrix[user_dict[line[2].lstrip()], item_dict[line[1].lstrip()]] = line[5]
#print(DataFrame(train_data_matrix))
train_data_matrix = getData(train_data_matrix)
#print(train_data_matrix[0])

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
	#print(line[1], line[2], line[5])
	test_data_matrix[user_dict[line[2].lstrip()], item_dict[line[1].lstrip()]] = line[5]

#print(DataFrame(train_data_matrix))

user_distance = pairwise_distances(train_data_matrix, metric='euclidean')
user_similarity = user_distance.copy()
for i in range(n_users):
	for j in range(n_users):
		user_similarity[i][j] = 1/(1 + user_similarity[i][j])


def mypred(ratings, similarity, user):
	max = 0
	maxItem = 0
	ratings = np.array(ratings)
	similarity = np.array(similarity)
	items = {}
	vals = []
	for item in range(n_items):
		if ratings[user][item] == 0:
			val = np.dot(ratings[:,item],similarity[user,:])
			val /= sum(similarity[user,:])
			items[val] = item
			vals.append(val)
	vals = sorted(vals)[:10]
	recommendations = [items[x] for x in vals]
	return recommendations

	#print(DataFrame(ratings))
	#print(DataFrame(similarity))
def predict(ratings, similarity, type='user'):
	mean_user_rating = ratings.mean(axis=1)
	#You use np.newaxis so that mean_user_rating has same format as ratings
	ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
	pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
	return pred

#user_prediction = predict(train_data_matrix, user_similarity, type='user')
#print(DataFrame(user_prediction))
def getRecommendation(userId):
	u_no = user_dict[userId]
	item_no = mypred(train_data_matrix, user_similarity, u_no)
	print("Recommended items")
	recommendedItems = []
	for k,v in item_dict.items():
		if v in item_no:
			recommendedItems.append(k)
	return recommendedItems
		
