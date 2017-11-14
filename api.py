import json
from pprint import pprint
from matrix_tru import getRecommendation
from get_food import ReadAsin
 
userId = input("enter user id\n")
recommendations = getRecommendation(userId)
print(recommendations)

ReadAsin(recommendations)

#data = json.load(open('data.json'))
#pprint(data)

