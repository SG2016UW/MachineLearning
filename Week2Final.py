# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 19:33:26 2016

@author: gujju
"""

import pandas as pd
import numpy as np
import xml.etree.cElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
import codecs
import sys

#FUNCTION WHICH WILL PRINT XML
def write_contents(pointer):
    user_id = test_profile_df['userid'].iloc[pointer-1]
    root.set(items[0],str(user_id))
    root.set(items[1],str(target_age))
    if(output_gender_predicted[pointer-1] == 1.0):
        root.set(items[2],'female')
    else:
        root.set(items[2],'male')
    root.set(items[3],str(output_traits_predicted_df.loc[pointer-1,'ext']))
    root.set(items[4],str(output_traits_predicted_df.loc[pointer-1,'neu']))
    root.set(items[5],str(output_traits_predicted_df.loc[pointer-1,'agr']))
    root.set(items[6],str(output_traits_predicted_df.loc[pointer-1,'con']))
    root.set(items[7],str(output_traits_predicted_df.loc[pointer-1,'ope']))
    tree = ET.ElementTree(root)
    output_path = output_folder + user_id + '.xml'
    tree.write(output_path, short_empty_elements = True)
    

input_training_path = "/data/training"
#FIRST PATH which will be input path to the test data, which can be obtained from sys.argv[0]
input_test_path = sys.argv[2]
training_profile_df = pd.read_csv(input_training_path + '/profile/profile.csv')
test_profile_df = pd.read_csv(input_test_path + '/profile/profile.csv')

#######################################################################################################
#COUNT VECTORIZER (NAIVE BAIS) FOR PREDICTING GENDER
#######################################################################################################

training_gender_df = training_profile_df.loc[:,['userid', 'gender']]
userIds_train = training_gender_df['userid']
input_training_text_arr = []
input_train_text_loc = input_training_path + "/text/"
for userId in userIds_train:
    file = input_train_text_loc + userId + ".txt"
    #print('filename : ' + file)
    #fo = open(file, "r")
    with codecs.open(file, 'r', encoding='utf-8', errors='ignore') as fo:
    	 nightmare = fo.read()
    input_training_text_arr.append(nightmare)    
    fo.close()
    
# Training a Naive Bayes model
count_vect = CountVectorizer()
input_status_train = count_vect.fit_transform(input_training_text_arr)
input_gender_train = training_gender_df['gender']
clf = MultinomialNB()
clf.fit(input_status_train, input_gender_train)

test_gender_df = test_profile_df.loc[:,['userid','gender']]
userIds_test = test_gender_df['userid']
input_test_text_arr = []
input_test_text_Loc = input_test_path + "/text/"
for userId in userIds_test:
    filename = input_test_text_Loc + userId + ".txt"
    #fo = open(input_test_text_Loc + filename, "r+")
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as fo:
    	 nightmare = fo.read()
    fo.close()
    input_test_text_arr.append(nightmare)  
    
input_status_test = count_vect.transform(input_test_text_arr)
output_gender_predicted = clf.predict(input_status_test)

#######################################################################################################
#COUNT VECTORIZER (NAIVE BAIS) FOR PREDICTING GENDER
#######################################################################################################

#######################################################################################################
#LINEAR REGRESSION FOR PREDICITING PERSONALITY TRAIT
#######################################################################################################


input_training_path = "/data/training"
input_train_liwc_loc = input_training_path + "/LIWC/LIWC.csv"

training_traits_df = pd.read_csv(input_train_liwc_loc)
training_traits_df.columns = training_traits_df.columns.str.lower()

training_traits_df = pd.merge(training_traits_df, training_profile_df, how='inner', on='userid')
input_test_liwc_loc = input_test_path + "/LIWC/LIWC.csv"

test_traits_df = pd.read_csv(input_test_liwc_loc, sep=',')
test_traits_df.columns = test_traits_df.columns.str.lower()
big5 = ['ope','ext','con','agr','neu']

feature_list = [x for x in training_traits_df.columns.tolist()[:] if not x in big5]
feature_list.remove('userid')
feature_list.remove('age')
feature_list.remove('gender')
feature_list.remove('Unnamed: 0')
feature_list.remove('seg')

sLength = len(test_traits_df['userid'])

for trait in big5: 
    test_traits_df[trait] = pd.Series(np.random.randn(sLength), index=test_traits_df.index)   
    input_train_liwc = training_traits_df[feature_list]
    input_train_traits = training_traits_df[trait]
    regr = linear_model.LinearRegression()
    regr.fit(input_train_liwc, input_train_traits)
    input_test_liwc = test_traits_df[feature_list]
    output_traits_predicted = regr.predict(input_test_liwc)
    test_traits_df[trait] = output_traits_predicted
    
output_traits_predicted_df = test_traits_df[['ope','ext','con','agr','neu']]    

#######################################################################################################
#LINEAR REGRESSION FOR PREDICITING PERSONALITY TRAIT
#######################################################################################################

#######################################################################################################
#BASELINE ALGORITHM FOR PREDICTING AGE
#######################################################################################################
target_age = "xx-24"
age_series = training_profile_df['age']
first_AG = age_series[age_series < 25].count()
second_AG = age_series[(age_series >= 25) & (age_series < 35)].count()
third_AG = age_series[(age_series >= 35) & (age_series < 50)].count()        
fourth_AG = age_series[age_series >= 50].count()
list1 = [first_AG, second_AG, third_AG, fourth_AG]
max_AG = max(list1)
if(max_AG == first_AG):
    target_age = 'xx-24'
elif(max_AG == second_AG):
    target_age = '25-34'
elif(max_AG == third_AG):
    target_age = '35-49'
else:
    target_age = '50-xx'
    
#######################################################################################################
#BASELINE ALGORITHM FOR PREDICTING AGE
#######################################################################################################    
    
#SECOND PATH which will be output path, which can be obtained from sys.argv[1]
output_folder = sys.argv[4]
items = np.array(['id','age_group','gender','extrovert','neurotic','agreeable','conscientious','open'])
# gender -> output_gender_predicted
# age_group -> output_age_predicted
# ext -> output_ext_predicted
# neu -> output_neu_predicted
# agr -> output_agr_predicted
# con -> output_con_predicted
root = ET.Element('user')
files_count = len(test_profile_df.index)
for k in range(1, files_count+1):
    write_contents(k)
