import pandas as pds
import numpy as np

training_data = pds.read_csv('crx.data.training',header= None)

def cal_distance(List1,List2):
    my_distance = [x1 - x2 for (x1, x2) in zip(List1, List2)]
    my_return_x = np.array(my_distance)
    return np.linalg.norm(my_return_x)

for i in range(0,13):
    if i != 1 and i != 2 and i != 7 and i != 10:
        training_data[i] = training_data[i].replace('?', training_data[i].mode()[0])
training_data = training_data.replace('?',np.nan)

for i in range(1,15):
    if i == 1 or i == 2 or i == 7 or i == 10 or i == 13 or i == 14:
        training_data[i] = training_data[i].astype(float)
        training_data[i] = training_data[i].replace(np.nan, training_data[i].mean())
print(training_data)
training_data.to_csv('crx.training.processed')
training_data_numerical = training_data[[1,2,7,10,13,14]]
training_data_numerical_matrix = training_data_numerical.as_matrix()

testing_data = pds.read_csv('crx.data.testing',header= None)
for i in range(0,13):
    if i != 1 and i != 2 and i != 7 and i != 10:
        testing_data[i] = testing_data[i].replace('?', testing_data[i].mode()[0])
testing_data = testing_data.replace('?',np.nan)

for i in range(1,15):
    if i == 1 or i == 2 or i == 7 or i == 10 or i == 13 or i == 14:
        testing_data[i] = testing_data[i].astype(float)
        testing_data[i] = testing_data[i].replace(np.nan, testing_data[i].mean())

testing_data.to_csv('crx.testing.processed')

testing_data_numerical_matrix = testing_data[[1,2,7,10,13,14]].as_matrix()

distance_matrix = [[0 for x in range(0, 552)] for y in range(0, 138)]

for i in range(0, 138):
    for j in range(0, 552):
        distance_matrix[i][j] = cal_distance(testing_data_numerical_matrix[i].tolist(),training_data_numerical_matrix[j].tolist())

def get_knn(k):
    temp_result=[]
    fina_result=[]
    for i in range(0, 138):
        temp_result.append(np.array(distance_matrix[i]).argsort()[:k].tolist())
        positive_count = 0
        negative_count = 0
        for index in temp_result[i]:
            if training_data[15][index] == '+':
                positive_count = positive_count + 1
            elif training_data[15][index] == '-':
                negative_count = negative_count + 1
        if positive_count > negative_count:
            fina_result.append('+')
        else:
            fina_result.append('-')
    return fina_result

print("For crx data:")
for i in range(3,8):
    testing_data[16] = get_knn(i)
    testing_data.to_csv('my_test.csv')

    count = 0
    for j in range(0, 138):
	    if testing_data[15][j] == testing_data[16][j]:
	        count += 1
    accuracy = count / 138
    print('When k = ' + str(i)+ ', the accuracy is ' + str(round(accuracy,3)))


lense_train = pds.read_csv('lenses.training',header= None)
lense_train_matrix = lense_train.as_matrix()
lense_test = pds.read_csv('lenses.testing',header= None)
lense_test_matrix = lense_test.as_matrix()

distance_matrix_lense = [[0 for x in range(0,18)] for y in range(0,6)]
for i in range(0,6):
    for j in range(0,18):
        distance_matrix_lense[i][j] = cal_distance(lense_test_matrix[i].tolist(),lense_train_matrix[j].tolist())

def get_knn_lense(k):
    temp_result = []
    fina_result = []
    for i in range(0,6):
        temp_result.append(np.array(distance_matrix_lense[i]).argsort()[:k].tolist())
        classify = [0,0,0,0]
        for j in temp_result[i]:
            if lense_train[4][j] == 1:
                classify[1] = classify[1] + 1
            elif lense_train[4][j] == 2:
                classify[2] = classify[2] + 1
            else:
                classify[3] = classify[3] + 1
        fina_result.append(classify.index(max(classify)))
    return fina_result

print("For lense data:")
for i in range(1,5):
    lense_test[5] = get_knn_lense(i)
    lense_test.to_csv('lense.output')
    count = 0
    for j in range(len(lense_test)):
        if lense_test[4][j] == lense_test[5][j]:
            count = count + 1
    accuracy_rate = count / 6
    print('When k = ' + str(i)+ ' ,the accuracy is ' + str(round(accuracy_rate,3)))
