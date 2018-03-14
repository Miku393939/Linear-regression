import scipy.io as scio
import matplotlib.pyplot as plt
# I don use the sklearn anymore
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score
# I don use the sklearn anymore
import pandas as pds
import numpy as np

data_load = scio.loadmat("detroit.mat")
data = data_load["data"]
pds_data = pds.DataFrame(data)
Y = np.matrix(pds_data[9]).T
# print(Y)
pds_data[10] = 1 #using linear regression, the first column of X should be 1
# print(pds_data)
error = []
for i in range(1,8):
    X_column = np.matrix(pds_data[[10,0,8,i]]) #using linear regression, the first column of X should be 1
    # print(X_column)
    X_column_dot_T = np.dot(X_column.T, X_column)
    X_column_dot_T_inverse = np.linalg.inv(X_column_dot_T)
    X_column_dot_T_inverse_dot_XT = np.dot(X_column_dot_T_inverse,X_column.T)
    X_column_dot_T_inverse_dot_XT_dot_Y = np.dot(X_column_dot_T_inverse_dot_XT,Y)
    Y_prime = np.dot(X_column,X_column_dot_T_inverse_dot_XT_dot_Y)
    Y_List = Y.tolist()
    Y_prime_List = Y_prime.tolist()
    sq_sum = 0
    for j in range(0,len(Y_List)):
        sq_sum = sq_sum + (Y_List[j][0] - Y_prime_List[j][0])**2
    # print(np.sqrt(sq_sum))
    error.append(sq_sum)
print(error)
plt.clf
plt.figure(1, figsize=(8, 4))
plt.plot(range(0,7), error)
plt.xticks(range(0,7), ["UEMP","MAN","LIC","GR","NMAN","GOV","HE"])
plt.show()




# The belows use sklearn library, pls ignore it
#
# def main():
#     regression = linear_model.LinearRegression()
#     data = scipy.io.loadmat('detroit.mat')['data']
#     Y = []
#     X = []
#     X_column1 = []
#     X_column2 = []
#     X_column3 = []
#     X_column4 = []
#     X_column5 = []
#     X_column6 = []
#     X_column7 = []
#     for i in data:
#         Y.append(i[9])
#         X.append(i[0:9])
#     for i in range(0,13):
#         X_column1.append([X[i][0], X[i][1], X[i][8]])
#         X_column2.append([X[i][0], X[i][2], X[i][8]])
#         X_column3.append([X[i][0], X[i][3], X[i][8]])
#         X_column4.append([X[i][0], X[i][4], X[i][8]])
#         X_column5.append([X[i][0], X[i][5], X[i][8]])
#         X_column6.append([X[i][0], X[i][6], X[i][8]])
#         X_column7.append([X[i][0], X[i][7], X[i][8]])
#
#     print(X_column1)
#     print(Y)
#     regression.fit(X_column1, Y)
#     pred1 = regression.predict(X_column1)
#
#     regression.fit(X_column2, Y)
#     pred2 = regression.predict(X_column2)
#
#     regression.fit(X_column3, Y)
#     pred3 = regression.predict(X_column3)
#
#     regression.fit(X_column4, Y)
#     pred4 = regression.predict(X_column4)
#
#     regression.fit(X_column5, Y)
#     pred5 = regression.predict(X_column5)
#
#     regression.fit(X_column6, Y)
#     pred6 = regression.predict(X_column6)
#
#     regression.fit(X_column7, Y)
#     pred7 = regression.predict(X_column7)
#
#     print("mean squared error of column 1 is %.2f"
#           % mean_squared_error(Y, pred1))
#     print('variance score of column 1 is %.2f' % r2_score(Y, pred1))
#
#     print("mean squared error of column 2 is %.2f"
#           % mean_squared_error(Y, pred2))
#     print('variance score of column 2 is %.2f' % r2_score(Y, pred2))
#
#     print("mean squared error of column 3 is %.2f"
#           % mean_squared_error(Y, pred3))
#     print('variance score of column 3 is %.2f' % r2_score(Y, pred3))
#
#     print("mean squared error of column 4 is %.2f"
#           % mean_squared_error(Y, pred4))
#     print('variance score of column 4 is %.2f' % r2_score(Y, pred4))
#
#     print("mean squared error of column 5 is %.2f"
#           % mean_squared_error(Y, pred5))
#     print('variance score of column 5 is %.2f' % r2_score(Y, pred5))
#
#     print("mean squared error of column 6 is %.2f"
#           % mean_squared_error(Y, pred6))
#     print('variance score of column 6 is %.2f' % r2_score(Y, pred6))
#
#     print("mean squared error of column 7 is %.2f"
#           % mean_squared_error(Y, pred7))
#     print('variance score of column 7 is %.2f' % r2_score(Y, pred7))
#
#     C1 = [row[1] for row in X_column1]
#     C2 = [row[1] for row in X_column2]
#     C3 = [row[1] for row in X_column3]
#     C4 = [row[1] for row in X_column4]
#     C5 = [row[1] for row in X_column5]
#     C6 = [row[1] for row in X_column6]
#     C7 = [row[1] for row in X_column7]
#
#     plt.scatter(C1, Y, color='black')
#     plt.plot(C1, pred1, color='blue', linewidth=3)
#     plt.show()
#
#     plt.scatter(C2, Y, color='black')
#     plt.plot(C2, pred1, color='blue', linewidth=3)
#     plt.show()
#
#     plt.scatter(C3, Y, color='black')
#     plt.plot(C3, pred1, color='blue', linewidth=3)
#     plt.show()
#
#     plt.scatter(C4, Y, color='black')
#     plt.plot(C4, pred1, color='blue', linewidth=3)
#     plt.show()
#
#     plt.scatter(C5, Y, color='black')
#     plt.plot(C5, pred1, color='blue', linewidth=3)
#     plt.show()
#
#     plt.scatter(C6, Y, color='black')
#     plt.plot(C6, pred1, color='blue', linewidth=3)
#     plt.show()
#
#     plt.scatter(C7, Y, color='black')
#     plt.plot(C7, pred1, color='blue', linewidth=3)
#     plt.show()
#
# if __name__ == '__main__':
#
#     main()
