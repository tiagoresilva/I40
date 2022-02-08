from re import X
import mysql.connector
import numpy as np

connection = mysql.connector.connect(host='localhost',
                                         database='db_pressao',
                                         user='root',
                                         password='user1')
Inputs_SQL = []
if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        cursor.execute("SELECT  * FROM db_pressao.pressao ORDER BY idpressao DESC LIMIT 56")
        
        # fetch all the matching rows 
        result = cursor.fetchall()
        # loop through the rows
        for row in result:
            Inputs_SQL.append(int(row[1]))
        # for elements in Inputs_SQL:
        #     print(elements)
Inputs_SQL = Inputs_SQL[::-1]
print(Inputs_SQL)
            


import pandas as pd
from joblib import dump, load
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import data from csv Files
data_path = r'C:\Users\arware\Desktop\Data'


dataset_train = pd.read_excel(data_path + "\IMM_train.xlsx")
dataset_train = np.array(dataset_train)
dataset_test = pd.read_excel(data_path + "\IMM_test.xlsx")
dataset_test = np.array(dataset_test)

Inputs_train = dataset_train[2:60,1:81]
Inputs_train = Inputs_train.transpose()

Outputs_train = [[]]
x = np.ones((56, 40))
y = np.zeros((56, 40))
Outputs_train = np.concatenate([x,y],axis=1)
Outputs_train = Outputs_train.transpose() 

Inputs_test = dataset_test[2:60,1:61]
with np.printoptions(threshold=np.inf):
    print(Inputs_SQL)
    print(len(Inputs_SQL))
    # for elements in Inputs_test[0:58,0]:
    #     print ("(% s)," % (elements))   



Outputs_test = [[]]
x = np.ones((56, 40))
y = np.zeros((56, 40))
Outputs_test = np.concatenate([x,y],axis=1)
Inputs_test = Inputs_test.transpose()
Outputs_test = Outputs_test.transpose()


# print("Train model_ANN")
# Mdl = MLPClassifier (hidden_layer_sizes=100, max_iter= 2000, activation= 'relu', solver='adam', verbose=True,early_stopping=False, validation_fraction=0.15)
# Mdl = Mdl.fit (Inputs_train, Outputs_train)
# predicted_train_ANN = Mdl.predict (Inputs_train)

# # save the model to disk
filename = 'finalized_model.sav'
# dump(Mdl, open(filename, 'wb'))

# load the model from disk
Mdl = load(open(filename, 'rb'))

# print("Accuracy Score in Training Data (ANN): ", accuracy_score(Outputs_train, predicted_train_ANN))
# print("Accuracy Report in Training Data (ANN): ", classification_report(Outputs_train, predicted_train_ANN))
# print (confusion_matrix(Outputs_train, predicted_train_ANN))

# print("Compute predictions")

predicted_test_ANN = Mdl.predict([Inputs_SQL])
# print(predicted_test_ANN)
# print(np.count_nonzero(predicted_test_ANN))
if (np.count_nonzero(predicted_test_ANN) > 28):
    print("Molde OK!")
else:
    print("Molde defeituoso!")
# print("Accuracy Score in Test Data (ANN): ", accuracy_score(Outputs_test, predicted_test_ANN))
# print("Accuracy Report in Test Data (ANN): ", classification_report(Outputs_test, predicted_test_ANN))
# #print (confusion_matrix(Outputs_test, predicted_test_ANN))



    
  

  


