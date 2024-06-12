import pandas as pd
import pickle
import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Class Labels for Distracted Driver Categories
categories=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']

#Training Images 
datadir=r"C:\Users\shail\Final_Year_Project_MP23CSE053\imgs\train"
if os.path.exists(datadir):
    # Your code to work with the file or directory
    print('Entered the Train Folder ')
else:
    print(f'The path {datadir} does not exist.')

flat_data_arr=[] #input array
target_arr=[] #output array

#Preprocessing of Data
for i in categories:
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(categories.index(i)) 
    print(f'loaded category:{i} successfully')
    
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
x=df.iloc[:,:-1] #input data 
y=df.iloc[:,-1] #output data

sample_size = 10000  # Adjust the sample size as needed
sample_df = df.sample(n=sample_size, random_state=42)
print(sample_df.shape)
x=sample_df.iloc[:,:-1] #input data 
y=sample_df.iloc[:,-1] #output data

#Splitting the Traning and Testing Data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42,stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=800)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#Model Training
svclassifier = SVC(kernel='linear', C=1, decision_function_shape='ovo')
sv=svclassifier.fit(X_train, y_train)


y_pred = svclassifier.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Validation Accuracy:", accuracy)

#pickle.dump(sv,open('ddd_pca.pkl','wb'))

model_data_ovo = {
    'model': sv,
    'scaler': scaler,
    'pca': pca
}

# Save the model data to a pickle file
with open('model_data.pkl', 'wb') as f:
    pickle.dump(model_data_ovo, f)