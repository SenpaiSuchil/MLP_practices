import pandas as pd
import numpy as np
from MLP import *
from plot_3d import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

def tagger(predict):
    tags = (predict > 0.5).astype(int)
    return tags
    
def report(labels,predicted):
    # Accuracy
    accuracy = accuracy_score(labels, predicted)
    print(f'Accuracy: {accuracy}')

    # Matriz de confusión
    conf_matrix = confusion_matrix(labels, predicted)
    print(f'Matriz de Confusión:\n{conf_matrix}')

    # F1-score
    f1 = f1_score(labels, predicted, average='weighted') 
    print(f'F1-score: {f1}')

    # Informe de clasificación
    class_report = classification_report(labels, predicted)
    print(f'Informe de Clasificación:\n{class_report}')

def csv_reader(name,norm=True):
    df = pd.read_csv(name)
    X = df[['x1', 'x2']].values.T
    y = df[['y']].values.T
    if norm:
        min_val = np.min(X)
        max_val = np.max(X)
        normalized = (X - min_val) / (max_val - min_val)
        return normalized,y
    
    return X,y
    

X_blobs,y_blobs=csv_reader(name="csv_files\\blobs.csv")
X_circles,y_circles=csv_reader(name="csv_files\\circles.csv")
X_moons,y_moons=csv_reader(name="csv_files\\moons.csv")
X_xor,y_xor=csv_reader(name="csv_files\\XOR.csv")


blob_mlp=MLP((X_blobs.shape[0],64,32,1))
circles_mlp=MLP((X_circles.shape[0],64,64,32,1))
moons_mlp=MLP((X_moons.shape[0],64,64,32,1))
xor_mlp=MLP((X_xor.shape[0],8,8,4,1))

blob_mlp.fit(X_blobs,y_blobs,epochs=1000)
circles_mlp.fit(X_circles,y_circles,epochs=1000)
moons_mlp.fit(X_moons,y_moons,epochs=1000)
xor_mlp.fit(X_xor,y_xor,epochs=1000)


blob_tags=tagger(blob_mlp.predict(X_blobs))

circles_tags=tagger(circles_mlp.predict(X_circles))

moons_tags=tagger(moons_mlp.predict(X_moons))

xor_tags=tagger(xor_mlp.predict(X_xor))

report(y_blobs[0],blob_tags[0])
MLP_binary_calss_2d(X_blobs,y_blobs,blob_mlp)

report(y_circles[0],circles_tags[0])
MLP_binary_calss_2d(X_circles,y_circles,circles_mlp)

report(y_moons[0],moons_tags[0])
MLP_binary_calss_2d(X_moons,y_moons,moons_mlp)

report(y_xor[0],xor_tags[0])
MLP_binary_calss_2d(X_xor,y_xor,xor_mlp)
