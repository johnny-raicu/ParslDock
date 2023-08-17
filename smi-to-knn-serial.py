import sys
import getopt
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn import metrics

#import sklearn
from sklearn.neighbors import KNeighborsRegressor

#import pandas as pd
import matplotlib.pyplot as plt

#import time
import math

def euclidean_distance(point1, point2):
    # Ensure both points have exactly two coordinates (x, y)
    if len(point1) != 2 or len(point2) != 2:
        raise ValueError("Both points must have exactly two coordinates (x, y)")
    
    # Calculate the squared differences between coordinates
    squared_diff_x = (point1[0] - point2[0]) ** 2
    squared_diff_y = (point1[1] - point2[1]) ** 2
    
    # Calculate the Euclidean distance
    distance = math.sqrt(squared_diff_x + squared_diff_y)
    
    return distance

def test_model(knn_classifier, dataset_path, nrows, batch_size, fingerprint_depth, fingerprint_size, percentile, neighbors, parallelism, threshold, log_file_name, DEBUG):
    
    log_file = open(log_file_name+"_predicted.txt","w")
    log_file.write("operation,ligand,smile,time,score,predicted_score\n")
    log_file.flush()

    print("test_model() starting...")

    print("reading data from disk...")
    df = pd.read_csv(dataset_path,na_values=['None'],nrows=nrows)
    print("drop NA values...")
    df = df.dropna()
    print("convert smiles to mol...")
    molecules = []
    
    mol_num = len(df['smile'])
    mol_index = 0
    for smile in df['smile']:
    	molecule = Chem.MolFromSmiles(smile)
    	molecules.append(molecule)
    	mol_index +=1
    	if mol_index%1000 == 0:
    		print("converting from smile to molecule",mol_index,"/",mol_num)
    	
    df['mol'] = molecules
    
    print("convert mol to fingerprints...")
    
    fingerprints = []
    fp_num = len(df['mol'])
    fp_index = 0
    for x in df['mol']:
    	fingerprint = AllChem.GetMorganFingerprintAsBitVect(x, fingerprint_depth, nBits=fingerprint_size)
    	fingerprints.append(fingerprint)
    	fp_index +=1
    	if fp_index%1000 == 0:
    		print("generating fingerprint",fp_index,"/",fp_num)
    		
    df['fp'] = fingerprints    
    
    print("convert fingerprint dataframe to a list")
    X_test = list(df['fp'])
    
    X_test_ligand = list(df['ligand'])
    X_test_smile = list(df['smile'])
    X_test_time = list(df['time'])
    X_test_score = list(df['score'])
    
    print("initialize predict data frame")
    df["predict"] = df['score'].astype(int)
    df["predict"] = df["predict"].fillna(0)
    y_pred = list(df['predict'].astype(int))

    print("convert scores to integer labels")
    y_test = list(df['score'].astype(int))

    dataset_size = len(df)

    print("processing dataset size",dataset_size)

    X_test_scaled = X_test
    print("threshold",threshold)
    recall_low = 0
    low_count = 0
    recall_high = 0
    high_count = 0
    threshold_int = int(threshold)
    print("begin testing")
    operation = ""
    for index in range(0,len(X_test_scaled),batch_size):
    	if index >= dataset_size:
    		print(index,"error index is larger than",dataset_size)
    	else:
    		bit_vectors = np.array(X_test_scaled[index:index+batch_size])
    		predicted_scores = knn_classifier.predict(bit_vectors)
    		for i in range(0,len(predicted_scores)):
    			predicted_score = predicted_scores[i]
    			y_pred[index+i] = predicted_score
    			if predicted_score<=threshold_int and y_test[index+i] <=threshold_int:
    				recall_low += 1
    				low_count += 1
    				operation = "knn_pred_tp"
    			elif predicted_score>threshold_int and y_test[index+i] >threshold_int:
    				recall_high +=1
    				high_count += 1
    				operation = "knn_pred_tn"
    			elif predicted_score<=threshold_int and y_test[index+1] >threshold_int:
    				high_count += 1
    				#log_file.write("knn_pred_fp"+","+str(X_test_ligand[index+i])+","+str(X_test_smile[index+i])+","+str(X_test_time[index+i])+","+str(X_test_score[index+i])+","+str(predicted_scores[i][0])+"\n")
    				operation = "knn_pred_fp"
    			else:
    				low_count += 1
    				operation = "knn_pred_fn"
    			
    			if (DEBUG):
    				print(operation+","+str(X_test_ligand[index+i])+","+str(X_test_smile[index+i])+","+str(X_test_time[index+i])+","+str(X_test_score[index+i])+","+str(predicted_score))
    			log_file.write(operation+","+str(X_test_ligand[index+i])+","+str(X_test_smile[index+i])+","+str(X_test_time[index+i])+","+str(X_test_score[index+i])+","+str(predicted_score)+"\n")
    		
    		if index%1000 ==0:
    			print(index,"/",dataset_size,"accuracy",(recall_low+recall_high)*100.0/(low_count+high_count))
    			log_file.flush()

    print("finished testing")
    recall_l = recall_low*100.0/low_count
    recall_h = recall_high*100.0/high_count
    
    point1 = (recall_l, recall_h)
    point2 = (100, 100)
    perf = 100 - euclidean_distance(point1, point2)
    print("Performance:", perf)    
    
    accuracy = (recall_low+recall_high)*100.0/(low_count+high_count)
    
    log_file.flush()
    log_file.close()
    return (high_count+low_count), recall_l, recall_h, perf, accuracy

def run_model(dataset_path, nrows, fingerprint_depth, fingerprint_size, percentile, neighbors, parallelism, DEBUG):

    df = pd.read_csv(dataset_path,na_values=['None'],nrows=nrows)
    
    print("run_model():",len(df))
    
    df = df.dropna()
    
    molecules = []
    
    mol_num = len(df['smile'])
    mol_index = 0
    for smile in df['smile']:
    	molecule = Chem.MolFromSmiles(smile)
    	molecules.append(molecule)
    	mol_index +=1
    	if mol_index%1000 == 0:
    		print("converting from smile to molecule",mol_index,"/",mol_num)
    	
    df['mol'] = molecules    
    
    fingerprints = []
    fp_num = len(df['mol'])
    fp_index = 0
    for x in df['mol']:
    	fingerprint = AllChem.GetMorganFingerprintAsBitVect(x, fingerprint_depth, nBits=fingerprint_size)
    	fingerprints.append(fingerprint)
    	fp_index +=1
    	if fp_index%1000 == 0:
    		print("generating fingerprint",fp_index,"/",fp_num)
    		
    df['fp'] = fingerprints    

    # Extract features and target variable
    X = list(df['fp'])

    threshold = df['score'].quantile(percentile/100.0)
    df['label'] = df['score'].apply(lambda score: 'best' if score <= threshold else 'worst')

    count_best = (df['label'] == 'best').sum()
    count_worst = (df['label'] == 'worst').sum()

    best_entries = df[df['label'] == 'best']
    worst_entries = df[df['label'] == 'worst']
    worst_entries_random = df.sample(n=count_best, random_state=42)

    df2 = pd.concat([best_entries, worst_entries_random], ignore_index=True)

    bins = [float('-inf'), threshold, float('inf')]
    labels = ['best', 'worst']

    y = list(df2['score'].astype(int))

    X = list(df2['fp'])
    dataset_size = len(df2)

    print("processing dataset size",dataset_size)


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features (optional but often recommended for KNN)
    X_train_scaled = X_train
    X_test_scaled = X_test

    # Create a KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=neighbors, algorithm='ball_tree', weights='distance', metric='euclidean', n_jobs=parallelism)

    # Train the classifier
    knn_classifier.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    print("threshold",threshold)
    recall_low = 0
    low_count = 0
    recall_high = 0
    high_count = 0
    threshold_int = int(threshold)
    X_scaled = X
    for index in range(0,len(X_scaled)):
        bit_vector = np.array(X_scaled[index])
        predicted_score = knn_classifier.predict(bit_vector.reshape(1, -1))
        if predicted_score[0]<=threshold_int and y[index] <=threshold_int:
            recall_low += 1
            low_count += 1
        elif predicted_score[0]>threshold_int and y[index] >threshold_int:
        	recall_high +=1
        	high_count += 1
        elif predicted_score[0]<=threshold_int and y[index] >threshold_int:
        	high_count += 1
        else:
        	low_count += 1

    recall_l = recall_low*100.0/low_count
    recall_h = recall_high*100.0/high_count
    
    point1 = (recall_l, recall_h)
    point2 = (100, 100)
    perf = 100 - euclidean_distance(point1, point2)
    print("Performance:", perf)    
    
    accuracy = (recall_low+recall_high)*100.0/(low_count+high_count)
    
    return knn_classifier, (high_count+low_count), threshold, recall_l, recall_h, perf, accuracy

def main(argv):
    DEBUG = False
    if len(argv) != 10:
        print ('python smi-to-knn-serial.py <smi_dock_file> <num_samples_build> <num_samples_test> <batch_size> <fingerprint_depth> <fingerprint_size> <percentile> <neighbors> <log_file> <debug>')
        sys.exit(2)
    else:
        smi_dock_file = argv[0]
        num_samples_build = int(argv[1])
        num_samples_test = int(argv[2])
        batch_size = int(argv[3])
        fp_depth = int(argv[4])
        fp_size = int(argv[5])
        percentile = float(argv[6])
        neighbors = int(argv[7])
        #parallelism = int(argv[8])
        parallelism = 16
        log_file_name = argv[8]
        if argv[9] == "True":
            DEBUG = True
        elif argv[9] == "False":
            DEBUG = False
        else:
            print("wrong debug flag:",argv[9])
            sys.exit(2)
            
    log_file = open(log_file_name,"a")
    start_time = time.time()
    if DEBUG:
    	print("starting building classifier...")
    knn_classifier, dataset_size,threshold,recall_l, recall_h, perf, accuracy = run_model(smi_dock_file, num_samples_build, fp_depth, fp_size, percentile, neighbors, parallelism, DEBUG)
    if DEBUG:
    	print("finished building classifier!")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("knn-unbalanced-simple3-scaled-train-test,"+smi_dock_file+","+str(elapsed_time)+","+str(num_samples_build)+","+str(dataset_size)+","+str(round(percentile,2))+","+str(threshold)+","+str(neighbors)+","+str(fp_depth)+","+str(fp_size)+","+str(round(recall_l,2))+","+str(round(recall_h,2))+","+str(round(perf,2))+","+str(round(accuracy,2)))
    log_file.write("knn-unbalanced-simple3-scaled-train-test,"+smi_dock_file+","+str(elapsed_time)+","+str(num_samples_build)+","+str(dataset_size)+","+str(round(percentile,2))+","+str(threshold)+","+str(neighbors)+","+str(fp_depth)+","+str(fp_size)+","+str(round(recall_l,2))+","+str(round(recall_h,2))+","+str(round(perf,2))+","+str(round(accuracy,2))+"\n")
    log_file.flush()
    
    start_time = time.time()
    if DEBUG:
    	print("starting testing classifier...")
    dataset_size,recall_l, recall_h, perf, accuracy = test_model(knn_classifier, smi_dock_file, num_samples_test, batch_size, fp_depth, fp_size, percentile, neighbors, parallelism, threshold, log_file_name, DEBUG)
    if DEBUG:
    	print("finished testing classifier!")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("knn-unbalanced-simple3-scaled-test,"+smi_dock_file+","+str(elapsed_time)+","+str(num_samples_test)+","+str(dataset_size)+","+str(round(percentile,2))+","+str(threshold)+","+str(neighbors)+","+str(fp_depth)+","+str(fp_size)+","+str(round(recall_l,2))+","+str(round(recall_h,2))+","+str(round(perf,2))+","+str(round(accuracy,2)))
    log_file.write("knn-unbalanced-simple3-scaled-test,"+smi_dock_file+","+str(elapsed_time)+","+str(num_samples_test)+","+str(dataset_size)+","+str(round(percentile,2))+","+str(threshold)+","+str(neighbors)+","+str(fp_depth)+","+str(fp_size)+","+str(round(recall_l,2))+","+str(round(recall_h,2))+","+str(round(perf,2))+","+str(round(accuracy,2))+"\n")
    log_file.flush()    
    
    log_file.close()
    
    
if __name__ == '__main__':
    main(sys.argv[1:])


