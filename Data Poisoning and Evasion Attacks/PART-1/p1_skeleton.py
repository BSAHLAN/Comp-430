import sys
import random

import numpy as np
import pandas as pd
import copy


from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest



###############################################################################
############################# Label Flipping ##################################
###############################################################################
def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p):
    """
    Performs a label flipping attack on the training data.

    Parameters:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    model_type: Type of model ('DT', 'LR', 'SVC')
    p: Proportion of labels to flip

    Returns:
    Accuracy of the model trained on the modified dataset
    """
    # TODO: You need to implement this function!
    # Implementation of label flipping attack
    
    total_accuracy = 0
    for i in range(100):
        new_train_labels = np.array(copy.deepcopy(y_train))
        label_num = len(new_train_labels)
        percent = label_num * p
    
        rand_idx = np.random.choice(len(new_train_labels), size=int(percent), replace=False)
        new_train_labels[rand_idx] = 1 - new_train_labels[rand_idx]
        
        if(model_type == "DT"):
            DT_model = DecisionTreeClassifier(max_depth=5, random_state=0)
            DT_model.fit(X_train, new_train_labels)
            DT_predict = DT_model.predict(X_test)
            total_accuracy += accuracy_score(y_test, DT_predict)
            
        elif(model_type == "LR"):    
            LR_model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
            LR_model.fit(X_train, new_train_labels)
            LR_predict = LR_model.predict(X_test)
            total_accuracy += accuracy_score(y_test, LR_predict)
        
        elif(model_type == "SVC"):
            SVC_model = SVC(C=0.5, kernel='poly', random_state=0)
            SVC_model.fit(X_train, new_train_labels)
            SVC_predict = SVC_model.predict(X_test)
            total_accuracy += accuracy_score(y_test, SVC_predict)
    
    
    return (total_accuracy/100)


###############################################################################
########################### Label Flipping Defense ############################
###############################################################################

def label_flipping_defense(X_train, y_train, p):
    """
    Performs a label flipping attack, applies outlier detection, and evaluates the effectiveness of outlier detection.

    Parameters:
    X_train: Training features
    y_train: Training labels
    p: Proportion of labels to flip

    Prints:
    A message indicating how many of the flipped data points were detected as outliers
    """
    # TODO: You need to implement this function!
    # Perform the attack, then the defense, then print the outcome
    
    new_train_labels = np.array(copy.deepcopy(y_train))
    label_num = len(new_train_labels)
    percent = label_num * p
    
    rand_idx = np.random.choice(len(new_train_labels), size=int(percent), replace=False)
    new_train_labels[rand_idx] = 1 - new_train_labels[rand_idx]

    new_labels_xtrain_combined = np.column_stack((X_train, new_train_labels))
    
    clf = LocalOutlierFactor(n_neighbors=5, contamination=p)
    y_pred = clf.fit_predict(new_labels_xtrain_combined)
    outlier_indices = np.where(y_pred == -1)[0]
   
    correct_guesses = 0
    for i in outlier_indices:
        if(i in rand_idx):
            correct_guesses += 1

    print(f"Out of {percent} flipped data points, {correct_guesses} were correctly identified.")
    


###############################################################################
############################# Evasion Attack ##################################
###############################################################################
def evade_model(trained_model, actual_example):
    """
    Attempts to create an adversarial example that evades detection.

    Parameters:
    trained_model: The machine learning model to evade
    actual_example: The original example to be modified

    Returns:
    modified_example: An example crafted to evade the trained model
    """
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    
    pred_class = actual_class
    while pred_class == actual_class:
    #do something to modify the instance
       temp_idx = 0
       if(actual_class == 1):
           #negative_num = sum(1 for num in modified_example if num < 0)
           negative_indices = np.where(modified_example < 0)
           #modified_example[np.argmin(modified_example)] *= -1
           if(len(negative_indices[0]) == 0):
               modified_example[np.argmin(modified_example)] += 0.1
           else:
              modified_example[negative_indices[0][temp_idx]] *= -1
           
           temp_idx += 1
           
       elif(actual_class == 0):
           positive_indices = np.where(modified_example > 0)
           modified_example[positive_indices[0][temp_idx]] *= -1
           temp_idx += 1 
        
       pred_class = trained_model.predict([modified_example])[0]
        
    return modified_example


def calc_perturbation(actual_example, adversarial_example):
    """
    Calculates the perturbation added to the original example.

    Parameters:
    actual_example: The original example
    adversarial_example: The modified (adversarial) example

    Returns:
    The average perturbation across all features
    """
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
########################## Transferability ####################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    """
    Evaluates the transferability of adversarial examples.

    Parameters:
    DTmodel: Decision Tree model
    LRmodel: Logistic Regression model
    SVCmodel: Support Vector Classifier model
    actual_examples: Examples to test for transferability

    Returns:
    Transferability metrics or outcomes
    """
    # TODO: You need to implement this function!
    # Implementation of transferability evaluation
    DT_to_LR = 0
    DT_to_SVC = 0
    LR_to_DT = 0
    LR_to_SVC = 0
    SVC_to_LR = 0
    SVC_to_DT = 0
    for actual_example in actual_examples:
        
        #calculating actual predictions
        DT_acc = DTmodel.predict([actual_example])[0]
        LR_acc = LRmodel.predict([actual_example])[0]
        SVC_acc = SVCmodel.predict([actual_example])[0]
        
        #Finding an adversarial example for each model
        DT_adv_example = evade_model(DTmodel, actual_example)
        LR_adv_example = evade_model(LRmodel, actual_example)
        SVC_adv_example = evade_model(SVCmodel, actual_example)
        
        #class predictions of the evaded examples
        DT_evaded_result = DT_acc ^ 1
        LR_evaded_result = LR_acc ^ 1
        SVC_evaded_result = SVC_acc ^ 1
        
        #DT to LR
        DT_to_LR_pred = LRmodel.predict([DT_adv_example])[0]
        if(DT_evaded_result == DT_to_LR_pred):
            DT_to_LR += 1
        
        #DT to SVC
        DT_to_SVC_pred = SVCmodel.predict([DT_adv_example])[0]
        if(DT_evaded_result == DT_to_SVC_pred):
            DT_to_SVC += 1
        
        #LR to DT
        LR_to_DT_pred = DTmodel.predict([LR_adv_example])[0]
        if(LR_evaded_result == LR_to_DT_pred):
            LR_to_DT += 1
        
        #LR to SVC
        LR_to_SVC_pred = SVCmodel.predict([LR_adv_example])[0]
        if(LR_evaded_result == LR_to_SVC_pred):
            LR_to_SVC += 1
        
        #SVC to LR
        SVC_to_LR_pred = LRmodel.predict([SVC_adv_example])[0]
        if(SVC_evaded_result == SVC_to_LR_pred):
            SVC_to_LR += 1
        
        #SVC to DT
        SVC_to_DT_pred = DTmodel.predict([DT_adv_example])[0]
        if(SVC_evaded_result == SVC_to_DT_pred):
            SVC_to_DT += 1
        
        

    print("Out of 40 adversarial examples crafted to evade DT :")
    print(f"-> {DT_to_LR} of them transfer to LR.")
    print(f"-> {DT_to_SVC} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade LR :")
    print(f"-> {LR_to_DT} of them transfer to DT.")
    print(f"-> {LR_to_SVC} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade SVC :")
    print(f"-> {SVC_to_DT} of them transfer to DT.")
    print(f"-> {SVC_to_LR} of them transfer to LR.")



###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ##
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##
def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    
    # Raw model accuracies:
    print("#" * 50)
    print("Raw model accuracies:")

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    print("#"*50)
    print("Label flipping attack executions:")
    model_types = ["DT", "LR", "SVC"]
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for p in p_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p)
            print("Accuracy of poisoned", model_type, str(p), ":", acc)

    # Label flipping defense executions:
    print("#" * 50)
    print("Label flipping defense executions:")
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for p in p_vals:
        print("Results with p=", str(p), ":")
        label_flipping_defense(X_train, y_train, p)

    # Evasion attack executions:
    print("#"*50)
    print("Evasion attack executions:")
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"]
    num_examples = 40
    for a, trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a], ":", total_perturb / num_examples)

    # Transferability of evasion attacks:
    print("#"*50)
    print("Transferability of evasion attacks:")
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])
    


if __name__ == "__main__":
    main()


