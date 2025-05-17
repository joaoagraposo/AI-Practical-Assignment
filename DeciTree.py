import pandas as pd
import numpy as np
import random 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, attribute=None, value=None, results=None, branches=None, counter=None):
        self.attribute = attribute          
        self.value = value                  
        self.results = results              
        self.branches = branches if branches is not None else {}     
        self.counter = counter     

def entropy(data):
    total = len(data)
    if total == 0:
        return 0
    label_counts = data.iloc[:, -1].value_counts()
    entropy_val = 0
    for count in label_counts:
        p = count / total
        if p > 0:
            entropy_val -= p * np.log2(p)
    return entropy_val


def splitData(data, attribute, value):
    branches = {}
    branches["true_branch"] = data[data[attribute] == value]
    branches["false_branch"] = data[data[attribute] != value]
    return branches

def buildTree(data, max_depth, depth=0):
    counter = len(data)

    if counter == 0:
        return Node(results=None, counter=0)

    # Parar se todos os exemplos têm a mesma classe ou se atingimos profundidade máxima
    if len(data.iloc[:, -1].unique()) == 1 or depth == max_depth:
        return Node(results=data.iloc[:, -1].mode()[0], counter=counter)
    
    bestGain = 0        
    bestCriteria = None
    bestSets = None
    currentEntropy = entropy(data)
    
    for attribute in data.columns[:-1]:  
        attributeValues = data[attribute].unique()
        for value in attributeValues:
            branches = splitData(data, attribute, value)

            if len(branches['true_branch']) == 0 or len(branches['false_branch']) == 0:
                continue

            p = len(branches['true_branch']) / len(data)
            infoGain = currentEntropy - p * entropy(branches['true_branch']) - (1 - p) * entropy(branches['false_branch'])

            if infoGain > bestGain:
                bestGain = infoGain
                bestCriteria = (attribute, value)
                bestSets = (branches['true_branch'], branches['false_branch'])

    if bestGain > 0 and bestSets is not None:
        true_branch = buildTree(bestSets[0], max_depth, depth + 1)
        false_branch = buildTree(bestSets[1], max_depth, depth + 1)
        return Node(attribute=bestCriteria[0], value=bestCriteria[1],
                    branches={"true_branch": true_branch, "false_branch": false_branch})
    else:
        return Node(results=data.iloc[:, -1].mode()[0], counter=counter)


def printTree(node, spacing=""):
    if node.results is not None:
        print(f"{spacing}{node.results} ({node.counter})")
    else:
        print(f"{spacing}<{node.attribute}> <{node.value}>")
        print(spacing + " TRUE:")
        printTree(node.branches["true_branch"], spacing + "  ")
        print(spacing + " FALSE:")
        printTree(node.branches["false_branch"], spacing + "  ")


def classifyExample(example, tree):
    if tree.results is not None:
        return tree.results
    else:
        branch = None
        if example[tree.attribute] == tree.value:
            branch = tree.branches["true_branch"]
        else:
            branch = tree.branches["false_branch"]
        return classifyExample(example, branch)

def quartis(df):
    for col in df.columns[1:]:
        if (df[col].dtype  == np.int32 or df[col].dtype == 'int64' or df[col].dtype == 'float64'): 
            try:
                df[col] = pd.qcut(df[col], q=4, duplicates='drop')
            except ValueError:
                print(f"Coluna {col} ignorada por não ter valores suficientes distintos.")
    return df



def main():
    dataset = input("Escolha o dataset: ")
    print()
    
    df=pd.read_csv(dataset)
    df = quartis(df)
    df.to_csv(dataset, index=False)
    max_depth = int(np.log2(len(df)) + 1)
    X = df.iloc[:, :-1]  # tudo menos a ultima
    y = df.iloc[:, -1]   # ultima coluna
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    training_data = pd.concat([X_train, y_train], axis=1)
    decision_tree = buildTree(training_data, max_depth)
    printTree(decision_tree)
    
    y_pred = [classifyExample(row, decision_tree) for _, row in X_test.iterrows()]
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: " , accuracy)


main()