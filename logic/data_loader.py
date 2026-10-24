import random
import pandas as pd
import numpy as np
import os

def load_data(filepath:str='penguins.csv'):
    # Load data
    df = pd.read_csv(filepath)
    return df

def split_data(df):
    # Convert to numpy array if it's a DataFrame
    if isinstance(df, pd.DataFrame):
        data = df.values
    else:
        data = df
    
    # Split by class (assuming classes are in order: first 50, next 50, last 50)
    class1 = data[:50, :]
    class2 = data[50:100, :]
    class3 = data[100:, :]
    
    # Shuffle each class
    random.shuffle(class1)
    random.shuffle(class2)
    random.shuffle(class3)
    
    # Take first 30 for train, next 20 for test from each class
    train1 = class1[:30, :]
    test1 = class1[30:50, :]
    
    train2 = class2[:30, :]
    test2 = class2[30:50, :]
    
    train3 = class3[:30, :]
    test3 = class3[30:50, :]
    
    # Concatenate all train and test samples
    train_d = np.concatenate([train1, train2, train3], axis=0)
    test_d = np.concatenate([test1, test2, test3], axis=0)

    # Shuffle the training data
    np.random.shuffle(train_d)
    
    return train_d, test_d

if __name__ == "__main__":
    Base_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(Base_DIR,'..','data/penguins.csv')
    df = load_data(csv_path)
    train, test = split_data(df)
    print(train.shape, test.shape)
    print(train)
