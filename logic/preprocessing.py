import os

import numpy as np
from category_encoders import OneHotEncoder
from logic.data_loader import load_data, split_data

NUMERIC_COLS = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
ORIGIN_LOCATION = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2,'Other':3}

def preprocessing(train_df, test_df):
    # Check missing values
    # print(train_df.info())
    # print(test_df.info())

    # define numerical columns
    Numerical_col = NUMERIC_COLS

    # Fill Numerical missing values with mean
    train_mean = train_df.groupby("Species")[Numerical_col].mean()
    train_df[Numerical_col]=train_df[Numerical_col].fillna(train_df.groupby("Species")[Numerical_col].transform('mean'))
    for col in Numerical_col:
        test_df[col]=test_df[col].fillna(test_df["Species"].map(train_mean[col]))

    if train_df.isnull().sum().sum() == 0 and test_df.isnull().sum().sum() == 0:
        print("No missing values there")
    else:
        print("Warning:Missing values")

    # Normalize numerical columns
    mean, std = train_df[Numerical_col].mean(), train_df[Numerical_col].std()
    train_df[Numerical_col] = (train_df[Numerical_col] - mean) / std
    test_df[Numerical_col] = (test_df[Numerical_col] - mean) / std

    # Encode categorical columns
    OriginLocation_encode = dict(ORIGIN_LOCATION)
    train_df['OriginLocation'] = train_df['OriginLocation'].map(OriginLocation_encode).fillna(OriginLocation_encode['Other'])
    test_df['OriginLocation'] = test_df['OriginLocation'].map(OriginLocation_encode).fillna(OriginLocation_encode['Other'])

    X_train = train_df.drop('Species', axis=1)
    y_train = train_df['Species']
    X_test = test_df.drop('Species', axis=1)
    y_test = test_df['Species']

    # Encode Species (one-hot encoding)
    Target_encoder = OneHotEncoder()
    Target_encoder.fit(train_df[['Species']])
    y_train = Target_encoder.transform(y_train)
    y_test = Target_encoder.transform(y_test)

    stats = {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "columns": list(X_train.columns),  # feature order the network was trained with
    }

    return X_train, y_train, X_test, y_test, Target_encoder,OriginLocation_encode,stats


def preprocess_sample(sample, stats, origin_encode):
    row = {col: (float(sample[col]) - stats["mean"][col]) / stats["std"][col] for col in NUMERIC_COLS}
    row['OriginLocation'] = origin_encode.get(sample['OriginLocation'], origin_encode['Other'])
    return np.array([[row[c] for c in stats["columns"]]], dtype=np.float64)

if __name__ == "__main__":
    Base_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(Base_DIR,'..','data/penguins.csv')
    df = load_data(csv_path)
    train, test = split_data(df)
    X_train, y_train, X_test, y_test,Target_encoder,OriginLocation_encode,stats = preprocessing(train, test)
    print(X_train.head())
    print(y_train.head())
