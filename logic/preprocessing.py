import os

from category_encoders import OneHotEncoder

from data_loader import load_data, split_data

def preprocessing(train_df, test_df):
    # Check missing values
    # print(train_df.info())
    # print(test_df.info())

    # define numerical columns
    Numerical_col = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']

    # Fill Numerical missing values with mean
    train_df[Numerical_col]=train_df[Numerical_col].fillna(train_df.groupby("Species")[Numerical_col].transform('mean'))
    test_df[Numerical_col]=test_df[Numerical_col].fillna(test_df.groupby("Species")[Numerical_col].transform('mean'))

    if train_df.isnull().sum().sum() == 0 and test_df.isnull().sum().sum() == 0:
        print("No missing values there")
    else:
        print("Warning:Missing values")

    # Normalize numerical columns
    mean, std = train_df[Numerical_col].mean(), train_df[Numerical_col].std()
    train_df[Numerical_col] = (train_df[Numerical_col] - mean) / std
    test_df[Numerical_col] = (test_df[Numerical_col] - mean) / std

    # Encode categorical columns
    OriginLocation_encode = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2,"Other":3}
    train_df['OriginLocation'] = train_df['OriginLocation'].map(OriginLocation_encode)
    test_df['OriginLocation'] = test_df['OriginLocation'].map(OriginLocation_encode)

    X_train = train_df.drop('Species', axis=1)
    y_train = train_df['Species']
    X_test = test_df.drop('Species', axis=1)
    y_test = test_df['Species']

    # Encode Species (one-hot encoding)
    Target_encoder = OneHotEncoder()
    Target_encoder.fit(train_df[['Species']])
    y_train = Target_encoder.transform(y_train)
    y_test = Target_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test, Target_encoder,OriginLocation_encode



if __name__ == "__main__":
    Base_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(Base_DIR,'..','data/penguins.csv')
    df = load_data(csv_path)
    train, test = split_data(df)
    X_train, y_train, X_test, y_test,Target_encoder,OriginLocation_encode = preprocessing(train, test)
    print(X_train.head())
    print(y_train.head())
