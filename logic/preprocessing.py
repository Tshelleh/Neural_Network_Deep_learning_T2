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
    train_df[Numerical_col] = (train_df[Numerical_col] - train_df[Numerical_col].mean()) / train_df[Numerical_col].std()
    test_df[Numerical_col] = (test_df[Numerical_col] - test_df[Numerical_col].mean()) / test_df[Numerical_col].std()

    # Encode categorical columns
    OriginLocation_encode = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
    train_df['OriginLocation'] = train_df['OriginLocation'].map(OriginLocation_encode)
    test_df['OriginLocation'] = test_df['OriginLocation'].map(OriginLocation_encode)

    # Encode Species (one-hot encoding)
    encoder = OneHotEncoder()
    encoder.fit(train_df[['Species']])
    train_df = encoder.transform(train_df)
    test_df = encoder.transform(test_df)

    return train_df, test_df, encoder



if __name__ == "__main__":
    Base_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(Base_DIR,'..','data/penguins.csv')
    df = load_data(csv_path)
    train, test = split_data(df)
    train_df, test_df, encoder = preprocessing(train, test)
    print(train_df.head())
    print(test_df.head())
