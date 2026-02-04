import pandas as pd
from datetime import datetime

def preprocess_data(path):

    df = pd.read_csv(path)

    # Drop car name
    df.drop('Car_Name', axis=1, inplace=True)

    # Feature Engineering → Car Age
    current_year = datetime.now().year
    df['Car_Age'] = current_year - df['Year']
    df.drop('Year', axis=1, inplace=True)

    # Convert categorical columns
    df = pd.get_dummies(df, drop_first=True)

    return df