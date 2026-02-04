import joblib
import pandas as pd

model = joblib.load("models/car_price_model.pkl")

def predict_price(data):

    df = pd.DataFrame([data])
    prediction = model.predict(df)

    return prediction[0]