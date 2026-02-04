from src.predict import predict_price

sample_data = {
    'Present_Price': 5.59,
    'Driven_kms': 27000,
    'Owner': 0,
    'Car_Age': 10,
    'Fuel_Type_Diesel': 0,
    'Fuel_Type_Petrol': 1,
    'Selling_type_Individual': 0,
    'Transmission_Manual': 1
}

price = predict_price(sample_data)

print("Predicted Price:", price)