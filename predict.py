import joblib
import pandas as pd

# This file is simply fo testing the model 
model = joblib.load('house_price_model.pkl')

#Manually Enter Features to predict prices
new_data = pd.DataFrame({
    'bedrooms': [3, 4],
    'bathrooms': [2, 3],
    'sqft_living': [1800, 2500],
    'sqft_lot': [5000, 6000],
    'floors': [1, 2],
    'waterfront': [0, 1],
    'view': [0, 1],
    'condition': [3, 4],
    'sqft_above': [1800, 2200],
    'sqft_basement': [0, 300],
    'yr_built': [2005, 2010],
    'yr_renovated': [0, 2020]
})

# Save to CSV
new_data.to_csv('new_houses.csv', index=False)
print("new_houses.csv saved successfully!")

new_house = pd.read_csv('new_houses.csv')

predicted_price = model.predict(new_house)
print(f"Predicted House Price: ${predicted_price[0]:,.2f}")

predictions = model.predict(new_data)
new_data['Predicted Price'] = predictions
print(new_data)
new_data.to_csv('predicted_prices.csv', index=False)
print("Predictions saved to predicted_prices.csv")