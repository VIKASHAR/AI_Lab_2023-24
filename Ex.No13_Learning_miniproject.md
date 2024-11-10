# Ex.No: 13 Learning – Use Supervised Learning  
### DATE:                                                                            
### REGISTER NUMBER : 212222040179
### AIM: 
To develop a predictive model using machine learning techniques to estimate the price of a house based on relevant features

###  Algorithm:

Step 1: Start the program.

Step 2: Load the dataset for house price prediction.

Step 3: Select relevant features (bedrooms, bathrooms, sqft_living, floors) and the target variable (price).

Step 4: Split the data into training and testing sets, with 80% for training and 20% for testing.

Step 5: Scale the features using StandardScaler to normalize the data.

Step 6: Initialize the RandomForestRegressor model with 100 estimators.

Step 7: Train the model using the scaled training data.

Step 8: Test the model by making predictions on the scaled test data.

Step 9: Save the trained model and the scaler using the pickle library.

Step 10: End the program.








### Program:
```
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

# Load your dataset
df = pd.read_csv('dataset.csv')  # Replace with the actual path to your dataset

# Select the relevant features and target variable
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'floors']]
y = df['price']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions (just for testing purpose)
y_pred = rf_model.predict(X_test_scaled)

# Save the trained model
with open('prediction_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and Scaler saved successfully!")

```
### Output:


![image](https://github.com/user-attachments/assets/85f9d38f-ff4e-4a91-82f0-39f4f211eb36)






### Result:
Thus the system was trained successfully and the prediction was carried out.
