import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
# import streamlit as st
# import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

from sklearn import metrics
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

data_frame = pd.read_csv('emissions_Canada.csv')
features = data_frame[['Make', 'Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
labels = data_frame['CO2 Emissions(g/km)']
numerical = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']
categories=['Make']

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(), categories)
    ]
)

model = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state = 42))
    ]
)


# print('Read dataset completed successfully.')
# print('Total number of rows: {0}\n\n'.format(len(data_frame.index)))
# # print(data_frame.head(10))
# # print(data_frame.corr(numeric_only = True))
# print(data_frame.describe(include='all'))

#No data missing in any columns so far
#Mean emmission is 255.584699 g/km
#max emmission 522, min 95 so there are outliers that are pretty big, using km not mpg because carbon emmission is per km
#print co2 per factor individually

# fig = px.scatter_matrix(training_df, dimensions=['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (mpg)', 'CO2 Emissions(g/km)', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption City (L/100 km)'])
# fig.show()



features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.2, random_state = 42)

# scaler_labels = StandardScaler()
# scaler_features = StandardScaler()

# features_train= scaler_features.fit_transform(features_train)
# features_val = scaler_features.transform(features_val)

# labels_train_scaled = scaler_labels.fit_transform(labels_train.values.reshape(-1, 1)).ravel()
# labels_val_scaled = scaler_labels.transform(labels_val.values.reshape(-1, 1)).ravel()



#use and train multiple models choose the best
# models = [RandomForestRegressor(random_state=42), SVR()]


model.fit(features_train, labels_train)
print(f'${model}: ')

training_predictions = model.predict(features_train)

# train_pred = scaler_labels.inverse_transform(training_predictions.reshape(-1, 1)).ravel()
val_predictions = model.predict(features_val)

# val_predictions = scaler_labels.inverse_transform(val_predictions.reshape(-1, 1)).ravel()
# joblib.dump(model, "model.pkl")
# joblib.dump(scaler_features, "scaler_features.pkl")
# joblib.dump(scaler_labels, "scaler_labels.pkl")

# st.title("CO2 Emissions Predictor")
# engine_size = st.number_input('Engine Size (L)', min_value=0.0, max_value=10.0, value=2.0)
# cylinders = st.number_input('Cylinders', min_value=2, max_value=16, value=4)
# fuel_consumption = st.number_input('Fuel Consumption (L/100 km)', min_value=1.0, max_value=30.0, value=8.0)

# user_input = np.array([[engine_size, cylinders, fuel_consumption]])
# user_input_scaled = scaler_features.transform(user_input)
# pred_scaled = rf_model.predict(user_input_scaled)
# pred_original = scaler_labels.inverse_transform(pred_scaled.reshape(-1,1))[0][0]

# st.success(f"Predicted CO2 Emissions: {pred_original:.2f} g/km")


# plt.scatter(labels_train, train_pred, color='blue', alpha=0.5, label='Training Predicted emmissions vs Actual emmissions')

# Test data
fig, axs = plt.subplots(1, 2, figsize = (10,4))
axs[0].scatter(labels_train, training_predictions, color='green')
axs[0].set_xlabel('Actual Training Emmissions')
axs[0].set_ylabel('Predicted TrainingEmmissions')
min_val = min(labels.min(), training_predictions.min(), val_predictions.min())
max_val = max(labels.max(), training_predictions.max(), val_predictions.max())
axs[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
axs[0].grid(True)

axs[1].scatter(labels_val, val_predictions, color='blue')
axs[1].set_xlabel('Actual Test Emmissions')
axs[1].set_ylabel('Predicted Test Emmissions')
min_val = min(labels.min(), training_predictions.min(), val_predictions.min())
max_val = max(labels.max(), training_predictions.max(), val_predictions.max())
axs[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
axs[1].grid(True)

plt.tight_layout()
plt.show()





# print('Training Error : ', mae(labels_train, training_predictions))
# print('MSE error: ', mse(labels_train, training_predictions))

# print('Validation Error : ', mae(labels_val, val_predictions))
# print('mse error: ', mse(labels_val, val_predictions))
# print('DONE \n')



