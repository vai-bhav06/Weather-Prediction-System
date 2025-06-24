# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from joblib import dump
import pickle

# Load the data
data = pd.read_csv("Training_data.csv")

# Encode categorical labels
label_encoder = LabelEncoder()
data['condition_encoded'] = label_encoder.fit_transform(data['condition_encoded'])

# Feature selection
X = data.drop(['condition_encoded', 'year'], axis=1)
y = data['condition_encoded']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Normalize data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for LSTM
X_train_LSTM = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_LSTM = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model_lstm = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    LSTM(units=64, return_sequences=True),
    Dropout(0.2),
    LSTM(units=64),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dense(units=len(y.unique()), activation='softmax')  # Multi-class classification
])

# Compile model
model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)

# Train the model
history = model_lstm.fit(X_train_LSTM, y_train, batch_size=32, epochs=100, validation_data=(X_test_LSTM, y_test), callbacks=[early_stopping])

# Evaluate the model
_, accuracy = model_lstm.evaluate(X_test_LSTM, y_test)
print(f'LSTM Model Test Accuracy: {accuracy}')

# Train ML Models
models = {
    "XGB": XGBClassifier(),
    "SVC": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Model Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

# Hyperparameter Tuning for AdaBoost
param_grid = {'learning_rate': [1, 2, 3], 'n_estimators': [100, 500, 1000]}
grid_search = GridSearchCV(models["AdaBoost"], param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

# Ensemble Voting Classifier
ensemble = VotingClassifier(
    estimators=[('svc', models['SVC']), ('knn', models['KNN'])], voting='hard'
)
ensemble.fit(X_train, y_train)
ensemble_accuracy = metrics.accuracy_score(y_test, ensemble.predict(X_test))
print(f"Ensemble Model Accuracy: {ensemble_accuracy}")

# Save Models
model_list = {**models, "LSTM": model_lstm, "Ensemble": ensemble, "GridSearch": grid_search}
for name, model in model_list.items():
    dump(model, f"{name}_model.joblib")
    with open(f"{name}_model.pkl", 'wb') as file:
        pickle.dump(model, file)

print("✅ All models saved successfully!")
