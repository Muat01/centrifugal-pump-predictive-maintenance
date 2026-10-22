import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_predictive_model(data_path):
    print("1. Loading dataset...")
    df = pd.read_csv(data_path)
    
    # 2. Feature Engineering: Adding Rolling Statistics for Industrial Signals
    print("2. Extracting industrial features (Rolling Mean & Std)...")
    df['pressure_rolling_mean'] = df['pressure_psi'].rolling(window=5).mean()
    df['vibration_rolling_std'] = df['vibration_mms'].rolling(window=5).std()
    
    # Drop rows with NaN values resulting from rolling window
    df = df.dropna().reset_index(drop=True)
    
    # 3. Defining Features (X) and Target (y)
    # Target: 0 for Normal, 1 for Leakage
    df['target'] = df['status'].apply(lambda x: 1 if x == 'Leakage' else 0)
    
    features = ['pressure_psi', 'flow_gpm', 'vibration_mms', 'pressure_rolling_mean', 'vibration_rolling_std']
    X = df[features]
    y = df['target']
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Training Machine Learning Model (Random Forest)
    print("3. Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_predictive_model("../data/pump_sensor_data.csv")
