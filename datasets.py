import pandas as pd
import numpy as np

def load_dataB(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        # Convert dates to datetime and calculate days since first measurement
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)  # Handle DD/MM/YYYY format
        df['Days'] = (df['Date'] - df['Date'].iloc[0]).dt.days
        
        # Ensure numeric values, sort by days, and take first 20 rows
        df = df.sort_values('Days').head(19)
        days = df['Days'].values.astype(float)
        height = df['Height'].values.astype(float)
        
        return np.array([days, height])
    
    except Exception as e:
        print(f"Error loading {csv_path}: {str(e)}")
        return np.array([[0], [0]])  # Return dummy data if error occurs