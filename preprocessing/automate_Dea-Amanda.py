import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    print("Memuat dataset raw...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Memulai proses pembersihan data...")
    X = df.drop(columns=['default'])
    y = df['default']
        
    # Mengisi nilai kosong dengan nilai tengah (median)
    numerical_cols = X.columns
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Menyamakan skala angka (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Menggabungkan data kembali
    df_clean = pd.DataFrame(X_scaled, columns=numerical_cols)
    df_clean['default'] = y.values
    
    print("Preprocessing selesai!")
    return df_clean

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset bersih disimpan di: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "credit_scoring_raw/data.csv"
    OUTPUT_FILE = "credit_scoring_preprocessing/data_clean.csv"
    
    try:
        raw_data = load_data(INPUT_FILE)
        clean_data = preprocess_data(raw_data)
        save_data(clean_data, OUTPUT_FILE)
    except Exception as e:
        print(f"Error: {str(e)}")
