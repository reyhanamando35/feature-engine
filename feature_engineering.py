import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

def run_feature_engineering():
    print("--- Memulai Feature Engineering ---")

    # --- 1. Tentukan Path ---
    # Ini akan membuat path-nya fleksibel
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    # Pastikan folder output ada
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 2. LOAD DATA ---
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    except FileNotFoundError:
        print(f"Error: File data tidak ditemukan. Pastikan file ada di folder {DATA_DIR}")
        return

    print(f"Bentuk data train asli: {train_df.shape}")
    print(f"Bentuk data test asli: {test_df.shape}")

    # --- 3. PISAHKAN TARGET DAN SIMPAN ID ---
    # Log transform SalePrice (standar untuk dataset ini)
    target = np.log1p(train_df['SalePrice'])
    train_id = train_df['Id']
    test_id = test_df['Id']

    train_df = train_df.drop(['Id', 'SalePrice'], axis=1)
    test_df = test_df.drop('Id', axis=1)

    # Gabungkan untuk pemrosesan
    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    print(f"Bentuk data gabungan: {all_data.shape}")

    # --- 4. HANDLE MISSING VALUES (NAN) ---
    print("Memproses missing values (NaN)...")

    # 4a. Kolom di mana 'NA' berarti "None" (Tidak Punya)
    for col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'MasVnrType']:
        all_data[col] = all_data[col].fillna('None')

    # 4b. Kolom di mana 'NA' berarti 0 (Numerik)
    for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 
                'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
                'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']:
        all_data[col] = all_data[col].fillna(0)

    # 4c. Kolom di mana 'NA' diisi dengan Modus (Nilai Paling Sering Muncul)
    for col in ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 
                'Exterior2nd', 'SaleType', 'Utilities']:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # 4d. Kasus Khusus: LotFrontage
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))

    print("Missing values selesai diproses.")

    # --- 5. FEATURE ENGINEERING (MEMBUAT FITUR BARU) ---
    print("Membuat fitur baru...")

    # 5a. Koreksi Tipe Data (Angka yang Sebenarnya Kategori)
    all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    # 5b. Membuat Fitur Total Luas
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    
    # 5c. Membuat Fitur Total Kamar Mandi
    all_data['TotalBath'] = (all_data['FullBath'] + 0.5 * all_data['HalfBath'] +
                             all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath'])

    # --- 6. ENCODING FITUR KATEGORIAL ---
    print(f"Jumlah kolom sebelum OHE: {all_data.shape[1]}")
    all_data_processed = pd.get_dummies(all_data)
    print(f"Jumlah kolom setelah OHE: {all_data_processed.shape[1]}")

    # --- 7. PISAHKAN DAN SIMPAN DATA ---
    X_train_processed = all_data_processed[:len(train_id)].copy()
    X_test_processed = all_data_processed[len(train_id):].copy()

    # Simpan 'target' (SalePrice yg sudah di-log) ke file train
    X_train_processed['SalePrice_Log'] = target

    print(f"Bentuk data train bersih: {X_train_processed.shape}")
    print(f"Bentuk data test bersih: {X_test_processed.shape}")

    # Simpan ke folder output
    train_path = os.path.join(OUTPUT_DIR, "train_processed.csv")
    test_path = os.path.join(OUTPUT_DIR, "test_processed.csv")
    
    X_train_processed.to_csv(train_path, index=False)
    X_test_processed.to_csv(test_path, index=False)
    
    print(f"\nFile 'train_processed.csv' dan 'test_processed.csv' berhasil disimpan di folder '{OUTPUT_DIR}'.")
    print("--- Feature Engineering Selesai ---")

if __name__ == "__main__":
    run_feature_engineering()