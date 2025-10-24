import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import os

# --- FUNGSI HELPER ---
# Fungsi metric: Root Mean Squared Error (RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Scorer untuk cross-validation
rmse_scorer = make_scorer(rmse, greater_is_better=False)

def run_model_training():
    print("\n--- Memulai Model Training ---")

    # --- 1. Tentukan Path ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    # --- 2. LOAD DATA PROCESSED ---
    try:
        train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train_processed.csv"))
        test_df = pd.read_csv(os.path.join(OUTPUT_DIR, "test_processed.csv"))
        
        # Ganti 'sampe_submission.csv' dengan nama file Anda yang benar jika typo
        sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv")) 
    except FileNotFoundError:
        print(f"Error: File data olahan tidak ditemukan. Pastikan file ada di folder {OUTPUT_DIR}")
        print("Jalankan 'feature_engineering.py' terlebih dahulu.")
        return

    print(f"Bentuk data train bersih: {train_df.shape}")
    print(f"Bentuk data test bersih: {test_df.shape}")

    # --- 3. SIAPKAN DATA UNTUK MODEL ---
    y_train = train_df['SalePrice_Log']
    X_train = train_df.drop('SalePrice_Log', axis=1)
    X_test = test_df.copy()

    # Penting: Pastikan kolom di X_train dan X_test sama persis
    X_train, X_test = X_train.align(X_test, join='inner', axis=1, fill_value=0)
    print(f"Jumlah fitur yang akan dipakai: {X_train.shape[1]}")

    # --- 4. CROSS-VALIDATION MODEL ---
    print("Menjalankan 5-Fold Cross-Validation...")
    model_lgb = lgb.LGBMRegressor(objective='regression',
                                  num_leaves=5,
                                  learning_rate=0.05,
                                  n_estimators=720,
                                  random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_lgb, X_train.values, y_train.values, 
                                scoring=rmse_scorer, cv=kf)

    print(f"Skor RMSE (Log Price) per fold: {-cv_scores}")
    print(f"Rata-rata RMSE (Log Price): {-cv_scores.mean():.5f}")

    # --- 5. LATIH MODEL FINAL DAN PREDIKSI ---
    print("Melatih model final pada semua data...")
    model_lgb.fit(X_train, y_train)

    print("Membuat prediksi pada data test...")
    predictions_log = model_lgb.predict(X_test)

    # Kembalikan ke nilai aslinya (Anti-Log)
    predictions = np.expm1(predictions_log)
    print("Prediksi selesai.")

    # --- 6. BUAT FILE SUBMISSION ---
    submission_df = pd.DataFrame()
    submission_df['Id'] = sample_sub['Id']
    submission_df['SalePrice'] = predictions

    # Simpan ke folder output
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nFile 'submission.csv' berhasil disimpan di folder '{OUTPUT_DIR}'.")
    print("Siap untuk di-submit ke Kaggle!")

    # Tampilkan 5 baris pertama dari hasil prediksi di terminal
    print("\nContoh hasil prediksi:")
    print(submission_df.head())
    print("--- Model Training Selesai ---")

if __name__ == "__main__":
    run_model_training()