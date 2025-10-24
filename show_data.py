import pandas as pd
import os

# Atur agar Pandas menampilkan lebih banyak kolom
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

try:
    # Tentukan path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    FILE_PATH = os.path.join(OUTPUT_DIR, "train_processed.csv")

    # Baca file hasil
    df = pd.read_csv(FILE_PATH)

    # Tampilkan 5 baris pertama
    print("--- Menampilkan 5 baris pertama dari train_processed.csv ---")
    print(df.head())

    print(f"\nBentuk data: {df.shape}")

except FileNotFoundError:
    print(f"Error: File tidak ditemukan di {FILE_PATH}")
    print("Jalankan 'feature_engineering.py' terlebih dahulu.")