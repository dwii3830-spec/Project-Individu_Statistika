[README.md](https://github.com/user-attachments/files/28800191/README.md)

# Mobile Phone Price Prediction
Proyek machine learning untuk memprediksi harga smartphone berdasarkan spesifikasi teknis perangkat menggunakan metode Multiple Linear Regression (Regresi Linear Berganda).

## Dataset
- Sumber: Kaggle — [Mobile Phone Specifications and Prices](https://www.kaggle.com/datasets/pratikgarai/mobile-phone-specifications-and-prices)
- Jumlah data: 1.200 baris
- Format: CSV
- Fitur: spesifikasi teknis berbagai brand (Samsung, Apple, Xiaomi, dll.)

## Model
Multiple Linear Regression (Regresi Linear Berganda)

Model ini memprediksi harga HP berdasarkan beberapa fitur numerik secara bersamaan. Fitur (parameter) yang digunakan sebagai variabel independen antara lain:
- RAM (GB)
- Storage internal (GB)
- Ukuran layar (inch)
- Kapasitas baterai (mAh)
- Resolusi kamera (MP)
- Dan fitur numerik lainnya dari dataset
Variabel target (Y): **Price** (harga smartphone)

## Tujuan
Membangun model regresi yang mampu memperkirakan harga HP berdasarkan spesifikasi teknisnya, serta mengevaluasi seberapa baik model menjelaskan variasi harga melalui nilai R².

## Evaluasi Model
Model dievaluasi menggunakan metrik **R² Score** (koefisien determinasi), yang menunjukkan seberapa besar variasi harga yang dapat dijelaskan oleh fitur-fitur dalam model.

## Hasil
Output notebook mencakup:
- Nilai R² Score pada data uji
- Grafik scatter plot perbandingan harga aktual vs harga prediksi
