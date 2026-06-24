# Prediksi tingkat pembelian 

1. Deskripsi Proyek

Customer Purchase Prediction System adalah aplikasi Machine Learning yang digunakan untuk memprediksi apakah seorang pelanggan akan melakukan pembelian berdasarkan perilaku mereka saat mengunjungi website e-commerce.

Proyek ini dibangun menggunakan algoritma Random Forest Classifier dan diimplementasikan ke dalam aplikasi web interaktif menggunakan Streamlit dan hasilnya juga masih terholong sederhana sehingga pengguna  hanya dapat melakukan prediksi secara langsung melalui browser.


2. Tujuan Proyek

Membantu bisnis e-commerce untuk:

- Mengidentifikasi pelanggan yang berpotensi melakukan pembelian.
- Mendukung strategi pemasaran yang lebih efektif.
- Mengoptimalkan proses konversi pelanggan.
- Memberikan insight berdasarkan perilaku pengguna website.



3. Dataset

Dataset yang digunakan:

"Online Shoppers Purchasing Intention Dataset"

Dataset berisi informasi perilaku pengunjung website e-commerce seperti:

- Administrative Pages
- Administrative Duration
- Informational Pages
- Informational Duration
- Product Related Pages
- Product Related Duration
- Bounce Rates
- Exit Rates
- Page Values
- Special Day
- Month
- Visitor Type
- Weekend
- Revenue (Target)

Jumlah data:

- 12.330 records
- 18 columns

Target: 

- 1 = Customer melakukan pembelian
- 0 = Customer tidak melakukan pembelian


## Algoritma yang Digunakan

 1. Logistic Regression , Digunakan sebagai baseline model untuk perbandingan performa.

 2. Random Forest Classifier, 
    Digunakan sebagai model utama karena menghasilkan performa yang lebih baik dibanding Logistic Regression.

    Keunggulan:

    - Mampu menangani hubungan non-linear.
    - Lebih stabil terhadap overfitting.
    - Memiliki fitur Feature Importance.



##  Tahapan Machine Learning
  
  1. Data Preprocessing

      Melakukan:

      - Pemeriksaan missing values
      - Encoding fitur kategorikal:
      - Month
      - VisitorType
      - Konversi data boolean:
      - Weekend
      - Revenue
      - Standardisasi fitur menggunakan StandardScaler

  2. Data Splitting

      Dataset dibagi menjadi:

      - Training Data : 80%
      - Testing Data : 20%


  3. Model Training

      Model yang dilatih:

      - Logistic Regression
      - Random Forest Classifier


  4. Hyperparameter Tuning

    Menggunakan:
    
    python
    GridSearchCV

    Parameter yang diuji:
    
    python
    {
      "n_estimators": [50, 100],
      "max_depth": [5, 10],
      "min_samples_split": [2, 5]
    }


  5. Model Evaluation

    Metrik yang digunakan:

    - Accuracy
    - Precision
    - Recall
    - F1 Score



## Hasil Evaluasi Model

1. Random Forest 

  Accuracy  :  87.35% 
  Precision : 57.03% 
  Recall    : 74.35% 
  F1 Score  :  64.55% 

Best Parameters:

python
{
    'max_depth': 10,
    'min_samples_split': 2,
    'n_estimators': 50
}


## Implementasi Aplikasi

Framework:  Streamlit

Fitur aplikasi:

- Input data pelanggan
- Prediksi pembelian
- Probabilitas pembelian
- Visualisasi hasil menggunakan chart
- Dashboard interaktif


##  Cara Menjalankan Project

1. Clone Repository

2. Masuk ke Folder Project

  cd prediksi_beli_atau_tidak

3. Buat Virtual Environment

  python -m venv venv

4. Aktifkan Virtual Environment

  venv\Scripts\activate

5. Install Dependencies

  pip install -r requirements.txt

6. Jalankan Aplikasi

  streamlit run app.py


## Teknologi yang Digunakan

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Streamlit
- Plotly
