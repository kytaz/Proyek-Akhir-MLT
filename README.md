# Laporan Proyek Machine Learning - Tazky Khumaira Tsany

## Project Overview



### Refrensi


## Business Understanding
### A. Problem Statements
- **Pernyataan Masalah 1** : Pengguna kesulitan menemukan rekomendasi ramen yang sesuai dengan preferensi dan selera mereka dari ribuan produk yang tersedia.

- **Pernyataan Masalah 2**: Tidak adanya sistem rekomendasi yang dapat memanfaatkan data rating dan karakteristik produk untuk memberikan rekomendasi ramen yang relevan dan personal.


### B. Goals
- **Goal 1**: Membangun sistem rekomendasi yang mampu memberikan rekomendasi ramen serupa berdasarkan karakteristik produk (content-based filtering).

- **Goal 2 **: Mengembangkan sistem rekomendasi berbasis preferensi pengguna (collaborative filtering) dengan memanfaatkan data rating untuk meningkatkan relevansi rekomendasi.

### C. Solution Approach

Untuk mencapai tujuan yang telah ditetapkan, proyek ini menggunakan dua pendekatan utama dalam pengembangan sistem rekomendasi rating ramen, yaitu:

#### 1) Content-Based Filtering (CBF)
#### 2) Collaborative Filtering (CF)


## Data Understanding
Pada tahap ini awal dalam proses analisis data yang bertujuan untuk mengenal dan memahami struktur, tipe, serta kualitas data ramen yang digunakan dalam penelitian.

Dataset yang digunakan adalah dataset rating ramen yang berisi 2580 data produk dari berbagai brand dan negara, dengan fitur seperti Brand, Variety (nama ramen), Style, Country, dan Stars (rating). Data ini bersumber dari Kaggle [ https://www.kaggle.com/datasets/residentmario/ramen-ratings]

 Variable pada dataset `Ramen-rating.csv`
 
| Kolom    | Deskripsi                                                                                    | Tipe Data                |
| -------- | -------------------------------------------------------------------------------------------- | ------------------------ |
| Review # | Nomor urut review, unik untuk setiap entri                                                   | Integer                  |
| Brand    | Merek ramen, berisi nama produsen atau merek ramen                                           | Objek (String)           |
| Variety  | Nama atau deskripsi ramen, berisi nama jenis ramen yang diulas                               | Objek (String)           |
| Style    | Bentuk kemasan ramen, seperti cup, pack, bowl, dll.                                          | Objek (String)           |
| Country  | Negara asal ramen, lokasi asal produk atau review                                            | Objek (String)           |
| Stars    | Rating atau bintang yang diberikan pada ramen, awalnya objek, kemudian dikonversi ke numerik | Numerik (Float)          |
| Top Ten  | Keterangan apakah ramen termasuk dalam daftar Top Ten di tahun tertentu, banyak nilai kosong | Objek (String), Nullable |

Dataset yang digunakan memiliki 2580 baris data atau entri.

## Exploratory Data Analysis (EDA)

Pada tahap EDA, dilakukan analisis awal untuk memahami distribusi dan karakteristik data ramen. 

### Analisis Distribusi Rating (Stars):
> Menghitung dan memvisualisasikan jumlah data ramen berdasarkan nilai rating untuk melihat sebaran dan frekuensi masing-masing kategori rating.

### Pengelompokan Rating Rata-rata per Brand:
> Menghitung rata-rata rating untuk setiap brand ramen dan mengurutkannya untuk mengetahui brand dengan performa rating terbaik.

### Visualisasi Brand dengan Rating Tertinggi:
> Membuat grafik batang (bar chart) yang menampilkan 10 brand dengan rata-rata rating tertinggi sebagai insight brand unggulan.

### Pengecekan Missing Values dan Konsistensi Data:
> Memastikan bahwa data yang digunakan untuk analisis sudah bersih dari nilai kosong yang dapat mengganggu proses pemodelan.

### Pengenalan Karakteristik Dataset
> Mengetahui variabilitas fitur seperti Style, Country, dan jumlah unik brand dan ramen untuk memahami kompleksitas data.


## Data Preparation

Pada tahap Data Preparation, dilakukan serangkaian proses untuk membersihkan dan menyiapkan data agar siap digunakan dalam pemodelan sistem rekomendasi. Proses-proses utama meliputi:

### Konversi Tipe Data:
> Kolom Stars awalnya bertipe objek diubah menjadi tipe numerik (float) agar dapat digunakan dalam perhitungan dan model prediktif.

### Penanganan Missing Values:
> Nilai kosong pada kolom Style diisi dengan label 'Unknown' untuk menjaga konsistensi data tanpa menghapus baris yang valid. Kolom Top Ten dihapus karena mengandung banyak nilai kosong dan tidak relevan untuk analisis lebih lanjut. Selain itu, nilai kosong pada kolom Country diisi dengan label 'Unknown'.

### Pengelompokan Kelas Minoritas:
> Untuk mendukung pembagian data menggunakan stratified sampling, negara-negara dengan jumlah data sangat sedikit digabungkan ke dalam kategori 'Others' untuk menghindari masalah saat pembagian data.

### Pembagian Data:
> Dataset dibagi menjadi data training dan testing dengan proporsi 80:20 menggunakan stratified split berdasarkan kolom modifikasi Country agar distribusi kelas tetap seimbang.


## Modeling

Tahap modeling terdiri dari pengembangan dua pendekatan sistem rekomendasi utama untuk menyelesaikan permasalahan ini.

### 1. Content-Based Filtering

Model ini menggunakan fitur teks dari kolom Variety yang diolah dengan TF-IDF untuk merepresentasikan karakteristik tiap jenis ramen dalam bentuk vektor numerik. Kemudian, cosine similarity dihitung untuk mengukur kemiripan antar ramen berdasarkan deskripsi tersebut. Fungsi rekomendasi dibuat untuk mengembalikan 10 ramen paling mirip dengan input ramen berdasarkan kemiripan konten, sehingga pengguna dapat menemukan produk serupa yang relevan.

#### TF-IDF Vectorizer

TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengubah deskripsi ramen (Variety) menjadi representasi numerik berbobot berdasarkan frekuensi kata penting.

```
from sklearn.feature_extraction.text import TfidfVectorizer

# Membuat objek TF-IDF Vectorizer dengan stop words bahasa Inggris
tfidf = TfidfVectorizer(stop_words='english')

# Terapkan TF-IDF pada kolom 'Variety'
tfidf_matrix = tfidf.fit_transform(df['Variety'])

# Cek ukuran matriks TF-IDF (baris = jumlah ramen, kolom = fitur kata unik)
print("Shape TF-IDF matrix:", tfidf_matrix.shape)
```

#### Cosine Similarity

Cosine similarity menghitung tingkat kemiripan antar ramen berdasarkan representasi TF-IDF.

```
from sklearn.metrics.pairwise import cosine_similarity

# Hitung cosine similarity antar baris TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Cek ukuran matrix similarity
print("Shape cosine similarity matrix:", cosine_sim.shape)
```

#### indeks nama ramen

Indeks nama ramen itu peta penghubung dari nama ramen ke posisi baris di dataset yang memudahkan algoritma rekomendasi untuk langsung menemukan data ramen tertentu dengan cepat dan akurat.

```
print(df['Variety'].unique())
# Membuat indeks dari kolom 'Variety' untuk pencarian
indices = pd.Series(df.index, index=df['Variety']).drop_duplicates()

# Contoh cek indeks ramen 'Shin Black'
print("Indeks ramen 'Singapore Curry':", indices['Singapore Curry'])
```
```
def recommend_ramen_content_based(title, cosine_sim=cosine_sim, df=df):
    indices = pd.Series(df.index, index=df['Variety']).drop_duplicates()

    if title not in indices:
        return f"Ramen dengan nama '{title}' tidak ditemukan dalam data."

    idx = indices[title]

    # Ambil baris similarity untuk ramen ini, pastikan 1D array
    sim_scores_array = cosine_sim[idx]

    # Kalau sim_scores_array bentuknya 2D, flatten dulu
    if sim_scores_array.ndim > 1:
        sim_scores_array = sim_scores_array.flatten()

    # Buat list tuple (index, similarity_score)
    sim_scores = list(enumerate(sim_scores_array))

    # Sort berdasarkan similarity descending, ambil 10 teratas kecuali diri sendiri
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Hapus diri sendiri (indeks idx)
    sim_scores = [x for x in sim_scores if x[0] != idx]

    # Ambil 10 teratas
    sim_scores = sim_scores[:10]

    ramen_indices = [i[0] for i in sim_scores]

    recommended = df.iloc[ramen_indices][['Brand', 'Variety', 'Style', 'Country', 'Stars']]

    return recommended.reset_index(drop=True)
result = recommend_ramen_content_based('Singapore Curry')
print(result)
```

#### Output
Sistem memberikan to-10 rekomendasi teratas dengan sistem rekomendasi dari Content-Based Filtering

| No | Brand         | Variety                              | Style | Country   | Stars |
| -- | ------------- | ------------------------------------ | ----- | --------- | ----- |
| 1  | Snapdragon    | Singapore Laksa Curry Soup Bowl      | Bowl  | USA       | 4.25  |
| 2  | Prima Taste   | Singapore Curry La Mian              | Pack  | Singapore | 5.00  |
| 3  | Myojo         | Extra Spicy Singapore Curry Big Bowl | Bowl  | Singapore | 5.00  |
| 4  | Nissin        | Cup Noodles Singapore Laksa          | Cup   | Japan     | 5.00  |
| 5  | Koka          | Curry                                | Pack  | Singapore | 4.25  |
| 6  | Prima Taste   | Singapore Curry Wholegrain La Mian   | Pack  | Singapore | 5.00  |
| 7  | Trident       | Singapore Soft Noodles               | Pack  | Australia | 2.75  |
| 8  | Samyang Foods | Curry Noodle                         | Cup   | Japan     | 3.75  |
| 9  | Prima Taste   | Singapore Laksa La Mian              | Pack  | Singapore | 5.00  |
| 10 | Golden Mie    | Chicken Curry                        | Pack  | Dubai     | 3.75  |


Kelebihan

Model ini mampu merekomendasikan ramen berdasarkan kemiripan deskripsi (fitur Variety), sehingga sangat personal dan dapat memberikan rekomendasi produk serupa tanpa bergantung pada data pengguna lain. Pendekatan ini juga memungkinkan rekomendasi untuk produk baru yang belum memiliki rating.

Kekurangan

Model ini kurang efektif untuk pengguna baru yang belum memiliki preferensi karena hanya mengandalkan fitur produk. Selain itu, rekomendasi yang diberikan cenderung terbatas pada produk yang sangat mirip sehingga kurang eksploratif dan variasi rekomendasi kurang luas.


## 2. Collaborative Filtering

Pada pendekatan Collaborative Filtering ini, kolom Country digunakan sebagai proxy pengguna untuk membentuk matriks user-item berdasarkan rating ramen. Kemudian, dihitung kemiripan antar pengguna menggunakan cosine similarity untuk menemukan pengguna dengan pola preferensi serupa. Rekomendasi diberikan dengan memilih ramen yang telah diberi rating tinggi oleh pengguna mirip namun belum pernah dicoba oleh pengguna target. Pendekatan ini memungkinkan sistem memberikan rekomendasi yang relevan berdasarkan pola kolektif preferensi komunitas meskipun informasi pengguna asli tidak tersedia secara eksplisit.Berikut tahapannya:

#### 1. Membagi data menjadi training dan testing

kita bagi data menjadi training (80%) dan testing (20%) dengan stratified sampling berdasarkan user agar distribusi rating tetap merata.

```
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Country_mod'])
```

#### 2. Membuat matriks user-item

Pada tahap ini, kita mengembangkan model rekomendasi dengan pendekatan Collaborative Filtering menggunakan kolom Country sebagai proxy pengguna untuk membuat matriks user-item berdasarkan rating ramen. 

```
# Membuat kolom modifikasi Country
country_counts = df['Country'].value_counts()
rare_countries = country_counts[country_counts < 2].index.tolist()
df['Country_mod'] = df['Country'].apply(lambda x: 'Others' if x in rare_countries else x)
country_counts = df['Country'].value_counts()
rare_countries = country_counts[country_counts < 2].index.tolist()
df['Country_mod'] = df['Country'].apply(lambda x: 'Others' if x in rare_countries else x)

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Country_mod'])
print("Data training shape:", train_data.shape)
print("Data testing shape:", test_data.shape)
```
```
# Membuat matriks user-item dari data training
train_matrix = train_data.pivot_table(index='Country', columns='Variety', values='Stars')

print("Shape matriks training user-item:", train_matrix.shape)
train_matrix.head()
```

#### 3. Membuat matriks user-item
```
Untuk membuat rekomendasi berbasis user (User-Based Collaborative Filtering), kita hitung similarity antar user menggunakan cosine similarity dengan mengisi nilai NaN menjadi 0.

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

# Isi NaN dengan 0 untuk hitung similarity
train_matrix_filled = train_matrix.fillna(0)

# Hitung cosine similarity antar user (Country)
user_similarity = 1 - pairwise_distances(train_matrix_filled, metric='cosine')

print("Shape matriks similarity user:", user_similarity.shape)
```
#### 4. Fungsi Rekomendasi Collaborative Filtering
```
def recommend_ramen_collaborative(user_id, user_similarity=user_similarity, train_matrix=train_matrix):
    if user_id not in train_matrix.index:
        return f"User '{user_id}' tidak ditemukan dalam data training."

    # Ambil indeks user
    user_idx = train_matrix.index.get_loc(user_id)

    # Ambil skor similarity user terhadap user lain
    sim_scores = list(enumerate(user_similarity[user_idx]))

    # Urutkan berdasarkan similarity tertinggi (kecuali diri sendiri)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != user_idx]

    # Ambil 10 user paling mirip
    top_users = [train_matrix.index[i[0]] for i in sim_scores[:10]]

    # Ambil ramen yang user belum rating
    user_rated = train_matrix.loc[user_id].dropna().index.tolist()

    # Rata-rata rating ramen dari user mirip yang belum dicoba user_id
    candidate_ratings = train_matrix.loc[top_users].mean(axis=0).dropna()
    candidate_ratings = candidate_ratings.drop(user_rated, errors='ignore')

    # Urutkan dan ambil 10 rekomendasi teratas
    recommendations = candidate_ratings.sort_values(ascending=False).head(10)

    result = pd.DataFrame({
        'Ramen': recommendations.index,
        'Predicted Rating': recommendations.values
    })

    return result.reset_index(drop=True)
```
```
sample_user = train_matrix.index[0]  # ambil user pertama sebagai contoh
print(f"Rekomendasi ramen untuk user '{sample_user}':")
print(recommend_ramen_collaborative(sample_user))
```

#### Output
Sistem memberikan to-10 rekomendasi teratas dengan sistem rekomendasi dari Collaborative Filtering

| No | Ramen                                               | Predicted Rating |
| -- | --------------------------------------------------- | ---------------- |
| 1  | Spicy Black Pepper                                  | 5.0              |
| 2  | Sour Soup & Minced Meat Flavor Chef's Grain Noodles | 5.0              |
| 3  | Straits Reborn Laksa                                | 5.0              |
| 4  | Straits Kitchen Laksa                               | 5.0              |
| 5  | Spicy King Bowl Noodle Spicy Chicken                | 5.0              |
| 6  | Spicy King Spicy Beef                               | 5.0              |
| 7  | Chow Mein Japanese Style Noodles Yakisoba           | 5.0              |
| 8  | Chongqing Noodles Spicy Hot Flavor                  | 5.0              |
| 9  | Creamy Soup With Crushed Noodles Curry Flavor       | 5.0              |
| 10 | Creamy Soup With Crushed Noodles Sweet Corn Flavor  | 5.0              |



Kelebihan

Model ini dapat menangkap pola preferensi pengguna secara kolektif berdasarkan rating dari negara-negara yang mirip, sehingga mampu memberikan rekomendasi yang lebih personal dan relevan. Pendekatan ini dapat merekomendasikan ramen yang belum pernah dicoba pengguna berdasarkan pengalaman komunitas serupa.

Kekurangan

Karena menggunakan Country sebagai proxy user, model ini memiliki keterbatasan jika data pengguna asli tidak tersedia dan dapat menghasilkan rekomendasi yang kurang spesifik. Selain itu, sparsity data yang tinggi dapat mempengaruhi kualitas rekomendasi dan model membutuhkan data rating yang cukup lengkap agar efektif.


## Evaluasi
Evaluasi sistem rekomendasi dilakukan dengan mengukur sejauh mana kedua model berhasil memenuhi tujuan proyek.

- ***Goal 1***
Evaluasi Content-Based Filtering difokuskan pada kemampuan model memberikan rekomendasi ramen serupa berdasarkan karakteristik produk. Keberhasilan model ini dapat dilihat dari relevansi dan konsistensi rekomendasi yang didasarkan pada kemiripan deskripsi ramen, serta kemunculan produk dengan rating tinggi yang menunjukkan kualitas rekomendasi konten.

- ***Goal 2 ***
Evaluasi Collaborative Filtering menilai efektivitas model dalam memanfaatkan data rating pengguna untuk menghasilkan rekomendasi yang relevan. Metrik seperti RMSE atau MAE digunakan untuk mengukur akurasi prediksi rating pada data testing, sementara keberhasilan memberikan rekomendasi top-N yang sesuai preferensi pengguna menjadi indikator utama relevansi dan personalisasi rekomendasi.


### Kesimpulan Evaluasi

Sistem rekomendasi ramen yang dikembangkan menggunakan pendekatan Content-Based Filtering dan Collaborative Filtering berhasil memberikan rekomendasi yang relevan dan personal sesuai dengan karakteristik produk serta preferensi pengguna. Content-Based Filtering efektif dalam merekomendasikan produk serupa berdasarkan deskripsi ramen, sementara Collaborative Filtering mampu memanfaatkan pola rating pengguna untuk meningkatkan akurasi dan personalisasi rekomendasi. Kedua model saling melengkapi dalam menghadirkan solusi yang komprehensif, sehingga sistem ini dapat membantu pengguna menemukan ramen yang sesuai dengan selera dan preferensi mereka secara lebih optimal.



















