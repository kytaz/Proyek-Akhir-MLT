# Laporan Proyek Machine Learning - Tazky Khumaira Tsany

## Project Overview

Industri makanan instan, khususnya produk ramen, mengalami pertumbuhan pesat dalam beberapa dekade terakhir dan telah menjadi salah satu makanan cepat saji favorit di berbagai negara (Kwon et al., 2020). Dengan banyaknya variasi produk ramen yang tersedia di pasaran, konsumen seringkali mengalami kesulitan dalam memilih produk yang sesuai dengan preferensi rasa dan kualitas yang diinginkan. Oleh karena itu, sistem rekomendasi menjadi solusi penting untuk membantu konsumen menemukan ramen yang tepat berdasarkan selera dan penilaian pengguna sebelumnya.

Dataset ramen ratings yang berisi informasi detail mengenai berbagai merek, varian, dan rating dari konsumen merupakan sumber data yang kaya untuk mengembangkan sistem rekomendasi berbasis machine learning. Pendekatan seperti Content-Based Filtering yang memanfaatkan fitur deskriptif produk dan Collaborative Filtering yang menggunakan pola preferensi pengguna telah banyak digunakan dalam sistem rekomendasi produk makanan dan minuman (Li & Karahanna, 2015). Sistem ini mampu meningkatkan pengalaman pengguna dengan memberikan rekomendasi personalisasi yang relevan dan efektif.

Namun, tantangan seperti data pengguna yang terbatas dan sparsity rating seringkali menjadi hambatan dalam pengembangan model rekomendasi yang akurat (Zhang et al., 2019). Oleh karena itu, penelitian dan pengembangan metode hybrid yang menggabungkan kedua pendekatan tersebut menjadi sangat relevan untuk meningkatkan kualitas rekomendasi ramen. Dengan memanfaatkan dataset ramen ratings, proyek ini bertujuan mengembangkan dan mengevaluasi sistem rekomendasi yang dapat membantu konsumen memilih produk ramen dengan lebih mudah dan tepat.



### Refrensi
- Kwon, D., Kim, H., & Lee, H. (2020). Consumer preferences for instant noodle products: A conjoint analysis. Journal of Food Science, 85(5), 1503-1512.

- Li, X., & Karahanna, E. (2015). Online recommendation systems in the food industry: A review. International Journal of Hospitality Management, 50, 101-112.

- Zhang, Y., Chen, X., & Wang, J. (2019). Addressing sparsity and cold start problems in food recommendation systems. Expert Systems with Applications, 123, 256-268.




## Business Understanding
### A. Problem Statements
- **Pernyataan Masalah 1** : Pengguna kesulitan menemukan rekomendasi ramen yang sesuai dengan preferensi dan selera mereka dari ribuan produk yang tersedia.

- **Pernyataan Masalah 2**: Tidak adanya sistem rekomendasi yang dapat memanfaatkan data rating dan karakteristik produk untuk memberikan rekomendasi ramen yang relevan dan personal.


### B. Goals
- **Goal 1**: Membangun sistem rekomendasi yang mampu memberikan rekomendasi ramen serupa berdasarkan karakteristik produk (content-based filtering).

- **Goal 2**: Mengembangkan sistem rekomendasi berbasis preferensi pengguna (collaborative filtering) dengan memanfaatkan data rating untuk meningkatkan relevansi rekomendasi.

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

### Penghapusan kolom TopTen:
> Kolom ini dihapus karena memiliki banyak nilai yang hilang, yaitu sebanyak 2.539 data, sehingga dihapus untuk menyesuaikan proses data preparation

### Penanganan kolom start dan style
> Tahap ini untuk membersihkan dan mempersiapkan data 'Stars' dan 'Style' dengan format yang sesuai (numerik untuk 'Stars', mengisi missing values untuk 'Style') agar siap digunakan dalam analisis dan pembangunan model rekomendasi

### Konversi Tipe Data:
> Kolom Stars awalnya bertipe objek diubah menjadi tipe numerik (float) agar dapat digunakan dalam perhitungan dan model prediktif.

### Penanganan Missing Values:
> Nilai kosong pada kolom Style diisi dengan label 'Unknown' untuk menjaga konsistensi data tanpa menghapus baris yang valid. Selain itu, nilai kosong pada kolom Country diisi dengan label 'Unknown'.

### Pengelompokan Kelas Minoritas:
> Untuk mendukung pembagian data menggunakan stratified sampling, negara-negara dengan jumlah data sangat sedikit digabungkan ke dalam kategori 'Others' untuk menghindari masalah saat pembagian data.

### Pembuatan dan fitting TF-IDF 
> Pada kolom ‘Variety’ diubah menjadi representasi numerik menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) dengan menghilangkan stop words bahasa Inggris.


## Modeling

Tahap modeling terdiri dari pengembangan dua pendekatan sistem rekomendasi utama untuk menyelesaikan permasalahan ini.

### 1. Model Development Content-Based Filtering

Model ini menggunakan fitur teks dari kolom Variety yang diolah dengan TF-IDF untuk merepresentasikan karakteristik tiap jenis ramen dalam bentuk vektor numerik. Kemudian, cosine similarity dihitung untuk mengukur kemiripan antar ramen berdasarkan deskripsi tersebut. Fungsi rekomendasi dibuat untuk mengembalikan 10 ramen paling mirip dengan input ramen berdasarkan kemiripan konten, sehingga pengguna dapat menemukan produk serupa yang relevan.


#### 1. Menghitung Cosine Similarity

Mengukur kemiripan antar setiap pasangan ramen menggunakan matriks TF-IDF yang dihasilkan.

```
from sklearn.metrics.pairwise import cosine_similarity

# Hitung cosine similarity antar baris TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Cek ukuran matrix similarity
print("Shape cosine similarity matrix:", cosine_sim.shape)
```

#### 2. Evaluasi Content-Based Filtering
```
def recommend_ramen_content_based(title, cosine_sim=cosine_sim, df=df):
    # Membuat indeks dari kolom 'Variety' untuk pencarian
    indices = pd.Series(df.index, index=df['Variety']).drop_duplicates()

    if title not in indices:
        return f"Ramen dengan nama '{title}' tidak ditemukan dalam data."

    # Ambil indeks. Jika indices[title] mengembalikan Series (karena duplikasi nama), ambil indeks pertama.
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0] # Ambil indeks pertama jika multiple matches

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

# Hitung Precision@10 untuk model content-based
precision_at_10_cb = evaluate_content_based(recommend_ramen_content_based, df, k=10, relevant_threshold=4.0)
print(f"Precision10 (Content-Based Filtering): {precision_at_10_cb:.4f}")
```

#### 3. Indeks nama ramen

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


## 2.  Model DevelopmentCollaborative Filtering

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
#### 4. Evalusi Collaborative Filtering
```
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def predict_collaborative(user_id, item_variety, user_similarity, train_matrix):
    
    if user_id not in train_matrix.index or item_variety not in train_matrix.columns:
        return np.nan # Tidak bisa prediksi jika user atau item tidak ada di training

    user_idx = train_matrix.index.get_loc(user_id)
    item_idx = train_matrix.columns.get_loc(item_variety)

    # Ambil skor similarity user terhadap user lain
    sim_scores = user_similarity[user_idx]

    # Temukan user yang juga pernah merating item ini
    users_who_rated_item = train_matrix[item_variety].dropna().index.tolist()

    if not users_who_rated_item:
        return np.nan # Tidak ada user yang pernah merating item ini

    # Filter user similarity hanya untuk user yang merating item ini
    users_to_consider = [u for u in users_who_rated_item if u in train_matrix.index]
    if not users_to_consider:
        return np.nan

    # Dapatkan indeks dari user yang dipertimbangkan
    users_to_consider_indices = [train_matrix.index.get_loc(u) for u in users_to_consider]

    # Ambil skor similarity user_id terhadap user-user yang dipertimbangkan
    relevant_sim_scores = sim_scores[users_to_consider_indices]

    # Ambil rating dari user-user yang dipertimbangkan untuk item ini
    relevant_ratings = train_matrix.loc[users_to_consider, item_variety]

    # Hitung prediksi rating
    # Weighted average of ratings by similar users
    # Handle case where sum of absolute similarities is 0
    if np.sum(np.abs(relevant_sim_scores)) == 0:
        return np.nan # Tidak bisa menghitung prediksi jika tidak ada similarity

    predicted_rating = np.sum(relevant_sim_scores * relevant_ratings) / np.sum(np.abs(relevant_sim_scores))

    return predicted_rating


# Siapkan data test untuk prediksi
y_true = test_data['Stars'].values
y_pred = []
actual_y_true = [] # Simpan nilai true rating yang berhasil diprediksi

# Lakukan prediksi untuk setiap baris di data test
for index, row in test_data.iterrows():
    user = row['Country']
    item = row['Variety']
    true_rating = row['Stars']

    predicted = predict_collaborative(user, item, user_similarity, train_matrix)

    if not np.isnan(predicted):
        y_pred.append(predicted)
        actual_y_true.append(true_rating)


# Hitung RMSE dan MAE
if actual_y_true:
    rmse = np.sqrt(mean_squared_error(actual_y_true, y_pred))
    mae = mean_absolute_error(actual_y_true, y_pred)

    print(f"\nRMSE (Collaborative Filtering - Test Data): {rmse:.4f}")
    print(f"MAE (Collaborative Filtering - Test Data): {mae:.4f}")
else:
    print("\nTidak ada prediksi yang berhasil dilakukan pada data test untuk Collaborative Filtering.")
```

#### 5. Fungsi Rekomendasi Collaborative Filtering
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
Evaluasi Content-Based Filtering difokuskan pada kemampuan model memberikan rekomendasi ramen serupa berdasarkan karakteristik produk. Keberhasilan model ini dapat dilihat dari relevansi dan konsistensi rekomendasi yang didasarkan pada kemiripan deskripsi ramen. Berdasarkan hasil evaluasi, model mencapai Precision@10 sebesar 0,4060, yang berarti sekitar 40,6% dari 10 rekomendasi teratas merupakan produk yang relevan dengan rating tinggi. Hal ini menunjukkan kualitas rekomendasi konten yang baik dan efektif dalam membantu pengguna menemukan ramen yang serupa sesuai preferensi.

- ***Goal 2***
Evaluasi Collaborative Filtering menilai efektivitas model dalam memanfaatkan data rating pengguna untuk menghasilkan rekomendasi yang relevan dan personal. Metrik RMSE sebesar 0,8651 dan MAE sebesar 0,4560 pada data testing menunjukkan tingkat akurasi yang memadai dalam memprediksi rating pengguna terhadap ramen yang belum pernah mereka nilai sebelumnya. Keberhasilan model dalam memberikan rekomendasi top-N yang sesuai preferensi pengguna menandakan kemampuan sistem untuk meningkatkan personalisasi dan relevansi rekomendasi secara signifikan.


### Kesimpulan Evaluasi

Sistem rekomendasi ramen yang dikembangkan menggunakan pendekatan Content-Based Filtering dan Collaborative Filtering berhasil memberikan rekomendasi yang relevan dan personal sesuai dengan karakteristik produk serta preferensi pengguna. Berdasarkan evaluasi, model Content-Based Filtering mencapai Precision@10 sebesar 0,4060, yang menunjukkan bahwa sekitar 40,6% dari 10 rekomendasi teratas merupakan item relevan berdasarkan rating tinggi. Model Collaborative Filtering, di sisi lain, menunjukkan performa prediksi rating yang baik pada data uji dengan nilai RMSE sebesar 0,8651 dan MAE sebesar 0,4560, mengindikasikan tingkat akurasi yang memadai dalam memperkirakan rating pengguna terhadap ramen yang belum pernah mereka lihat sebelumnya. Content-Based Filtering efektif dalam merekomendasikan produk serupa berdasarkan deskripsi ramen, sedangkan Collaborative Filtering memanfaatkan pola rating pengguna untuk meningkatkan akurasi dan personalisasi rekomendasi. Kedua model tersebut saling melengkapi, sehingga sistem rekomendasi ini mampu membantu pengguna menemukan ramen yang sesuai dengan selera dan preferensi mereka secara lebih optimal.


---Ini adalah bagian akhir laporan---


















