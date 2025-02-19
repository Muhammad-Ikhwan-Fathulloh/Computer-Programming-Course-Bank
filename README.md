# **Silabus: Pemrograman Python untuk Data Science (14 Pertemuan)**

| **Pertemuan** | **Topik**                                | **Subtopik** |
|--------------|-----------------------------------------|-------------|
| 1           | Pengenalan Python & Instalasi          | Sejarah Python, Kelebihan, Instalasi Python & Jupyter Notebook, Menjalankan Python (REPL, Script, Notebook) |
| 2           | Sintaks Dasar & Tipe Data              | Variabel, Aturan Penamaan, Tipe Data (Integer, Float, String, Boolean), Konversi Tipe Data, F-String |
| 3           | Operasi Dasar & Struktur Kontrol       | Operasi Aritmatika, Perbandingan, Logika, if-else, Looping (for, while), Loop Control (break, continue) |
| 4           | Struktur Data Python                   | List, Tuple, Dictionary, Set, Operasi Dasar, List Comprehension |
| 5           | Fungsi & Pemrograman Modular           | Fungsi Built-in, Fungsi Kustom (def), Lambda Function, Map, Filter, Reduce, Exception Handling |
| 6           | Pandas: Manipulasi Data                | Membaca CSV & Excel (pd.read_csv, pd.read_excel), DataFrame & Series, Indexing, Filtering, Sorting |
| 7           | Pandas: Transformasi Data              | Grouping (groupby), Pivot Table, Missing Data (fillna, dropna), Data Cleaning |
| 8           | Matplotlib & Seaborn: Visualisasi Data | Scatter Plot, Line Chart, Histogram, Bar Plot, Heatmap |
| 9           | NumPy: Perhitungan Numerik             | Array, Indexing, Slicing, Operasi Matematika, Broadcasting, Reshaping |
| 10          | Statistika Dasar dalam Data Science    | Mean, Median, Modus, Standar Deviasi, Korelasi, Distribusi Normal |
| 11          | Machine Learning dengan Scikit-Learn   | Konsep Machine Learning, Linear Regression, Classification (KNN, Decision Tree) |
| 12          | Clustering & Unsupervised Learning     | K-Means, PCA (Principal Component Analysis) |
| 13          | Mini Project: Analisis Data            | Studi Kasus: Prediksi Harga, Customer Segmentation, atau Analisis Data Lainnya |
| 14          | Review & Final Project                 | Presentasi Hasil Mini Project, Evaluasi & Feedback |

## **Pertemuan 1: Pengenalan Python & Instalasi**

### **1.1 Sejarah dan Kelebihan Python**
Python adalah bahasa pemrograman yang dikembangkan oleh **Guido van Rossum** pada tahun 1991. Bahasa ini terkenal karena **kemudahan sintaksis**, **komunitas yang besar**, dan **kompatibilitas luas** dengan berbagai pustaka.

### **1.2 Instalasi Python & Jupyter Notebook**
- Download dan install **Python** dari [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Install **Jupyter Notebook** menggunakan perintah berikut:
  ```bash
  pip install jupyter
  ```
- Jalankan Jupyter Notebook dengan perintah:
  ```bash
  jupyter notebook
  ```

---

## **Pertemuan 2: Sintaks Dasar & Tipe Data**

### **2.1 Variabel dan Aturan Penamaan**
Python menggunakan aturan penamaan variabel sebagai berikut:
- Harus dimulai dengan huruf atau underscore `_`
- Tidak boleh menggunakan kata kunci Python

```python
nama = "Ikhwan"
umur = 25
is_student = True
```

### **2.2 Tipe Data**
```python
integer = 10  # Integer
float_num = 3.14  # Float
string = "Hello, Python!"  # String
boolean = True  # Boolean
```

---

## **Pertemuan 3: Operasi Dasar & Struktur Kontrol**

### **3.1 Operasi Aritmatika**
```python
a = 10
b = 3
print(a + b)  # Penjumlahan
print(a - b)  # Pengurangan
print(a * b)  # Perkalian
print(a / b)  # Pembagian
print(a ** b)  # Pangkat
```

### **3.2 Struktur Kontrol**
```python
angka = 10
if angka > 5:
    print("Angka lebih besar dari 5")
```

```python
for i in range(5):
    print(i)
```

---

## **Pertemuan 4: Struktur Data Python**

```python
my_list = [1, 2, 3, 4]
my_tuple = (1, 2, 3, 4)
my_dict = {"nama": "Ikhwan", "umur": 25}
my_set = {1, 2, 3, 4}
```

---

## **Pertemuan 5: Fungsi & Pemrograman Modular**

```python
def salam(nama):
    return f"Halo, {nama}!"

print(salam("Ikhwan"))
```

```python
# Lambda Function
kali = lambda x, y: x * y
print(kali(2, 3))
```

---

## **Pertemuan 6: Pandas - Manipulasi Data**

```python
import pandas as pd

data = pd.read_csv("data.csv")
print(data.head())
```

---

## **Pertemuan 7: Pandas - Transformasi Data**

```python
# Handling missing data
data.fillna(0, inplace=True)
data.dropna(inplace=True)
```

---

## **Pertemuan 8: Matplotlib & Seaborn - Visualisasi Data**

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data['umur'])
plt.show()
```

---

## **Pertemuan 9: NumPy - Perhitungan Numerik**

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(arr * 2)  # Operasi array
```

---

## **Pertemuan 10: Statistika Dasar dalam Data Science**

```python
print(data['umur'].mean())  # Mean
print(data['umur'].median())  # Median
print(data['umur'].mode())  # Modus
```

---

## **Pertemuan 11: Machine Learning dengan Scikit-Learn**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([10, 20, 30, 40, 50])

model = LinearRegression()
model.fit(X, y)
print(model.predict([[6]]))
```

---

## **Pertemuan 12: Clustering & Unsupervised Learning**

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(data[['umur', 'pendapatan']])
```

---

## **Pertemuan 13-14: Mini Project & Final Project**
- Menganalisis dataset nyata dengan teknik Data Science
- Menerapkan model machine learning
- Presentasi hasil

---

## **Kesimpulan**
Silabus ini berfokus pada **Python untuk Data Science**, mencakup **Python dasar, analisis data, statistik, dan machine learning**. 🚀
