# Sesi 1 — Introduction to Python & NumPy
### Dokumentasi Penjelasan Notebook | Speech Recognition Lab, BINUS University

Dokumen ini adalah pendamping notebook `code/Sesi 1 - Introduction to Python.ipynb`.
Isinya penjelasan lengkap tiap bagian, fungsi yang dipakai, dan hal-hal yang perlu ditekankan saat mengajar.

---

## Struktur Notebook

| Bagian | Topik | Estimasi |
|---|---|---|
| 1 | Output, Variable, Input, Operator | 15 menit |
| 2 | Selection & Repetition | 15 menit |
| 3 | Function | 10 menit |
| 4 | Built-in Data Structure | 20 menit |
| 5 | Slicing | 10 menit |
| 6 | NumPy: Array, Atribut, Generator | 20 menit |
| 7 | NumPy: Indexing, Slicing, Operasi | 15 menit |
| 8 | NumPy: Stacking | 10 menit |
| — | Studi kasus + Latihan | 15 menit |

**Prasyarat:** Python 3.8+ dan NumPy (`pip install numpy`). Tidak ada dependency lain.

---

# BAGIAN 1 — Output, Variable, Input, Operator

## 1.1 `print()`

Menampilkan nilai ke layar. Parameter yang perlu diketahui:

| Parameter | Default | Fungsi |
|---|---|---|
| `*values` | — | Nilai-nilai yang ditampilkan, dipisah koma |
| `sep` | `" "` | Pemisah antar nilai |
| `end` | `"\n"` | Karakter penutup di akhir |

```python
print("A", "B", sep=" - ")   # A - B
print("x", end=" ")          # tidak pindah baris
```

> **Poin ajar:** jelaskan bahwa `end=" "` berguna saat mencetak hasil loop dalam satu baris — dipakai lagi di Bagian 2.

## 1.2 Variable & Tipe Data

Python memakai **dynamic typing**: tipe ditentukan otomatis dari nilai yang diberikan, tidak perlu ditulis.

| Tipe | Contoh | Keterangan |
|---|---|---|
| `str` | `"Stanley"` | Teks |
| `int` | `21` | Bilangan bulat |
| `float` | `3.75` | Bilangan desimal |
| `bool` | `True` / `False` | Nilai logika |

**Fungsi yang dipakai:** `type(x)` mengembalikan tipe data dari `x`.

### f-string

Cara memformat string: awali dengan `f`, tulis variable di dalam `{}`.

```python
f"Durasi: {durasi:.2f} detik"    # 2 angka di belakang koma
f"Total: {durasi * 16000:.0f}"   # ekspresi boleh langsung ditulis
```

Format specifier yang sering dipakai: `:.2f` (float 2 desimal), `:.0f` (bulat), `:10s` (string rata kiri selebar 10 karakter — dipakai di Bagian 4.4).

## 1.3 `input()`

Membaca ketikan user dan **selalu mengembalikan `str`**.

```python
umur = int(input("Umur: "))   # WAJIB dikonversi kalau mau dihitung
```

> **Poin ajar:** ini sumber bug klasik mahasiswa — `input()` yang tidak dikonversi membuat `"5" + 1` error, atau `"5" * 3` menghasilkan `"555"`, bukan `15`.
>
> **Catatan teknis:** di notebook, cell `input()` akan menunggu isian dan menahan eksekusi cell berikutnya. Karena itu di notebook baris `input()` sengaja dikomentari dan diganti simulasi, supaya notebook bisa di-*Run All* tanpa macet. Saat demo di kelas, buka komentarnya.

## 1.4 Operator

| Kategori | Operator | Catatan penting |
|---|---|---|
| Aritmatika | `+` `-` `*` `/` `//` `%` `**` | `/` **selalu** menghasilkan `float` (`10 / 2` → `5.0`) |
| Perbandingan | `==` `!=` `>` `<` `>=` `<=` | Hasilnya `bool` |
| Logika | `and` `or` `not` | Bukan `&&`, `\|\|`, `!` seperti C/Java |

- `//` (floor division): membulatkan hasil bagi **ke bawah** → `17 // 5 = 3`
- `%` (modulo): sisa bagi → `17 % 5 = 2`. Sering dipakai untuk cek genap/ganjil (`x % 2 == 0`)
- `**` (pangkat) → `17 ** 2 = 289`

---

# BAGIAN 2 — Selection & Repetition

## 2.1 `if` / `elif` / `else`

```python
if kondisi_1:
    ...
elif kondisi_2:
    ...
else:
    ...
```

**Yang wajib ditekankan:** Python memakai **indentasi (4 spasi)** untuk menandai blok kode, bukan `{}`. Salah indentasi = `IndentationError`. Setiap baris `if`/`elif`/`else`/`for`/`while`/`def` diakhiri titik dua `:`.

Pengecekan dilakukan **dari atas ke bawah** dan berhenti pada kondisi pertama yang bernilai `True`.

### Ternary operator

Bentuk singkat `if-else` satu baris:

```python
status = "Lulus" if nilai >= 65 else "Tidak Lulus"
```

## 2.2 `for` + `range()`

`for` melakukan iterasi terhadap setiap item dalam sebuah koleksi.

**`range(start, stop, step)`** menghasilkan deret angka. Aturan kunci: **`stop` tidak ikut**.

| Pemanggilan | Hasil |
|---|---|
| `range(5)` | 0, 1, 2, 3, 4 |
| `range(1, 10, 2)` | 1, 3, 5, 7, 9 |
| `range(10, 0, -1)` | 10, 9, …, 1 |

**Fungsi bantu:**
- `enumerate(koleksi)` — mengembalikan pasangan `(index, nilai)` sehingga tidak perlu counter manual
- `len(x)` — jumlah elemen
- `.upper()` / `.lower()` — method string untuk mengubah huruf besar/kecil

## 2.3 `while`, `break`, `continue`

`while` mengulang **selama** kondisinya `True`.

> **Poin ajar:** wajib ada baris yang mengubah kondisi (`n -= 1`). Kalau lupa → infinite loop. Di Jupyter, hentikan dengan tombol **Interrupt (■)**.

| Keyword | Efek |
|---|---|
| `break` | Menghentikan loop sepenuhnya |
| `continue` | Melewati sisa iterasi saat ini, lanjut ke iterasi berikutnya |

---

# BAGIAN 3 — Function

Function = blok kode bernama yang bisa dipanggil berulang. Tujuannya menghindari duplikasi kode.

```python
def nama_function(parameter1, parameter2=default):
    """Docstring: penjelasan singkat function."""
    return hasil
```

### Konsep yang diajarkan di notebook

| Konsep | Contoh di notebook | Penjelasan |
|---|---|---|
| Parameter & `return` | `luas_persegi_panjang(5, 3)` | Nilai dikirim masuk, hasil dikembalikan |
| Keyword argument | `luas_persegi_panjang(lebar=4, panjang=10)` | Urutan argumen jadi bebas |
| Default parameter | `durasi_audio(n, sample_rate=16000)` | Argumen opsional, pakai default kalau tidak diisi |
| Multiple return | `return min(a), max(a), mean` | Sebenarnya mengembalikan satu **tuple**, lalu di-*unpack* |
| `lambda` | `kuadrat = lambda x: x ** 2` | Function anonim satu baris |

**Built-in function yang dipakai:** `min()`, `max()`, `sum()`, `len()`, `map()`, `list()`.

> **Poin ajar:** `return min(a), max(a), rata` menghubungkan Bagian 3 dengan Bagian 4 — jelaskan bahwa Python membungkusnya jadi tuple, lalu `t, b, r = statistik(data)` adalah *tuple unpacking*.

---

# BAGIAN 4 — Built-in Data Structure

Tabel pembanding (juga ada di notebook):

| Struktur | Sintaks | Terurut | Mutable | Duplikat | Akses |
|---|---|---|---|---|---|
| **List** | `[1, 2, 3]` | Ya | Ya | Boleh | Index |
| **Tuple** | `(1, 2, 3)` | Ya | **Tidak** | Boleh | Index |
| **Set** | `{1, 2, 3}` | Tidak | Ya | **Tidak** | — (tidak bisa di-index) |
| **Dictionary** | `{"a": 1}` | Ya (3.7+) | Ya | Key unik | Key |

## 4.1 List

| Method / Fungsi | Fungsi |
|---|---|
| `.append(x)` | Menambah `x` di akhir list |
| `.insert(i, x)` | Menyisipkan `x` pada index `i` |
| `.remove(x)` | Menghapus berdasarkan **nilai** (bukan index), yang pertama ditemukan |
| `.pop()` | Mengambil **dan** menghapus elemen terakhir (`.pop(i)` untuk index tertentu) |
| `len(lst)` | Jumlah elemen |
| `sorted(lst)` | Mengembalikan list **baru** yang terurut (`.sort()` mengubah list aslinya) |
| `max()` / `min()` | Nilai terbesar / terkecil |

**List comprehension** — cara ringkas membuat list baru:

```python
[ekspresi for item in koleksi if kondisi]

[x ** 2 for x in range(1, 6)]           # [1, 4, 9, 16, 25]
[x for x in range(10) if x % 2 == 0]    # [0, 2, 4, 6, 8]
```

Setara dengan `for` biasa + `.append()`, tapi lebih ringkas dan lebih cepat.

## 4.2 Tuple

Sama seperti list tapi **immutable** — tidak bisa diubah setelah dibuat. Percobaan `tup[0] = x` menghasilkan `TypeError`.

Kegunaan: menyimpan data yang memang tidak boleh berubah (koordinat, konfigurasi, metadata file audio), dan sebagai hasil `return` ganda dari function.

**Unpacking:** `x, y = koordinat` — memecah tuple ke beberapa variable sekaligus.

## 4.3 Set

Koleksi **unik** dan **tidak terurut**. Duplikat otomatis dibuang saat pembuatan.

| Operator | Nama | Arti |
|---|---|---|
| `A \| B` | Union | Semua elemen dari A dan B |
| `A & B` | Intersection | Elemen yang ada di keduanya |
| `A - B` | Difference | Ada di A tapi tidak di B |

Use case paling umum: menghapus duplikat dari list → `sorted(set(lst))`.

> **Catatan:** di notebook dipakai `sorted(set(...))`, bukan `list(set(...))`, karena set tidak punya urutan sehingga `list(set(...))` bisa menghasilkan urutan berbeda-beda dan membingungkan mahasiswa.

## 4.4 Dictionary

Menyimpan pasangan **key → value**. Akses dengan key, bukan index.

| Method | Fungsi |
|---|---|
| `d[key]` | Ambil value; **error** (`KeyError`) kalau key tidak ada |
| `d.get(key, default)` | Ambil value; mengembalikan `default` kalau key tidak ada (aman) |
| `d[key] = value` | Menambah key baru **atau** mengubah value yang sudah ada |
| `.keys()` | Semua key |
| `.values()` | Semua value |
| `.items()` | Pasangan `(key, value)` — dipakai untuk looping |

```python
for key, value in mahasiswa.items():
    print(f"{key:10s}: {value}")
```

Value boleh berupa tipe apa saja, termasuk list (di notebook: `"nilai": [90, 85, 95]`).

---

# BAGIAN 5 — Slicing

Format: **`data[start : stop : step]`**

| Aturan | Penjelasan |
|---|---|
| `start` | Termasuk (inclusive). Kosong = dari awal |
| `stop` | **Tidak** termasuk (exclusive). Kosong = sampai akhir |
| `step` | Loncatan. Default 1. Negatif = mundur |

Ilustrasi index:

```
data  =  [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
index     0   1   2   3   4   5   6   7   8   9
negatif -10  -9  -8  -7  -6  -5  -4  -3  -2  -1
```

| Ekspresi | Hasil | Arti |
|---|---|---|
| `data[2:5]` | `[20, 30, 40]` | Index 2, 3, 4 |
| `data[:4]` | `[0, 10, 20, 30]` | Dari awal sampai index 3 |
| `data[6:]` | `[60, 70, 80, 90]` | Index 6 sampai akhir |
| `data[::2]` | `[0, 20, 40, 60, 80]` | Loncat 2 |
| `data[::-1]` | terbalik | Membalik urutan |
| `data[-3:]` | `[70, 80, 90]` | 3 elemen terakhir |

Slicing berlaku untuk **semua sequence**: list, tuple, string, dan array NumPy.

> **Relevansi ke Speech Recognition:** memotong potongan sinyal audio (`sinyal[:16000]` = 1 detik pertama pada 16 kHz) adalah operasi slicing biasa. Ini alasan slicing diajarkan sebelum masuk NumPy.

---

# BAGIAN 6 — NumPy: Array, Atribut, & Generator

**NumPy** = library komputasi numerik standar Python. Objek intinya adalah **`ndarray`** (n-dimensional array).

**Kenapa tidak pakai list biasa?**

| Aspek | List Python | NumPy array |
|---|---|---|
| Tipe elemen | Campuran | Seragam (satu `dtype`) |
| Kecepatan | Lambat (looping) | Cepat (vectorized, ditulis dalam C) |
| Memori | Boros | Efisien |
| Operasi matematika | Perlu loop manual | Langsung per elemen |

Konvensi import: `import numpy as np`.

## 6.1 `np.array()`

Mengubah list (atau list of list) menjadi ndarray.

```python
np.array([1, 2, 3])                # 1D — vektor
np.array([[1, 2, 3], [4, 5, 6]])   # 2D — matriks
```

Jumlah tingkat kurung siku = jumlah dimensi.

## 6.2 Atribut Array

| Atribut | Contoh hasil | Arti |
|---|---|---|
| `.shape` | `(2, 3)` | Ukuran tiap dimensi → 2 baris, 3 kolom |
| `.ndim` | `2` | Jumlah dimensi |
| `.size` | `6` | Total elemen (= perkalian semua nilai shape) |
| `.dtype` | `int64` | Tipe data elemen |

> Perhatikan: ini **atribut**, bukan method — ditulis tanpa tanda kurung (`arr.shape`, bukan `arr.shape()`).

**`dtype`** bisa ditentukan sendiri: `np.array([1,2,3], dtype=np.float32)`. Relevan untuk audio, karena sinyal suara umumnya disimpan sebagai `float32`.

**`.reshape(baris, kolom)`** mengubah bentuk array tanpa mengubah isi. Syaratnya jumlah elemen harus sama (`np.arange(12)` bisa jadi `3x4`, `2x6`, `2x2x3`, tapi bukan `5x3`). Nilai `-1` berarti "hitung otomatis": `a.reshape(4, -1)` → `(4, 3)`.

## 6.3 Array Generator

| Fungsi | Hasil | Catatan |
|---|---|---|
| `np.zeros(n)` / `np.zeros((r, c))` | Semua 0.0 | Untuk 2D, shape ditulis sebagai **tuple** |
| `np.ones(n)` | Semua 1.0 | Bisa `dtype=int` |
| `np.full(n, v)` | Semua bernilai `v` | |
| `np.empty((r, c))` | **Tidak diinisialisasi** | Isinya sisa nilai di memori (acak) |
| `np.arange(start, stop, step)` | Deret dengan step tetap | Seperti `range()`, `stop` tidak ikut |
| `np.linspace(start, stop, n)` | `n` titik merata | `stop` **ikut** (default `endpoint=True`) |
| `np.eye(n)` | Matriks identitas | Diagonal 1, sisanya 0 |

> **Poin ajar — `empty` vs `zeros`:** `np.empty` hanya memesan memori tanpa mengisinya, jadi lebih cepat, tapi isinya sampah. Aman dipakai hanya kalau semua elemen dijamin akan ditimpa. Untuk pemula, defaultnya tetap `np.zeros`.

> **Poin ajar — `arange` vs `linspace`:** `arange` = "saya tahu **step**-nya"; `linspace` = "saya tahu **berapa banyak titik** yang saya mau". Untuk membuat sumbu waktu sinyal audio, `linspace` yang dipakai.

## 6.4 `np.random`

| Fungsi | Hasil |
|---|---|
| `np.random.seed(n)` | Mengunci generator agar hasilnya reproducible |
| `np.random.rand(d0, d1, ...)` | Uniform pada rentang `[0, 1)` |
| `np.random.randn(d0, d1, ...)` | Distribusi normal (mean 0, std 1) — bisa negatif |
| `np.random.randint(low, high, size)` | Bilangan bulat, `high` tidak ikut |
| `np.random.choice(koleksi, size)` | Ambil acak dari koleksi |

> **Poin ajar:** `seed()` penting untuk eksperimen machine learning — tanpa seed, hasil training tidak bisa direproduksi. Perhatikan perbedaan penulisan: `rand(2, 3)` memakai argumen terpisah, sedangkan `zeros((2, 3))` memakai tuple.

---

# BAGIAN 7 — Indexing, Slicing, & Operasi Array

## 7.1 Indexing 2D

Untuk array 2D, format aksesnya **`m[baris, kolom]`**:

| Ekspresi | Arti |
|---|---|
| `m[1, 2]` | Elemen baris 1, kolom 2 |
| `m[0]` | Seluruh baris pertama |
| `m[:, 1]` | Seluruh **kolom** ke-1 (`:` = semua baris) |
| `m[0:2, 1:3]` | Sub-matriks baris 0–1, kolom 1–2 |

> **Poin ajar:** ini keunggulan array dibanding list. Pada list bersarang harus ditulis `lst[1][2]`, dan mengambil satu kolom penuh butuh loop. NumPy cukup `m[:, 1]`.

## 7.2 Boolean Masking

Menyaring elemen berdasarkan kondisi:

```python
data > 10             # array of bool: [True, False, True, ...]
data[data > 10]       # hanya elemen yang True
data[(data > 5) & (data < 30)]   # gabungan kondisi
```

> **Penting:** gunakan `&` dan `|` (bukan `and` / `or`) untuk kondisi pada array, dan setiap kondisi **wajib dibungkus kurung** karena `&` punya prioritas lebih tinggi daripada `>` / `<`.

**`np.where(kondisi, nilai_jika_true, nilai_jika_false)`** — versi vektor dari ternary operator.

## 7.3 Elementwise & Broadcasting

Operasi aritmatika pada array dilakukan **per elemen** tanpa loop — inilah yang disebut **vectorization**.

```python
x + y     # menjumlahkan elemen ke-i dengan elemen ke-i
x * y     # perkalian per elemen (BUKAN perkalian matriks — itu np.dot / @)
x + 100   # broadcasting: skalar disebar ke semua elemen
```

**Broadcasting** = NumPy otomatis "melebarkan" array berukuran lebih kecil agar cocok dengan yang lebih besar.

**Universal function:** `np.sqrt()`, `np.sin()`, `np.cos()`, `np.exp()`, `np.abs()` — semuanya bekerja per elemen.

## 7.4 Agregasi & `axis`

| Fungsi | Arti |
|---|---|
| `.sum()`, `.mean()`, `.std()` | Total, rata-rata, standar deviasi |
| `.min()`, `.max()` | Nilai terkecil / terbesar |
| `.argmin()`, `.argmax()` | **Index** dari nilai terkecil / terbesar |

Parameter **`axis`** menentukan arah perhitungan — ini konsep tersulit di bagian ini:

| `axis` | Arah | Hasil pada array `(2, 3)` |
|---|---|---|
| tidak diisi | Semua elemen | 1 angka |
| `axis=0` | Ke bawah, **per kolom** | 3 angka |
| `axis=1` | Ke samping, **per baris** | 2 angka |

> **Cara mengingat:** `axis=n` berarti dimensi ke-`n` "dihabiskan"/dijumlahkan. Pada shape `(2, 3)`, `axis=0` menghabiskan dimensi baris → tersisa 3 kolom.

---

# BAGIAN 8 — Menggabungkan Array

| Fungsi | Arah | Contoh: dua array `(3,)` | Contoh: dua array `(2,2)` |
|---|---|---|---|
| `np.vstack((a, b))` | Vertikal — menumpuk ke bawah | → `(2, 3)` | → `(4, 2)` |
| `np.hstack((a, b))` | Horizontal — menyambung ke samping | → `(6,)` | → `(2, 4)` |
| `np.concatenate((a, b), axis=n)` | Umum, arah ditentukan `axis` | `axis=0` ≡ vstack | `axis=1` ≡ hstack |
| `A.T` | Transpose — menukar baris dan kolom | | `(2,2)` → `(2,2)` |

> **Poin ajar:** perhatikan array yang digabung ditulis dalam **satu tuple** — `np.vstack((a, b))` dengan dua tanda kurung, bukan `np.vstack(a, b)`. Ini kesalahan yang sangat sering terjadi.
>
> Syarat penggabungan: ukuran pada dimensi yang **tidak** digabung harus sama. `vstack` butuh jumlah kolom sama, `hstack` butuh jumlah baris sama.

---

# Studi Kasus Mini — Sinyal Audio Sederhana

Bagian penutup yang menyatukan semua materi ke konteks Speech Recognition.

### Langkah 1 — Membangkitkan gelombang sinus

```python
sample_rate = 16000
durasi      = 1.0
frekuensi   = 440

t      = np.linspace(0, durasi, int(sample_rate * durasi), endpoint=False)
sinyal = np.sin(2 * np.pi * frekuensi * t)
```

| Baris | Penjelasan |
|---|---|
| `sample_rate = 16000` | Jumlah sample per detik. 16 kHz adalah standar untuk speech |
| `t = np.linspace(...)` | Sumbu waktu: 16000 titik merata dari 0 sampai 1 detik |
| `endpoint=False` | Titik 1.0 tidak diikutkan, agar hasilnya tepat 16000 sample dan bisa disambung mulus |
| `np.sin(2 * np.pi * f * t)` | Rumus gelombang sinus, dihitung untuk **seluruh** array sekaligus (vectorization) |

Hasil verifikasi: `shape = (16000,)`, `ndim = 1`, rentang nilai -1.0 sampai 1.0, mean ≈ 0 (karena sinus simetris), std ≈ 0.707 (= 1/√2, nilai RMS gelombang sinus).

### Langkah 2 — Framing

```python
panjang_frame = 400                                # 25 ms @ 16 kHz
jumlah_frame  = len(sinyal) // panjang_frame       # 40 frame
frames = sinyal[:jumlah_frame * panjang_frame].reshape(jumlah_frame, panjang_frame)
energi = (frames ** 2).mean(axis=1)
```

Sinyal dipotong menjadi frame-frame pendek, lalu dihitung energi tiap frame.
Ini adalah langkah pertama pada hampir semua pipeline ekstraksi fitur suara (MFCC, spektrogram, dsb).

Konsep yang dipakai di sini: slicing (`sinyal[:n]`), floor division (`//`), `reshape`, operasi elementwise (`** 2`), dan agregasi dengan `axis=1`. Semuanya sudah dibahas sebelumnya — tekankan hubungan ini ke mahasiswa.

Hasil energi tiap frame ≈ 0.5, konsisten dengan energi rata-rata gelombang sinus beramplitudo 1.

---

# Latihan & Kunci Jawaban

**1. Konversi suhu**
```python
def konversi_suhu(celsius):
    return celsius * 9 / 5 + 32

print(konversi_suhu(30))   # 86.0
```

**2. List comprehension**
```python
angka = [5, 12, 8, 130, 44, 3]
hasil = [x for x in angka if x > 10]   # [12, 130, 44]
```

**3. Nilai tertinggi dari dictionary**
```python
nilai = {"Andi": 85, "Budi": 92, "Citra": 78}
tertinggi = max(nilai, key=nilai.get)
print(tertinggi, nilai[tertinggi])     # Budi 92
```
*(Alternatif untuk pemula: loop biasa dengan variable penampung.)*

**4. Array acak 4x5**
```python
np.random.seed(0)
arr = np.random.randint(0, 10, size=(4, 5))
print(arr.shape, arr.ndim)
print(arr.sum(axis=0))    # jumlah per kolom
```

**5. Menggabungkan array**
```python
a = np.arange(1, 5)
b = np.arange(5, 9)
print(np.vstack((a, b)))   # shape (2, 4)
print(np.hstack((a, b)))   # shape (8,)
```

---

# Catatan Pengajaran

**Error yang paling sering muncul di kelas:**

| Error | Penyebab | Solusi |
|---|---|---|
| `IndentationError` | Indentasi tidak konsisten | Selalu 4 spasi, jangan campur tab dan spasi |
| `TypeError: can only concatenate str` | `input()` tidak dikonversi | Bungkus dengan `int()` / `float()` |
| `KeyError` | Key dictionary tidak ada | Pakai `.get(key, default)` |
| `NameError` | Cell dijalankan tidak berurutan | Jalankan cell dari atas, atau *Restart & Run All* |
| `IndexError` | Index melebihi panjang data | Cek dengan `len()` / `.shape` |
| `ValueError: cannot reshape` | Jumlah elemen tidak cocok | Pastikan `size` = perkalian shape baru |
| `TypeError` pada `np.vstack(a, b)` | Lupa membungkus dalam tuple | `np.vstack((a, b))` |

**Alur yang disarankan:** jalankan cell satu per satu sambil menjelaskan, lalu ubah nilainya secara langsung (live coding) dan tanyakan ke mahasiswa apa hasilnya sebelum dijalankan.

**Persiapan sebelum kelas:** buka notebook, `Kernel → Restart & Run All` untuk memastikan environment siap, lalu `Restart & Clear Output` agar mahasiswa melihat output muncul saat demo.

**Menuju sesi berikutnya:** semua yang dipelajari di sini — array, shape, slicing, `axis`, stacking — akan langsung dipakai saat membaca file audio dan melakukan ekstraksi fitur suara.
