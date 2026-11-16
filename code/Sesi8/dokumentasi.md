# Sesi 8 — HMM-GMM Part 2

Materi utama ada dalam **[sesi8.ipynb](sesi8.ipynb)**. Buka notebook, pilih kernel Python 3.11, lalu jalankan sel dari atas ke bawah. Jika dependensi belum tersedia, aktifkan baris `%pip install` pada sel pertama. Notebook dapat dijalankan dari root repository atau folder `code/Sesi8`.

## Alur materi

1. **Ekstraksi fitur:** melanjutkan STFT dan enam puncak frekuensi dari Sesi 7. Audio langsung dibaca dari `code/Sesi7/Audio`. Setiap rekaman menjadi matriks `(frame, 6 fitur)` dengan panjang masing-masing. Sample rate 8 kHz; FFT 256 sampel dan hop 80 berarti frame 32 ms setiap 10 ms.
2. **Pembagian data:** 75% file untuk training, 25% untuk testing. StandardScaler hanya belajar dari data latih.
3. **Training:** satu GMM-HMM per kata. HMM mempelajari urutan hidden state; GMM memodelkan variasi fitur pada setiap state.
4. **Prediksi:** pilih kata dengan log-likelihood tertinggi, dengan asumsi prior kata sama.
5. **Evaluasi dan penyimpanan:** tampilkan akurasi serta confusion matrix, lalu simpan model bersama scaler untuk mencoba WAV lain.

## Hubungan dengan Sesi 7

Kelas `gmmhmm` di Sesi 7 sebenarnya masih memakai satu Gaussian per state. Sesi ini menggunakan `hmmlearn.GMMHMM` untuk menghadirkan beberapa Gaussian per state dan menyelesaikan pengenal kata dengan kode singkat. Notebook Sesi 7 tidak perlu dijalankan atau diubah.

Library menjalankan forward–backward dan EM/Baum–Welch: menghitung posterior state dan mixture, lalu memperbarui parameter berdasarkan seluruh rekaman latih. `X` berisi gabungan frame; argumen **`lengths` menjaga batas tiap rekaman**, sehingga akhir satu file tidak dianggap tersambung ke awal file berikutnya.

| Parameter | Penjelasan |
| --- | --- |
| `n_components=3` | Tiga hidden state per model kata; bukan jumlah kata. |
| `n_mix=2` | Dua Gaussian per state. Jika 1, menjadi Gaussian HMM. |
| `covariance_type='diag'` | Setiap fitur memiliki varians sendiri; kovarians antarfitur diabaikan. |
| `n_iter=30` | Maksimum 30 iterasi EM. |
| `implementation='log'` | Mengurangi underflow melalui perhitungan dalam domain log. |

Transisi dipelajari secara bebas, mengikuti pendekatan awal Sesi 7. State tidak otomatis sama dengan fonem. Parameter `*_prior` dan `*_weight` memberi regularisasi agar probabilitas dan estimasi Gaussian tetap valid pada dataset kecil.

## Membaca hasil

Akurasi adalah proporsi file uji yang diprediksi benar. Pada confusion matrix, **baris = aktual**, **kolom = prediksi**; diagonal menunjukkan jawaban benar. Log-likelihood bukan confidence dan boleh bernilai negatif.

Contoh ini mengenali **satu kata per WAV**, belum mentranskripsikan kalimat. Kata di luar kosakata tetap dipilihkan salah satu label. Dataset kecil dan split per file belum menjamin pembicara train/test berbeda. Gunakan validation set saat memilih parameter, dan test set untuk evaluasi akhir.

Sudah diuji **Run All** pada Python 3.11: 225 rekaman, 168 latih, 57 uji; akurasi **85,96%**. Parameter model tersimpan juga diperiksa valid. Regularisasi memunculkan satu pesan `Model is not converging` saat likelihood sedikit menurun; seluruh sel tetap selesai. Akurasi split contoh bukan jaminan akurasi rekaman baru.

**Latihan:** ubah `n_mix` menjadi 1 dan jelaskan perbedaannya, atau ganti `audio_baru` pada sel terakhir dengan rekaman sendiri. Setelah mengubah fitur atau parameter training, latih dan simpan ulang model. File `model_sesi8.joblib` dibuat otomatis saat sel penyimpanan dijalankan.

Referensi: [tutorial resmi hmmlearn](https://hmmlearn.readthedocs.io/en/stable/tutorial.html) dan [API GMMHMM](https://hmmlearn.readthedocs.io/en/stable/api.html#hmmlearn.hmm.GMMHMM).
