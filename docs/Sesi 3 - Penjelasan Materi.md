# Sesi 3 — MFCC Part 1: Introduction to MFCC & Create MFCC using Audio Library
### Dokumentasi Penjelasan Notebook | Speech Recognition Lab, BINUS University

Dokumen ini adalah pendamping notebook `code/session-3/Session 3 - MFCC Part 1.ipynb`.
Isinya teori MFCC secara utuh (6 langkah), rumus-rumus yang dipakai, penjelasan tiap bagian kode, dan hal-hal yang perlu ditekankan saat mengajar.

**Cakupan sesi:**
- **Sesi 3 (Part 1)** — teori MFCC secara utuh + implementasi langkah 1–4 (Normalization → Frame Blocking → Windowing → FFT).
- **Sesi 4 (Part 2)** — lanjut langkah 5–6 (Mel Filterbank → DCT) sampai MFCC jadi.

---

## Struktur Notebook

| Bagian | Topik | Estimasi |
|---|---|---|
| 1 | Apa itu MFCC? Feature extraction, time vs frequency domain | 10 menit |
| 2 | Mel Scale & Cepstral Coefficients (+ eksperimen kode) | 15 menit |
| 3 | 6 Langkah MFCC (gambaran besar) | 10 menit |
| 4 | Setup: import library & membaca file audio | 10 menit |
| 5 | Langkah 1 — Normalization | 10 menit |
| 6 | Langkah 2 — Frame Blocking | 20 menit |
| 7 | Langkah 3 — Windowing | 15 menit |
| 8 | Langkah 4 — FFT & Power Spectrum | 20 menit |
| 9 | Rangkuman, preview Sesi 4, latihan | 10 menit |

**Prasyarat:** Python 3.8+, `numpy`, `scipy`, `matplotlib`, `ipython` (`pip install numpy scipy matplotlib ipython`). File `audio.wav` dan `AudioWav.wav` harus berada **di folder yang sama dengan notebook**.

**Referensi:** PPT *Speech Recognition Session 3–4 (MFCC)* dan ringkasan *Speech Recog MFCC.pdf* di `references/Session 3/`.

---

# BAGIAN 1 — Apa itu MFCC?

## 1.1 Feature Extraction

Bagi komputer, file audio hanyalah **deretan angka** amplitudo. Rekaman 18 detik pada 44100 Hz stereo = **1,6 juta angka**. Terlalu banyak, terlalu mentah, dan terlalu berisik untuk langsung dipakai model.

**Feature extraction** = mengubah sinyal mentah menjadi **fitur**: representasi numerik ringkas yang tetap menyimpan karakteristik penting.

> **Fitur** — properti/karakteristik terukur dari data yang dipakai sebagai input untuk melatih model dan membuat prediksi. Dalam konteks speech recognition: representasi numerik dari karakteristik utama sinyal suara yang bisa dipakai model untuk membedakan/mengenali suara.

**MFCC (Mel-Frequency Cepstral Coefficients)** adalah metode feature extraction paling umum untuk speech recognition. Bedah namanya:

| Kata | Arti |
|---|---|
| **Mel-Frequency** | frekuensi diukur memakai *Mel scale* — skala yang meniru cara telinga manusia mendengar |
| **Cepstral** | fitur diambil dari *cepstrum* — "spektrum dari spektrum" |
| **Coefficients** | hasil akhir berupa sekumpulan koefisien (angka), umumnya 12–13 per frame |

**Tujuan MFCC (poin ajar utama):** meringkas sinyal suara menjadi *beberapa angka per potongan waktu* yang menggambarkan bentuk spektrum suara **sesuai persepsi manusia**, sehingga model bisa membedakan kata, fonem, atau penutur — bukan sekadar membedakan volume atau noise.

## 1.2 Time Domain vs Frequency Domain

| | Time Domain | Frequency Domain |
|---|---|---|
| Sumbu X | waktu (detik) | frekuensi (Hz) |
| Sumbu Y | amplitudo | kekuatan sinyal pada frekuensi tersebut |
| Bentuk | waveform | spektrum |

File `.wav` berada di time domain. Untuk speech recognition, informasi penting (vokal "a" vs "i", suara laki-laki vs perempuan) jauh lebih mudah dilihat di **frequency domain** karena tiap bunyi punya pola frekuensi yang khas.

> **Poin ajar:** tunjukkan waveform (Bagian 4.3) — tanyakan ke mahasiswa: "Bisa tebak ini bunyi apa dari gambar ini?" Hampir pasti tidak bisa. Lalu tunjukkan spectrogram (Bagian 8.5) — di sana pola bunyi jauh lebih terlihat. Itulah alasan kita pindah domain.

---

# BAGIAN 2 — Mel Scale & Cepstral Coefficients

## 2.1 Eksperimen persepsi (dari slide)

Buka https://www.szynalski.com/tone-generator/ di kelas dan mainkan:

| Pasangan | Selisih Hz | Yang terdengar |
|---|---|---|
| C2 (65 Hz) → C4 (262 Hz) | ± 197 Hz | perbedaan **sangat drastis** (2 oktaf) |
| G6 (1568 Hz) → A6 (1760 Hz) | ± 192 Hz | perbedaan **tipis** (1 nada) |

Kesimpulan: **persepsi manusia terhadap frekuensi tidak linear**. Sensitif di frekuensi rendah, "tumpul" di frekuensi tinggi. Kalimat dari slide yang bagus untuk dikutip:

> Perbedaan nada 100 Hz → 300 Hz terdengar lebih drastis daripada 1100 Hz → 1300 Hz. Akan tetapi, perbedaan 100 Mel → 300 Mel akan terdengar **sama** dengan 1100 Mel → 1300 Mel.

## 2.2 Mel Scale

Skala frekuensi yang dirancang agar jarak yang sama di skala Mel **terdengar sama** oleh manusia. Hampir **linear di bawah 1000 Hz**, **logaritmik di atasnya**. Titik referensi: **1000 Hz = 1000 Mel**.

**Rumus Hz → Mel:**

$$\text{Mel}(f) = 2595 \times \log_{10}\left(1 + \frac{f}{700}\right)$$

**Rumus Mel → Hz:**

$$f(\text{Mel}) = 700 \times \left(10^{\frac{\text{Mel}}{2595}} - 1\right)$$

| Konstanta | Arti |
|---|---|
| **700** | titik transisi di mana skala berubah dari linear ke logaritmik |
| **2595** | konstanta empiris agar hasil cocok dengan eksperimen persepsi manusia |

**Kode:**
```python
def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def mel_to_freq(mels):
    return 700.0 * (10 ** (mels / 2595.0) - 1)
```

**Output eksperimen di notebook:**

| Dari (Hz) | Ke (Hz) | Selisih Hz | Selisih Mel |
|---|---|---|---|
| 65 | 262 | 197 | 258.2 |
| 1568 | 1760 | 192 | 91.6 |
| 100 | 300 | 200 | 251.5 |
| 1100 | 1300 | 200 | 118.7 |

> **Poin ajar:** selisih Hz hampir sama di semua baris, tetapi selisih Mel-nya berbeda 2–3 kali lipat — dan **itu cocok dengan apa yang kita dengar**. Ini bukti bahwa Mel scale "benar" secara persepsi. Kedua function ini akan dipakai lagi di Sesi 4 untuk membangun Mel Filterbank.

## 2.3 Cepstral Coefficients

Kumpulan fitur yang umum dipakai di speech processing/recognition — **hasil akhir** proses ekstraksi fitur. Dihitung menggunakan **cepstrum**.

- Istilah *cepstrum* = permainan kata dari *spectrum* ("spec" dibalik menjadi "ceps").
- Cepstrum = spektrum dari log-spektrum. Idenya: log-spektrum suara sendiri berbentuk "gelombang" (envelope halus + harmonik rapat). Kalau kita transformasi lagi, komponen-komponen itu terpisah, dan **beberapa koefisien pertama** cukup mewakili bentuk envelope-nya.
- Sifat yang membuatnya bagus sebagai fitur: **ringkas** (12–13 angka per frame), **tidak saling berkorelasi**, dan **menggambarkan bentuk vocal tract** (rongga mulut/tenggorokan) — hal yang membedakan bunyi satu dengan lainnya.

> **Poin ajar:** cukup sampai konsep. Detail perhitungannya (log + DCT) dibahas di Sesi 4.

---

# BAGIAN 3 — 6 Langkah MFCC

| # | Langkah | Tujuan | Sesi |
|---|---|---|---|
| 1 | **Normalization** | menstandarisasi amplitudo ke rentang tertentu agar volume seragam | 3 |
| 2 | **Frame Blocking** | memecah suara menjadi frame pendek yang saling overlap | 3 |
| 3 | **Windowing** | menghindari distorsi akibat pemotongan frame (memperhalus ujung frame) | 3 |
| 4 | **FFT** | mengubah sinyal dari time domain ke frequency domain | 3 |
| 5 | **Mel Filterbank** | mengubah spektrum frekuensi ke skala Mel | 4 |
| 6 | **DCT** | mengambil fitur utama → cepstral coefficients | 4 |

**Perubahan bentuk data sepanjang pipeline** (dengan `audio.wav`, `FFT_SIZE=2048`, `HOP_SIZE=15`):

```
wavfile.read      → (831744, 2)      int16, stereo
stereo → mono     → (831744,)
Normalization     → (831744,)        rentang -1..1
Frame Blocking    → (1257, 2048)     jumlah_frame × sample_per_frame
Windowing         → (1257, 2048)
FFT               → (1257, 1025)     jumlah_frame × bin_frekuensi   (kompleks)
Power spectrum    → (1257, 1025)     real, ≥ 0
Mel Filterbank    → (1257, n_mel)    misal n_mel = 10–40            ← Sesi 4
DCT               → (1257, n_mfcc)   misal n_mfcc = 12–13           ← Sesi 4
```

> **Poin ajar:** gambarkan tabel shape ini di papan tulis dan isi bersama mahasiswa sambil menjalankan notebook. Mahasiswa yang paham perubahan shape biasanya paham seluruh pipeline.

---

# BAGIAN 4 — Setup: Import Library & Membaca Audio

## 4.1 Library

| Import | Kegunaan |
|---|---|
| `import numpy as np` | kalkulasi matematika & array |
| `from scipy.io import wavfile` | membaca file `.wav` |
| `import scipy.fftpack as fft` | Fast Fourier Transform |
| `from scipy.signal import get_window` | membuat window function |
| `import matplotlib.pyplot as plt` | plotting |
| `from IPython.display import Audio` | memutar audio di notebook |

Inilah yang dimaksud "audio library" pada course outline: `scipy` (I/O, FFT, windowing) di atas `numpy`.

## 4.2 `wavfile.read(path)`

Mengembalikan tuple `(sample_rate, audio)`.

- **`sample_rate`** — jumlah sample per detik (Hz). `audio.wav` = 44100 Hz.
- **`audio`** — array NumPy amplitudo. `dtype=int16` → rentang −32768 s.d. 32767.

**Output notebook:**
```
Sample rate : 44100 Hz
Shape audio : (831744, 2)
Tipe data   : int16
```

### Stereo → mono

Shape `(831744, 2)` berarti **2 kanal** (kiri/kanan). MFCC memakai 1 kanal. Cara paling sederhana: rata-ratakan kedua kanal.

```python
if audio.ndim == 2:
    audio = audio.mean(axis=1)     # axis=1 → rata-rata antar kolom (kanal)
```

> **Catatan teknis (penting):** kode referensi lama memakai `audio.flatten()`. Untuk audio stereo ini **salah** — `flatten()` menyusun sample kiri-kanan berselang-seling menjadi satu array 2× lebih panjang, bukan menggabungkannya. Hasilnya durasi terhitung dua kali lipat dan spektrumnya kacau. Gunakan `mean(axis=1)` (atau ambil satu kanal `audio[:, 0]`).

### Durasi

$$\text{durasi (detik)} = \frac{\text{jumlah sample}}{\text{sample rate}} = \frac{831744}{44100} \approx 18{,}86 \text{ s}$$

## 4.3 Plot waveform

`np.linspace(0, durasi, num=len(audio))` membuat sumbu waktu — satu titik per sample. Ini review langsung dari Sesi 1.

> **Poin ajar:** `Audio(audio, rate=sample_rate)` membuat pemutar audio di notebook. Putar dulu supaya mahasiswa tahu bunyi apa yang sedang dianalisis. Perlu diketahui: `audio.wav` yang disediakan adalah cuplikan **musik**, bukan ucapan — tidak masalah untuk mengajarkan pipeline, tetapi sebutkan bahwa pipeline yang sama dipakai untuk suara manusia.

---

# BAGIAN 5 — Langkah 1: Normalization

**Masalah:** rekaman berbeda punya volume berbeda (jarak mic, gain, dsb.). Tanpa normalisasi, model bisa "belajar" perbedaan volume alih-alih perbedaan isi suara.

**Solusi:** bagi seluruh sinyal dengan **nilai absolut maksimumnya** → rentang −1 s.d. 1.

$$x_{norm}[n] = \frac{x[n]}{\max\left(|x[n]|\right)}$$

```python
def normalize_audio(audio):
    return audio / np.max(np.abs(audio))
```

| Fungsi | Peran |
|---|---|
| `np.abs(audio)` | semua nilai jadi positif |
| `np.max(...)` | nilai terbesar (puncak tertinggi) |
| pembagian | puncak tertinggi menjadi tepat ±1, sisanya proporsional |

**Output notebook:** `min = -0.9950`, `max = 1.0000`. Bentuk gelombang **tidak berubah**, hanya skalanya.

> **Poin ajar:** tanyakan "kenapa `max` hasilnya tepat 1.0000 tapi `min` tidak tepat −1?" Jawab: karena puncak positif (31064) lebih besar daripada puncak negatif (−26918); yang dibagi adalah nilai absolut terbesar, yaitu 31064.

---

# BAGIAN 6 — Langkah 2: Frame Blocking

## 6.1 Kenapa dipotong-potong?

FFT mengasumsikan sinyal **stasioner** (karakteristik tidak berubah terhadap waktu). Suara manusia **non-stasioner** — dalam 1 detik ada banyak bunyi berbeda. FFT pada seluruh rekaman sekaligus akan mencampur semuanya → distorsi/tidak bermakna.

**Solusi:** potong menjadi **frame** sangat pendek (**20–40 ms**). Dalam durasi sependek itu sinyal bisa dianggap stasioner.

## 6.2 Overlapping

Frame dibuat **tumpang tindih**: awal sebuah frame adalah akhir frame sebelumnya. Tujuannya menjaga **kontinuitas informasi** dan **korelasi antar frame** — bunyi yang kebetulan di batas frame tidak hilang, transisi antar frame mulus.

```
audio  : |----------------------------------------------|
frame 0: [==========]
frame 1:      [==========]
frame 2:           [==========]
              ^hop  ^hop
```

## 6.3 Parameter

| Parameter | Arti | Nilai | Catatan |
|---|---|---|---|
| `FFT_SIZE` | jumlah sample per frame (= panjang frame) | 2048 | bebas, tapi **pangkat 2** agar FFT cepat |
| `HOP_SIZE` | jarak awal frame ke awal frame berikutnya (**ms**) | 15 | bebas |

Turunan pada 44100 Hz:
- durasi 1 frame = 2048 / 44100 ≈ **46,4 ms**
- hop dalam sample = 44100 × 15 / 1000 ≈ **662 sample**
- overlap ≈ 2048 − 662 = 1386 sample (≈ 68 % dari frame)

## 6.4 Kode

```python
def frame_blocking(audio, FFT_SIZE, HOP_SIZE, sample_rate):
    audio = np.pad(audio, int(FFT_SIZE / 2), mode="reflect")            # 1
    frame_len = np.round(sample_rate * (HOP_SIZE / 1000)).astype(int)   # 2
    frame_num = int((len(audio) - FFT_SIZE) / frame_len) + 1            # 3
    frames = np.zeros((frame_num, FFT_SIZE))                            # 4
    for i in range(frame_num):                                          # 5
        frames[i] = audio[i * frame_len : i * frame_len + FFT_SIZE]
    return frames
```

| # | Baris | Penjelasan |
|---|---|---|
| 1 | `np.pad(..., mode="reflect")` | tambah `FFT_SIZE/2` sample di awal & akhir. `reflect` = cermin dari audio itu sendiri, bukan nol. Tujuannya agar sample paling awal/akhir tetap bisa berada di **tengah** sebuah frame |
| 2 | `frame_len` | konversi hop dari ms → sample. `/1000` mengubah ms → detik; `np.round().astype(int)` karena index harus bulat |
| 3 | `frame_num` | berapa frame yang muat. Dikurangi `FFT_SIZE` supaya frame terakhir utuh; `+1` untuk frame pertama |
| 4 | `np.zeros((frame_num, FFT_SIZE))` | kerangka array 2D, diisi 0 dulu |
| 5 | slicing loop | frame ke-`i` dimulai di `i * frame_len`, panjangnya `FFT_SIZE` |

**Demo `reflect` dengan array kecil** (ada di notebook, sangat membantu):
```
Asli    : [1 2 3 4 5]
Reflect : [3 2 1 2 3 4 5 4 3]
```

**Output notebook:** `Shape frames : (1257, 2048)`.

> **Poin ajar:** nama variable `frame_len` di kode referensi sebenarnya adalah *panjang hop*, bukan panjang frame (panjang frame = `FFT_SIZE`). Namanya dipertahankan agar konsisten dengan slide, tetapi jelaskan ini ke mahasiswa supaya tidak bingung.
>
> **Catatan teknis:** kode referensi di slide tidak membungkus `frame_num` dengan `int()`. Di NumPy versi baru, `np.zeros((float, int))` dan `range(float)` akan error (`TypeError`). Notebook sudah memperbaikinya.

---

# BAGIAN 7 — Langkah 3: Windowing

## 7.1 Masalah

FFT memperlakukan frame seolah-olah **periodik** (diulang terus). Kalau ujung awal dan akhir frame nilainya berbeda, saat "disambung" terjadi **lompatan nilai** mendadak → muncul di hasil FFT sebagai frekuensi palsu (**spectral leakage**/distorsi).

Dari slide: *"Untuk melakukan FFT, audio harus periodic (sudah, setelah frame blocking) dan continuous. Untuk membuatnya continuous, kita pakai window function pada setiap frame."*

## 7.2 Window function

Fungsi yang **≈ 0 di kedua ujung** dan **1 di tengah**. Tiap frame **dikalikan** dengan window → ujung frame menjadi halus mendekati 0.

$$\text{frame}_{win}[n] = \text{frame}[n] \times w[n]$$

**Hann (Hanning) window:**

$$w[n] = 0{,}5\left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right), \quad 0 \le n \le N-1$$

Tipe window lain: `bartlett`, `blackman`, `hamming`, `flattop`, `welch`, `boxcar` (kotak = tanpa windowing).

## 7.3 `get_window("hann", FFT_SIZE, fftbins=True)`

| Parameter | Arti |
|---|---|
| `"hann"` | jenis window |
| `FFT_SIZE` | panjang window = panjang frame, supaya seluruh frame terkena. Kalau diisi `FFT_SIZE/2`, hanya setengah frame yang di-window |
| `fftbins=True` | window dibuat versi *periodik* (cocok untuk analisis spektral/FFT), bukan versi simetris |

```python
window    = get_window("hann", FFT_SIZE, fftbins=True)   # shape (2048,)
audio_win = framed_audio * window                        # shape (1257, 2048)
```

`framed_audio * window` memakai **broadcasting** NumPy: array `(1257, 2048)` dikalikan `(2048,)` → tiap baris dikalikan window yang sama. Ini review Sesi 1 (operasi array).

> **Poin ajar:** plot "sebelum vs sesudah windowing" pada frame 454 menunjukkan ujung-ujung frame "dijepit" ke 0 sedangkan tengahnya hampir tidak berubah. Frame 454 dipilih karena berisi bunyi yang tertahan (frame-frame awal masih hening).

---

# BAGIAN 8 — Langkah 4: Fast Fourier Transform

## 8.1 Konsep

Fourier Transform memecah sinyal menjadi jumlahan gelombang sinus dengan frekuensi berbeda, dan memberi tahu seberapa besar kontribusi tiap frekuensi. Hasilnya: **time domain → frequency domain**.

Untuk sinyal diskrit: **DFT (Discrete Fourier Transform)**

$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j\frac{2\pi}{N}kn}, \qquad k = 0, 1, \dots, N-1$$

| Simbol | Arti |
|---|---|
| $x[n]$ | sample ke-$n$ dalam frame |
| $N$ | panjang frame (`FFT_SIZE`) |
| $X[k]$ | bilangan kompleks: magnitude & fase dari frekuensi ke-$k$ |

**FFT** = algoritma cepat untuk DFT: $O(N^2) \to O(N \log N)$. Efisien terutama bila $N$ pangkat 2 — alasan `FFT_SIZE = 2048`.

## 8.2 Dua fakta penting

1. **Hasil FFT sinyal real bersifat simetris.** Dari $N$ output, separuh kedua adalah cerminan separuh pertama. Cukup simpan **`1 + FFT_SIZE // 2` = 1025** nilai (frekuensi 0 Hz s.d. Nyquist = `sample_rate/2`).
2. **Hasilnya bilangan kompleks.** Untuk analisis energi kita hitung **power spectrum**:

$$P[k] = |X[k]|^2$$

Frekuensi yang diwakili bin ke-$k$:

$$f_k = k \times \frac{\text{sample\_rate}}{\text{FFT\_SIZE}} \quad\Rightarrow\quad \text{resolusi} = \frac{44100}{2048} \approx 21{,}5 \text{ Hz per bin}$$

## 8.3 Kode

```python
audio_winT = np.transpose(audio_win)                       # (2048, 1257): tiap kolom = 1 frame

audio_fft = np.empty((1 + FFT_SIZE // 2, audio_winT.shape[1]),
                     dtype=np.complex64, order='F')        # (1025, 1257)

for i in range(audio_fft.shape[1]):
    audio_fft[:, i] = fft.fft(audio_winT[:, i], axis=0)[:audio_fft.shape[0]]

audio_fft   = np.transpose(audio_fft)                      # (1257, 1025)
audio_power = np.square(np.abs(audio_fft))                 # |X|^2
```

| Baris | Penjelasan |
|---|---|
| `np.transpose` pertama | supaya tiap **kolom** = 1 frame; FFT dihitung per kolom |
| `np.empty(..., dtype=np.complex64)` | wadah hasil; `complex64` karena output FFT kompleks |
| `order='F'` | data disimpan per kolom di memori (column-major) — cocok dengan cara kita mengisi per kolom, sedikit lebih cepat |
| `fft.fft(...)[:audio_fft.shape[0]]` | hitung FFT penuh (2048), ambil 1025 pertama saja (simetri) |
| `np.transpose` kedua | kembalikan ke `(frame, frekuensi)` |
| `np.square(np.abs(...))` | magnitude kuadrat = power |

> **Catatan teknis:** kode referensi menulis `audio_winT[1]` pada `np.empty` — itu bug (mengambil baris ke-1, bukan jumlah kolom). Yang benar `audio_winT.shape[1]`.
>
> **Alternatif yang bisa disebut:** seluruh loop di atas setara dengan satu baris `np.fft.rfft(audio_win, axis=1)` (rfft = FFT untuk sinyal real, otomatis hanya mengembalikan separuh + 1). Loop dipertahankan agar mahasiswa melihat prosesnya per frame.

## 8.4 Desibel

Plot spektrum dan spectrogram memakai $10\log_{10}(P + 10^{-10})$:
- `log10` supaya nilai sangat kecil dan sangat besar sama-sama terlihat (rentang dinamis suara sangat lebar).
- `+ 1e-10` mencegah `log(0) = -inf`.

## 8.5 Spectrogram

`plt.imshow(10*np.log10(audio_power.T + 1e-10), aspect='auto', origin='lower', cmap='inferno', extent=[0, durasi, 0, sample_rate/2])`

| Argumen | Arti |
|---|---|
| `.T` | transpose supaya frekuensi ke sumbu Y, waktu ke sumbu X |
| `origin='lower'` | 0 Hz di bawah |
| `extent=[...]` | label sumbu dalam detik & Hz, bukan index |
| `cmap='inferno'` | gelap = lemah, terang = kuat |

**Cara membaca:** garis **horizontal** bertingkat = harmonik nada yang ditahan; garis **vertikal** = bunyi tiba-tiba (ketukan/konsonan letup) yang energinya menyebar ke semua frekuensi. Energi terkonsentrasi di frekuensi rendah.

> **Poin ajar (jembatan ke Sesi 4):** spectrogram ini = 1257 × 1025 ≈ **1,3 juta angka**. Masih terlalu besar dan masih dalam skala Hz linear. Langkah 5 (Mel Filterbank) akan meringkas 1025 bin menjadi ~10–40 pita Mel; langkah 6 (DCT) meringkas lagi menjadi ~13 koefisien. Itulah MFCC.

---

# BAGIAN 9 — Preview Sesi 4 (untuk pengajar)

Supaya pengajar siap menjawab pertanyaan "terus habis ini apa?", berikut ringkasan langkah 5–6.

**5. Mel Filterbank** — kumpulan filter **segitiga** yang diletakkan merata di skala **Mel** (jadi rapat di frekuensi rendah, renggang di frekuensi tinggi). Tiap filter dikalikan dengan power spectrum dan dijumlahkan → energi per pita Mel.

Langkah pembuatannya: tentukan `f_min`, `f_max` (≤ `sample_rate/2`, aturan **Nyquist** — sample rate harus ≥ 2× frekuensi tertinggi untuk mencegah **aliasing**) → konversi ke Mel dengan `freq_to_mel` → bagi rata menjadi `n_mel + 2` titik → konversi balik dengan `mel_to_freq` → petakan ke index bin FFT → bangun segitiga.

$$\text{filterbank energy} = \text{audio\_power} \cdot \text{filters}^T \quad \to \quad (1257,\ n_{mel})$$

**6. DCT (Discrete Cosine Transform)** — ambil `log` dari energi filterbank, lalu DCT untuk memperoleh **cepstral coefficients**. Basis DCT-II:

$$C_{i,j} = \sqrt{\frac{2}{N}} \cos\left(i \times \frac{(2j+1)\pi}{2N}\right)$$

Hasil akhirnya matriks **MFCC** berukuran `(jumlah_frame, n_mfcc)`.

---

# Jawaban Latihan

**1. `HOP_SIZE` 10 dan 25**
- Hop 10 ms → 441 sample → `frame_num = 1887` frame.
- Hop 25 ms → 1102 sample → `frame_num = 755` frame.
- Hop lebih kecil → lompatan lebih pendek → lebih banyak frame (overlap lebih besar). `FFT_SIZE` tidak berubah sehingga kolom tetap 2048.

**2. `FFT_SIZE = 1024`**
- Durasi 1 frame = 1024 / 44100 ≈ **23,2 ms**.
- Bin frekuensi = 1 + 1024 // 2 = **513**.
- Resolusi frekuensi menjadi 44100/1024 ≈ 43 Hz per bin (lebih kasar), tetapi resolusi waktu lebih halus. Ini *trade-off* klasik time–frequency.

**3. Window `hamming` vs `boxcar`**
- `hamming` mirip `hann` (ujung tidak tepat 0, tapi ≈ 0,08).
- `boxcar` = tanpa windowing: spectrogram tampak lebih "berkabut"/smeared secara vertikal karena spectral leakage — energi bocor ke bin-bin tetangga.

**4. `AudioWav.wav`**
- Sample rate **24000 Hz**, durasi ≈ 58,6 s, juga stereo.
- Sumbu Y spectrogram maksimum menjadi 12000 Hz (Nyquist), bukan 22050 Hz.
- Durasi 1 frame jadi 2048/24000 ≈ 85 ms — lebih panjang dari rekomendasi 20–40 ms; diskusi bagus: `FFT_SIZE` sebaiknya disesuaikan dengan sample rate (misal 512 atau 1024).

**5. Function gabungan**
```python
def hitung_power_spectrum(path, FFT_SIZE=2048, HOP_SIZE=15):
    sample_rate, audio = wavfile.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio  = normalize_audio(audio)
    frames = frame_blocking(audio, FFT_SIZE, HOP_SIZE, sample_rate)
    frames = frames * get_window("hann", FFT_SIZE, fftbins=True)
    spec   = np.fft.rfft(frames, axis=1)          # setara loop FFT di Bagian 8
    return np.abs(spec) ** 2, sample_rate
```

---

# Catatan Pengajaran

**Error yang paling sering muncul di kelas:**

| Error | Penyebab | Solusi |
|---|---|---|
| `FileNotFoundError: audio.wav` | notebook dijalankan dari folder lain | pastikan `audio.wav` satu folder dengan notebook, atau ubah `TRAIN_PATH` |
| `TypeError: 'float' object cannot be interpreted as an integer` | `frame_num` tidak di-`int()` | bungkus dengan `int(...)` |
| `ValueError: operands could not be broadcast` | audio masih stereo `(N, 2)` saat `np.pad`/`* window` | konversi ke mono dulu (`mean(axis=1)`) |
| Durasi terhitung 2× lipat | memakai `flatten()` pada stereo | ganti dengan `mean(axis=1)` |
| `RuntimeWarning: divide by zero in log10` | `log10(0)` di plot | tambahkan `+ 1e-10` |
| Spectrogram polos/satu warna | lupa `10*np.log10` (plot power linear) | pakai skala dB |
| `NameError: window is not defined` | cell dijalankan tidak berurutan | *Restart & Run All* |

**Alur yang disarankan:**
1. Bagian 1–3 (teori) ± 35 menit, sambil membuka Tone Generator dan menjalankan eksperimen Mel di notebook — ini yang paling "hidup" untuk memancing diskusi.
2. Bagian 4–8 (kode) dijalankan cell per cell. Setiap selesai satu langkah, tulis **shape**-nya di papan tulis.
3. Setelah spectrogram muncul, tutup dengan tabel rangkuman Bagian 9 dan jembatan ke Sesi 4.

**Persiapan sebelum kelas:** buka notebook, `Kernel → Restart & Run All` untuk memastikan environment dan file audio siap, lalu `Restart & Clear Output` agar output muncul saat demo. Pastikan speaker/laptop bisa memutar audio (untuk `Audio()` dan Tone Generator).

**Menuju sesi berikutnya:** variable `audio_power`, `sample_rate`, `FFT_SIZE`, serta function `freq_to_mel` / `mel_to_freq` adalah input langsung untuk Mel Filterbank di Sesi 4. Simpan notebook ini; Sesi 4 melanjutkan dari cell terakhirnya.
