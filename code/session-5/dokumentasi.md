# Sesi 5 — Wav2vec: Introduction to wav2vec & Create wav2vec using Torchaudio
### Dokumentasi Penjelasan Code & Materi | Speech Recognition Lab, BINUS University

Dokumen ini adalah pendamping untuk file di folder `code/session-5/`. Isinya teori wav2vec, rumus yang dipakai, penjelasan tiap bagian kode, dan hal yang perlu ditekankan saat mengajar.

**Cakupan sesi (sesuai Course Outline):**
- [x] Introduction to wav2vec
- [x] Create wav2vec using Torchaudio — memakai **model pretrained** wav2vec 2.0 dari `torchaudio.pipelines`

---

## Isi Folder

| File | Fungsi |
|---|---|
| `Session 5 - Wav2vec.ipynb` | notebook utama untuk mengajar (sudah berisi output) |
| `wav2vec.py` | versi script ringkas dari notebook — cukup `python wav2vec.py` |
| `speech.wav` | contoh audio bahasa Inggris, 3,4 detik, 16 kHz mono (dataset VOiCES, dipakai juga di tutorial resmi Torchaudio) |
| `dokumentasi.md` | dokumen ini |

**Transkrip yang benar dari `speech.wav`:** `I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT`

## Struktur Notebook

| Bagian | Topik | Estimasi |
|---|---|---|
| 1 | Apa itu wav2vec? Dari MFCC ke *learned features* | 20 menit |
| 2 | Setup: import library & membaca file audio | 10 menit |
| 3 | Memuat model pretrained dari Torchaudio + resample | 15 menit |
| 4 | Ekstraksi fitur wav2vec (12 layer Transformer) | 15 menit |
| 5 | Emission: skor karakter per frame | 10 menit |
| 6 | CTC Greedy Decoder: emission → teks | 20 menit |
| 7 | Fungsi `transcribe()` untuk audio sendiri | 10 menit |
| 8 | Rangkuman & latihan | 10 menit |

## Prasyarat

```bash
pip install torch torchaudio soundfile matplotlib
```

- Versi `torchaudio` **harus sama** dengan `torch` (contoh: torch 2.10.0 ↔ torchaudio 2.10.0). Sudah dites dengan Python 3.12, torch/torchaudio 2.10.0 (CPU).
- **Butuh internet saat pertama kali** — bobot model ±360 MB di-download otomatis ke cache (`~/.cache/torch/hub/checkpoints/`). Sebaiknya download sebelum kelas supaya tidak menunggu.
- Tidak butuh GPU. Di CPU, inferensi audio 3,4 detik hanya butuh < 1 detik.

---

# BAGIAN 1 — Teori: Apa itu wav2vec?

## 1.1 Dari MFCC ke Learned Features

Di Sesi 3–4 kita membuat MFCC: **Normalization → Frame Blocking → Windowing → FFT → Mel Filterbank → DCT**. Setiap langkah dirancang manusia berdasarkan pengetahuan tentang sinyal dan pendengaran manusia. Fitur seperti ini disebut **hand-crafted features**.

Kelemahannya:
1. Hanya melihat **satu frame** (±25 ms) — tidak tahu konteks kata sebelum/sesudahnya.
2. Informasi yang dibuang sudah ditentukan dari awal (misal hanya ambil 13 koefisien), padahal belum tentu itu yang terbaik untuk tugas kita.
3. Untuk bisa mengenali kata, MFCC masih butuh model lain (HMM, RNN, dll.) yang dilatih dengan **banyak data berlabel** (audio + transkrip). Data berlabel itu **mahal** — harus ditranskripsikan manusia.

**wav2vec** menjawab ketiganya: biarkan **neural network belajar sendiri** representasi suara langsung dari **raw waveform**, dan belajarnya dari audio **tanpa label** yang jumlahnya melimpah.

> **Poin ajar:** analogikan dengan anak kecil. Anak mendengar ribuan jam orang berbicara sebelum belajar membaca. Saat belajar membaca, ia cukup diajari sedikit contoh huruf. wav2vec meniru proses itu: *dengar dulu banyak audio (pre-training) → baru diajari sedikit transkrip (fine-tuning)*.

## 1.2 Sejarah Singkat

| Tahun | Model | Ide utama |
|---|---|---|
| 2019 | **wav2vec** (Schneider dkk., Facebook AI) | CNN di atas raw audio, dilatih menebak frame **masa depan** (*contrastive*) tanpa label |
| 2019 | **vq-wav2vec** | menambahkan **quantization** — fitur kontinu diubah jadi "unit suara" diskrit |
| 2020 | **wav2vec 2.0** (Baevski dkk., NeurIPS 2020) | CNN + **Transformer** + **masking** (mirip BERT pada teks). Ini yang dipakai sekarang |
| 2020+ | XLSR, HuBERT, WavLM | turunan wav2vec 2.0: multibahasa, target training berbeda, dll. |

Yang kita pakai di kelas adalah **wav2vec 2.0**. Di Torchaudio namanya `WAV2VEC2_*`.

**Hasil yang membuat wav2vec 2.0 terkenal:** dengan hanya **10 menit** data berlabel (setelah pre-training 53 ribu jam audio tanpa label), model sudah mencapai WER 4,8% / 8,2% di LibriSpeech test-clean / test-other. Sebelumnya, angka seperti itu butuh ratusan jam data berlabel.

## 1.3 Arsitektur wav2vec 2.0

```
 raw waveform x  (16 kHz)
        │
        ▼
 ┌──────────────────────────┐
 │  CNN Feature Encoder     │  7 layer konvolusi 1D, 512 channel
 │  (f : X → Z)             │  1 vektor z_t setiap 20 ms
 └──────────────────────────┘
        │  z_1 … z_T  (latent speech representation)
        ├──────────────────────────────┐
        ▼  (sebagian di-MASK)          ▼
 ┌──────────────────────────┐   ┌─────────────────────┐
 │  Transformer             │   │  Quantization Module│
 │  (g : Z → C)             │   │  (Z → Q)            │
 │  12 layer (BASE)         │   │  hanya dipakai saat │
 └──────────────────────────┘   │  pre-training       │
        │  c_1 … c_T            └─────────────────────┘
        ▼  (context representation)      │ q_t (target)
   contrastive loss  ◄───────────────────┘
```

### a. CNN Feature Encoder — "pengganti framing + FFT"

7 blok konvolusi 1D (tiap blok: Conv1D → LayerNorm/GroupNorm → GELU).

| Layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Kernel | 10 | 3 | 3 | 3 | 3 | 2 | 2 |
| Stride | 5 | 2 | 2 | 2 | 2 | 2 | 2 |

- **Total stride** = 5 × 2⁶ = **320 sample** → di 16 kHz = **20 ms** per vektor (= *hop size*).
- **Receptive field** = **400 sample** = **25 ms** (= *frame length*).

Angka 25 ms frame dan 20 ms hop ini mirip sekali dengan pengaturan framing MFCC. Bedanya, filter konvolusinya **dipelajari**, bukan rumus FFT/Mel yang tetap.

**Rumus panjang output konvolusi** (dipakai untuk menghitung jumlah frame):

$$L_{out} = \left\lfloor \frac{L_{in} - \text{kernel}}{\text{stride}} \right\rfloor + 1$$

Contoh untuk `speech.wav` (54.400 sample):

```
54400 → 10879 → 5439 → 2719 → 1359 → 679 → 339 → 169 frame
```

Hasilnya **169 frame** — persis shape `(1, 169, 768)` yang muncul di notebook. Secara kasar: 3,4 detik × 50 frame/detik ≈ 170.

### b. Transformer (Context Network)

- Menerima z_1 … z_T dan menghasilkan **c_1 … c_T**.
- Dengan **self-attention**, setiap frame bisa "melihat" **seluruh kalimat**. Inilah keunggulan dibanding MFCC yang hanya melihat 1 frame.
- Posisi waktu diberikan lewat konvolusi (*relative positional embedding*), bukan sinusoidal.

| Ukuran | Layer | Dimensi | Attention heads | Parameter |
|---|---|---|---|---|
| BASE | 12 | 768 | 8 | ±95 juta |
| LARGE | 24 | 1024 | 16 | ±317 juta |

### c. Quantization Module (hanya saat pre-training)

Mengubah z_t (kontinu) menjadi **q_t** (diskrit) — semacam "kamus unit suara".
- Memakai **G = 2 codebook**, masing-masing **V = 320 entri** → 320 × 320 = 102.400 kombinasi unit suara.
- Pilihan entri memakai **Gumbel-Softmax** supaya tetap bisa di-*backpropagate*.
- q_t dipakai sebagai **jawaban/target** saat pre-training.

## 1.4 Pre-training: Self-Supervised dengan Masking

Mirip BERT di NLP ("isi bagian yang dihilangkan"):

1. Pilih acak titik awal mask (probabilitas **p = 0,065**), lalu tutup **M = 10** frame berturut-turut dari titik itu → ±**49%** frame tertutup.
2. Frame yang tertutup diganti vektor khusus (*mask embedding*) sebelum masuk Transformer.
3. Untuk setiap frame tertutup t, model harus bisa **memilih q_t yang benar** di antara q_t asli + **K = 100 distractor** (q dari frame lain di kalimat yang sama).

**Contrastive loss:**

$$\mathcal{L}_m = -\log \frac{\exp\left(\text{sim}(c_t, q_t)/\kappa\right)}{\sum_{\tilde{q} \in Q_t} \exp\left(\text{sim}(c_t, \tilde{q})/\kappa\right)}$$

| Simbol | Arti |
|---|---|
| c_t | output Transformer di frame t (yang di-mask) |
| q_t | target terkuantisasi yang benar untuk frame t |
| Q_t | himpunan kandidat: q_t + K distractor |
| sim(a, b) | *cosine similarity* = aᵀb / (‖a‖‖b‖) |
| κ | temperatur (0,1) |

Artinya: buat c_t **mirip** dengan q_t yang benar, dan **tidak mirip** dengan distractor. Mirip soal pilihan ganda — model harus memilih jawaban benar dari 101 pilihan.

**Total loss pre-training:**

$$\mathcal{L} = \mathcal{L}_m + \alpha \mathcal{L}_d$$

$\mathcal{L}_d$ = *diversity loss*, memaksa semua entri codebook terpakai merata (supaya model tidak hanya memakai sedikit "unit suara"). α = 0,1.

> **Poin ajar:** tekankan bahwa **tidak ada transkrip sama sekali** di tahap ini. Labelnya dibuat dari audio itu sendiri — itulah arti *self-supervised*.

## 1.5 Fine-tuning dengan CTC

Setelah pre-training, model sudah "paham" suara tetapi belum bisa menulis huruf. Untuk ASR:

1. Buang quantization module.
2. Tambahkan **1 layer linear** di atas Transformer: 768 → **29 label** (`-`, `|`, `'`, A–Z).
3. Latih dengan data berlabel memakai **CTC loss**.

Model yang kita pakai, `WAV2VEC2_ASR_BASE_960H`, sudah melalui kedua tahap: pre-training di 960 jam LibriSpeech (tanpa label) lalu fine-tuning dengan 960 jam transkrip LibriSpeech.

## 1.6 CTC (Connectionist Temporal Classification)

**Masalah:** model menghasilkan 1 prediksi per 20 ms (169 frame), padahal kalimatnya hanya 45 karakter. Kita tidak tahu frame mana milik huruf apa (tidak ada *alignment*).

**Solusi CTC:** tambahkan token **blank** (`-`) dan definisikan fungsi **B** yang mengubah urutan per-frame (*path*) menjadi teks:
1. gabungkan label berulang yang berurutan,
2. hapus semua blank.

```
path  : H H - E E - L L - - L - O O
B(path): H E L L O
```

Blank di antara dua `L` membuat huruf dobel tetap terjaga. Tanpa blank, `LL` akan tergabung menjadi `L`.

**Probabilitas teks** = jumlah probabilitas **semua path** yang menghasilkan teks itu:

$$P(y \mid x) = \sum_{\pi \in B^{-1}(y)} \prod_{t=1}^{T} p(\pi_t \mid x)$$

$$\mathcal{L}_{CTC} = -\log P(y \mid x)$$

**Saat inferensi (yang kita lakukan di kelas)** tidak perlu menjumlah semua path. Cukup **greedy decoding**: ambil label terbaik di tiap frame, lalu terapkan B.

$$\hat{y} = B\left(\arg\max_{k} \; p(k \mid x, t) \;\; \text{untuk } t = 1 \dots T\right)$$

(Alternatif yang lebih akurat: *beam search* + *language model*, di luar cakupan sesi ini.)

## 1.7 MFCC vs wav2vec 2.0

| | MFCC (Sesi 3–4) | wav2vec 2.0 (Sesi 5) |
|---|---|---|
| Jenis fitur | hand-crafted | learned (dipelajari) |
| Input | raw audio | raw audio |
| Ukuran per frame | 13 koefisien | 768 dimensi (BASE) |
| Frame / hop | ±25 ms / ±10–20 ms | 25 ms / 20 ms |
| Konteks | 1 frame | seluruh kalimat (Transformer) |
| Butuh data berlabel? | banyak (untuk model setelahnya) | sedikit (cukup untuk fine-tuning) |
| Biaya komputasi | sangat ringan | berat (±95 juta parameter) |
| Bisa langsung output teks? | tidak | ya (versi ASR + CTC) |

MFCC tetap berguna untuk perangkat kecil (*embedded*, mikrokontroler) atau dataset kecil yang tidak butuh model besar.

---

# BAGIAN 2 — Penjelasan Code

Alur keseluruhan:

```
speech.wav → sf.read → tensor (1, samples) → resample 16 kHz → model
          → features (12 × (1, 169, 768))       ← "fitur wav2vec"
          → emission (1, 169, 29)                ← skor tiap label
          → GreedyCTCDecoder → "I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT"
```

## 2.1 Import & device

```python
import torch, torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

| Library | Dipakai untuk |
|---|---|
| `torch` | tensor, `inference_mode`, `argmax` |
| `torchaudio` | model pretrained (`pipelines`) + `functional.resample` |
| `soundfile` | membaca `.wav` |
| `matplotlib` | plot waveform, fitur, emission |
| `IPython.display.Audio` | memutar audio di notebook |

## 2.2 Membaca audio

```python
audio, sample_rate = sf.read(AUDIO_PATH, dtype="float32")
waveform = torch.from_numpy(audio).unsqueeze(0)   # (54400,) -> (1, 54400)
```

- `dtype="float32"` → nilai otomatis dalam rentang **-1 … 1** (setara langkah *normalization* di MFCC) dan tipe data sesuai model.
- `unsqueeze(0)` menambah dimensi **batch** karena model menerima input `(batch, samples)`.
- **Kenapa tidak `torchaudio.load()`?** Sejak torchaudio 2.9, fungsi `load()` dipindahkan ke library `torchcodec` dan akan error `ImportError: TorchCodec is required...` kalau `torchcodec` belum di-install. `soundfile` lebih sederhana dan pasti jalan.

## 2.3 Memuat bundle & model

```python
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
model.eval()
```

**Bundle** = paket berisi bobot model + konfigurasi:

| Atribut / method | Isi |
|---|---|
| `bundle.sample_rate` | `16000` — sample rate yang diharapkan model |
| `bundle.get_labels()` | `('-', '|', 'E', 'T', 'A', ...)` — 29 label output |
| `bundle.get_model()` | objek `Wav2Vec2Model` dengan bobot pretrained |

`model.eval()` mematikan *dropout* dsb. sehingga hasil prediksi selalu sama (deterministik).

**Bundle wav2vec lain yang tersedia di Torchaudio:**

| Bundle | Keterangan |
|---|---|
| `WAV2VEC2_BASE`, `WAV2VEC2_LARGE`, `WAV2VEC2_LARGE_LV60K` | hanya pre-training → untuk **ekstraksi fitur**, tidak bisa output teks |
| `WAV2VEC2_ASR_BASE_10M` / `_100H` / `_960H` | BASE + fine-tune 10 menit / 100 jam / 960 jam (Inggris) |
| `WAV2VEC2_ASR_LARGE_*`, `WAV2VEC2_ASR_LARGE_LV60K_*` | versi LARGE, lebih akurat tapi lebih berat |
| `WAV2VEC2_XLSR53`, `WAV2VEC2_XLSR_300M/1B/2B` | pre-training **multibahasa** (termasuk bahasa Indonesia) → hanya fitur, perlu fine-tune sendiri untuk ASR |
| `VOXPOPULI_ASR_BASE_10K_DE/EN/ES/FR/IT` | ASR bahasa Jerman/Inggris/Spanyol/Prancis/Italia |
| `HUBERT_*`, `WAVLM_*` | turunan wav2vec 2.0 dengan cara pre-training berbeda |

Semua bundle dipakai dengan cara yang sama — cukup ganti nama bundle.

## 2.4 Resample

```python
if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
```

Model hanya dilatih dengan audio 16 kHz. Kalau diberi audio 44,1 kHz tanpa resample, 1 detik audio akan "terdengar" seperti 2,76 detik bagi model (suara jadi sangat lambat dan berat) → transkrip kacau. `speech.wav` sudah 16 kHz sehingga blok ini dilewati, tetapi tetap penting untuk audio rekaman sendiri.

## 2.5 Ekstraksi fitur

```python
with torch.inference_mode():
    features, _ = model.extract_features(waveform)
```

- `torch.inference_mode()` → tidak menyimpan gradient, lebih cepat & hemat memori (kita tidak training).
- `features` = **list berisi 12 tensor** (1 per layer Transformer), masing-masing `(1, 169, 768)` = `(batch, frame, dimensi)`.
- Nilai kedua (`_`) adalah panjang valid tiap audio dalam batch — tidak dipakai karena batch kita hanya 1.

**Visualisasi:** plot `imshow` per layer. Skala warna dibatasi pada persentil 98% (`quantile(0.98)`) karena ada beberapa dimensi dengan nilai sangat besar (outlier) yang membuat sisa gambar pudar. Di plot terlihat frame 0–30 (bagian hening) polanya berbeda dari bagian bersuara.

**Penggunaan fitur:** fitur ini bisa menggantikan MFCC sebagai input model lain, misalnya klasifikasi emosi, identifikasi pembicara, atau deteksi kata kunci. Untuk mendapat **1 vektor per audio**, rata-ratakan terhadap waktu: `features[-1][0].mean(dim=0)` → shape `(768,)`.

## 2.6 Emission

```python
with torch.inference_mode():
    emission, _ = model(waveform)       # (1, 169, 29)
```

- Memanggil `model(...)` = feature encoder → Transformer → layer linear ASR.
- `emission[0, t, k]` = skor (logit) bahwa frame ke-t adalah label ke-k. Belum di-softmax, tetapi untuk `argmax` tidak masalah karena urutannya sama.
- Di heatmap, baris `-` (blank) paling sering terang — sebagian besar frame berisi "tidak ada karakter baru".

## 2.7 GreedyCTCDecoder

```python
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission):
        indices = torch.argmax(emission, dim=-1)           # 1
        indices = torch.unique_consecutive(indices)        # 2
        indices = [i for i in indices if i != self.blank]  # 3
        text = "".join(self.labels[i] for i in indices)
        return text.replace("|", " ").strip()              # 4
```

| Langkah | Kode | Fungsi |
|---|---|---|
| 1 | `argmax(dim=-1)` | indeks label dengan skor tertinggi di tiap frame |
| 2 | `unique_consecutive` | gabungkan label yang **berurutan** sama (`TT` → `T`); beda dengan `unique` biasa yang menghapus semua duplikat |
| 3 | filter `!= blank` | buang `-` (indeks 0) |
| 4 | `replace("|", " ")` | pemisah kata → spasi |

Output nyata dari notebook (frame 30–80):

```
1. Argmax per frame : I--||-H-A-D-||TTH-ATT||--C----U-RR--I------OO---S-
2. Gabung berulang  : I-|-H-A-D-|TH-AT|-C-U-R-I-O-S-
3. Buang blank      : I|HAD|THAT|CURIOS
4. '|' -> spasi     : I HAD THAT CURIOS
```

Hasil penuh: **`I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT`** ✓

> **Poin ajar:** minta mahasiswa men-*decode* baris "Argmax per frame" secara manual di papan tulis sebelum menjalankan cell berikutnya.

## 2.8 Fungsi `transcribe()`

Menggabungkan semua langkah: baca → stereo ke mono (`mean(dim=1)`) → resample → model → decoder. Dipakai untuk file audio lain:

```python
print(transcribe("./rekaman_saya.wav"))
```

`wav2vec.py` berisi alur yang sama dalam bentuk script:

```bash
python wav2vec.py                 # transkripsi speech.wav
python wav2vec.py rekaman.wav     # transkripsi file lain
```

**Batasan:**
- Model hanya mengerti **bahasa Inggris**. Bahasa Indonesia akan ditulis dengan "ejaan Inggris" yang tidak bermakna.
- Format `.wav`/`.flac`/`.ogg` didukung `soundfile`; `.mp3`/`.m4a` sebaiknya dikonversi dulu ke `.wav`.
- Gunakan audio pendek (≤ 30 detik). Audio panjang memakan memori besar karena self-attention Transformer; potong menjadi beberapa bagian.

---

# BAGIAN 3 — Latihan & Kunci Jawaban

| No | Latihan | Kunci / hasil yang diharapkan |
|---|---|---|
| 1 | Ganti bundle ke `WAV2VEC2_ASR_BASE_10M`, bandingkan dengan `960H` | 10M lebih sering salah eja karena data fine-tuning hanya 10 menit — tetapi tetap mengejutkan bagusnya untuk data sesedikit itu. Hubungkan dengan hasil paper (bagian 1.2) |
| 2 | Rekam suara sendiri (bahasa Inggris), transkripsikan | Pastikan resample berjalan (rekaman HP biasanya 44,1/48 kHz). Hasil dipengaruhi aksen, noise, dan jarak mikrofon |
| 3 | Tambahkan noise `level` 0,02 / 0,05 / 0,1 | Hasil uji (`torch.manual_seed(0)`): 0,02 dan 0,05 → transkrip masih sempurna; 0,1 → `I HAD THAT CURIOFITY DEFIED ME AT THE FALL`. wav2vec cukup tahan noise ringan, tapi rusak di noise berat |
| 4 | `features[-1][0].mean(dim=0)` — shape & kegunaan | Shape `(768,)`: 1 vektor ringkasan untuk seluruh audio → input untuk klasifikasi (emosi, speaker, bahasa), atau dibandingkan dengan cosine similarity |

---

# BAGIAN 4 — Troubleshooting

| Error | Penyebab | Solusi |
|---|---|---|
| `ImportError: TorchCodec is required for load_with_torchcodec` | memakai `torchaudio.load()` di torchaudio ≥ 2.9 | pakai `soundfile` seperti di kode ini |
| `OSError: [WinError 1114] ... c10.dll` hanya di Jupyter (di script biasa lancar) | `pyzmq` versi lama (25.x) membawa `msvcp140.dll` lama yang bentrok dengan torch | `pip install --upgrade pyzmq`, lalu restart kernel |
| `OSError` / `undefined symbol` saat `import torchaudio` | versi torch ≠ torchaudio | install pasangan versi yang sama, mis. `pip install torchaudio==<versi torch>` |
| Download model gagal / sangat lama | koneksi lab | download sebelum kelas (jalankan `bundle.get_model()` sekali), file tersimpan di `~/.cache/torch/hub/checkpoints/` |
| Transkrip kacau total | sample rate bukan 16 kHz tanpa resample, atau audio bukan bahasa Inggris | cek `sample_rate` & bahasa audio |

---

## Referensi

- Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. NeurIPS 2020. arXiv:2006.11477
- Schneider, S., Baevski, A., Collobert, R., & Auli, M. (2019). *wav2vec: Unsupervised Pre-training for Speech Recognition*. arXiv:1904.05862
- Graves, A. dkk. (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. ICML 2006.
- Dokumentasi Torchaudio — *Speech Recognition with Wav2Vec2* dan `torchaudio.pipelines`.
