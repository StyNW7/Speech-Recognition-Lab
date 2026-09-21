import numpy as np 
from scipy.io import wavfile 
import scipy.fftpack as fft 
from scipy.signal import get_window
import matplotlib.pyplot as plt 

# numpy -> utk kalkulasi matematika 
# wavfile -> utk input audio dari file wav 
# fftpack -> utk melakukan fast fourier transform 
# get_window -> utk melakukan windowing 
# matplotlib.pyplot -> utk plotting hasil analisa MFCC 

# Train path -> utk menampung path dari file audio wav
TRAIN_PATH = './audio.wav'
# ambil sample rate dan audio dari wav file nya 
sample_rate, audio = wavfile.read(TRAIN_PATH)

# 1. NORMALIZATION -> atur amplitudo nya dari range -1 hingga 1 sepanjang video 
# kita akan membagi audio dengan nilai maximum absolute dari audio 
audio = audio / np.max(np.abs(audio)) 
# np.abs -> akan mencari nilai absolute dari audio (jadi yg tadinya negatif -> positif)
# np.max -> akan mencari nilai max dari nilai nilai tersebut 

# 2. FRAME BLOCKING -> membagi audio menjadi bagian bagian kecil (frame) yg saling overlap 
# overlap -> awal dari sebuah frame == akhir dari sebuah frame sebelumnya (vice versa)

# a. Define beberapa variable : fft size, hop size

# cari fft size -> ukurannya bebas, dan kt bs mencoba angka angka utk mendapatkan hasil yg sesuai yg kt inginkan 
# hop size -> waktu antar setiap start dari audio frame (arbitery / bebas, bisa selain 15)

FFT_SIZE = 2048 
HOP_SIZE = 15 

# b. padding (append bagian awal dan akhir audio dengan data data audio itu sendiri)
# 1. apa yg akan di pad 
# 2. lebar dari padding nya (1/2 dari FFT size, bebas)
# 3. mode (reflect) -> agar nilai nilai yg dimasukkan di awal dan akhir, merupakan nilai dari audio itu sendiri 
# reflect ini sifat nya kaya cermin, jadi dia akan mencerminkan / mereflect idx ke 0 utk padding di awal audio dan idx terakhir utk padding di akhir audio 
audio = np.pad(audio, int(FFT_SIZE / 2), mode='reflect')
# np.pad -> akan mereturn audio yg sudah di pad, makanya disimpan lagi di variable audio  

# c. hitung panjang dari setiap frame 
frame_len = np.round(sample_rate * HOP_SIZE / 1000).astype(int)
# dibagi 1000 supaya hasilnya jadi detik
# di pakein function np.round() supaya hasilnya bulat 
# di typecast jadi integer 

# hitung jumlah frame dari audio yg ada 
# 1. bisa cari dengan cara : len(audio) / framelen 
# nah, tapi ada kemungkinan dimana frame terakir panjangnya itu ga sampai 1 frame
# maka kt perlu mengurangi panjang dari audio dengan 1 FFT_SIZE utk meperhitungkan adanya sample yg hilang dari proses FFT 
frame_num = ((len(audio) - FFT_SIZE) / frame_len) + 1
# ditambahin 1 utk memperhitungkan adanya frame pertama di awal audio   

# buat kerangka dari array framesnya (frame yg akan disimpan dari hasi pembagian audio) -> bentuknya array 2d
# pake np.zeros() -> akan menciptakan 2d array
frames = np.zeros((frame_num, FFT_SIZE))
# akan membuat array dengan isi angka 0, dengan ukuran frame num x fft size 
# dibikin pake angka 0 -> utk memastikan bahwa entry dalam array yg tidak berisi audio akan bernilai 0 

# isi frames nya dengan masing masing bagian dari audio 
# masing masing entry harus memiliki data sebesar FFT size -> krn itu merupakan ukuran dari frame nya 
for i in range(frame_num) : 
    frames[i] = audio[i * frame_len : i * frame_len + FFT_SIZE]
    # utk memastikan agar yg di slice panjangnya sma dengan FFT SIZE 

# 3. WINDOWING -> membuat window function pada setiap frame 

# window function -> function yg akan membuat ujung awal dan akhir dari sinyal kita dekat dengan 0, supaya tidak ada lompatan nilai yg akan menimbulkan distorsi

# kalo kt buat awal dan akhir nilai sinyal nya mendekati 0, nanti perpindahan antara 1 frame ke frame lainnya itu akan lebih smooth 

# skrg, audio kt ada di time domain (terhadap waktu). kt mau mengubahnya ke frequency domain (terhadap frekuensi). makanya kt harus pake FFT 

# tp, utk melakukan FFT, audio yg diberikan harus periodic dan continuous. Tadi kt ud melakukan framing utk membuat audio kt periodic, nah skrg kt akan membuatnya menjadi continuous menggunakan window function pada setiap frame 

# Macam macam tipe window -> barlett, blackman, flat top hanning, hamming, welch 

# Window yg digunakan -> Hanning 
# 1. pake function get_window yg diimport dari scipy.signal 
# parameter 1 : jenis window (hann)
# parameter 2 : panjang dari window nya (disamain dengan FFT size agar window function mengubah seluruh frame nya (1 Frame size == FFT size), kalo diisi FFT SIZE / 2 nanti cma diubah setengah frame nya aja)
# parameter 3 : fftbins = True (kt mau window function nya ter normalisasi agar hasil window function = hasil fft bins -> energy dari sinyal tetap sama setelah window function di jalankan)
window = get_window("hann", FFT_SIZE, fftbins=True)

# audio win akan menampung audio yg sudah di windowing
audio_win = frames * window 

# 4. Fast Fourier Transform (FFT) -> mengubah audio dari time domain -> frequency domain 
# 1. transpose audio -> agar freq digambarin di baris dan  frame digambarin di kolom 
audio_winT = np.transpose(audio_win)

# 2. buat array kosong (bernilai audio yg sudah di FFT)
# dibuat dengan function np.empty -> menerima argumen shape / dimensi dari 2 dimensional array 
# dimensi pertama -> 1 + FFT_SIZE// 2 -> nah jadi output dari fft itu akan simetris, krn dia simetris kt ga perlu simpen nilai nilai yg negatif, kt cma perlu simpan nilai 0 - setengah dari FFT size utk nilai positif saja
# dimensi kedua -> ukuran audio window yg udah di transpose 
# krn kt mau np.empty bisa menyimpan data bilangan kompleks 64 bit -> dtype = np.complex64
# order = 'f' -> agar nilai di arry disimpan dalam memory scr column order krn komputasi akan dilakukan scr column order (biar lebih cpt aja komputasinya)
audio_fft = np.empty((1 + FFT_SIZE // 2, audio_winT[1]), dtype=np.complex64, order='F')

# algoritma FFT 
for i in range(audio_fft.shape[1]) : 
    audio_fft[:, i] = fft.fft(audio_winT[:, i], axis=0)[:audio_fft.shape[0]]

# transpose kembali -> freq di column, frame di row 
audio_fft = np.transpose(audio_fft)
audio_power = np.square(np.abs(audio_fft))

import matplotlib.pyplot as plt 

plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(audio_power.T + 1e-10), aspect='auto', origin='lower', cmap='inferno', extent=[0, len(audio)/sample_rate, 0, sample_rate/2])
plt.colorbar(label='Power (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram (FFT Result)')
plt.show()
