import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("C:/Users/oken72/Desktop/Dev/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FDAW0/SA1.WAV",
                     res_type='kaiser_best')

plt.figure(figsize=(20, 16))
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, hop_length=512, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectogram')

