from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as sp

sound_file = AudioSegment.from_wav("xyz.wav")
audio_chunks = split_on_silence(sound_file,
    # must be silent for at least half a second
    min_silence_len=20,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-40,

    keep_silence= 50
)

print(type(sound_file))

f,d = sp.read("xyz.wav")
plt.plot(d)
plt.show()
for i, chunk in enumerate(audio_chunks):

    out_file = "chunk{0}.wav".format(i)
    print("exporting", out_file)
    chunk.export(out_file, format="wav")

print(audio_chunks)