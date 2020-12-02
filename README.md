# Lip-Reading
This repository contains the codes for lip reading using **3D cross audio-visual** Convolutional Neural Networks. 
Link to the report : [ref](https://meghbhalerao.github.io/pdfs/Megh-Bhalerao-Lip-Reading-Report.pdf)
## Steps to run the code
1. To extract the lip region (bounding box) using [Histogram of Oriented Gradients](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf): `cd Visual_Preprocessing`. Then run `python mouth_cropping_in_video.py` for getting the crops of the mouth region from the video.
2. To run the audio preprocessing: `cd Audio_Preperocessing`. Then run the file: `matlab MMSESTSA84.m`, which performs the audio preprocessing using the [MMSE STSA](https://www.researchgate.net/publication/321785229_Automatic_and_Efficient_Denoising_of_Bioacoustics_Recordings_Using_MMSE_STSA) method. Another Audio Preprocessing, Voice Activity Detection, which is an energy based method is also supported, which can be run using `python unsupervised_vad.py`. 

## Training the CNN Model
1. To train the CNN model, run `python train.py`, with the appropriate paths to the audio and video files.

## References
1. [3D Convolutional Neural Networks for Cross Audio-Visual Matching Recognition](https://ieeexplore.ieee.org/document/8063416). Amirsina Torfi, Seyed Mehdi Iranmanesh, Nasser Nasrabadi, Jeremy Dawson et al. *IEEE Access, Volume 5.*

