# Lip-Reading
This repository contains the codes for lip reading using **3D cross audio-visual** Convolutional Neural Networks.\
Link to our project report : [ref](https://meghbhalerao.github.io/pdfs/Megh-Bhalerao-Lip-Reading-Report.pdf)

## Brief Description of the project
In this small project, we tried to re-engineer [1], by using similar network architecture, but using our **own data** and different video and audio preprocessing techniques, as described below. Due to **large computational** requirements for Audio and Visual Preprocessing, we trained the model on a dummy dataset, with random placeholders for the data, instead of actual intensity values. 

## Steps to run the code
### Audio and Video Preprocessing
1. Download either [VidTimit](https://conradsanderson.id.au/vidtimit/) or [the BBC Lip Reading in the Wild](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) datasets and place them in `./dataset/` folder
1. To extract the lip region (bounding box) using [Histogram of Oriented Gradients](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf): `cd Visual_Preprocessing`. Then run `python mouth_cropping_in_video.py` for getting the crops of the mouth region from the video.
2. To run the audio preprocessing: `cd Audio_Preperocessing`. Then run the file: `matlab MMSESTSA84.m`, which performs the audio preprocessing using the [MMSE STSA](https://www.researchgate.net/publication/321785229_Automatic_and_Efficient_Denoising_of_Bioacoustics_Recordings_Using_MMSE_STSA) method. Another Audio Preprocessing, Voice Activity Detection, which is an energy based method is also supported, which can be run using `python unsupervised_vad.py`. 

### Training the CNN Model
1. To train the CNN model, run `python train.py`, with the appropriate paths to the audio and video files.

## Dependencies
- [`tensorflow`](https://www.tensorflow.org)
- [`pydub`](https://pypi.org/project/pydub/)
- [`matplotlib`](https://matplotlib.org)
## References
1. [3D Convolutional Neural Networks for Cross Audio-Visual Matching Recognition](https://ieeexplore.ieee.org/document/8063416). Amirsina Torfi, Seyed Mehdi Iranmanesh, Nasser Nasrabadi, Jeremy Dawson et al. *IEEE Access, Volume 5.*

