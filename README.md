# 🎧 Urban Sound Classification & 🖼️ Image Caption Generation

## Part 1: Dataset and Its Structure

### ✅ UrbanSound8K Dataset
- Dataset Link: [UrbanSound8K](https://urbansounddataset.weebly.com/)
- Download Link: [Google Drive](https://goo.gl/8hY5ER) (~6GB)
  
### 📁 Structure after Extraction:
- **audio/**: Contains 10 folders (fold1, fold2, ..., fold10), each with ~800 audio files (4s each).
- **metadata/**: Contains `UrbanSound8K.csv` with columns:  
  `file_id`, `label`, `class_id`, `salience`, etc.

### 🧱 Audio Representation
- Each audio file is sampled at 22050 Hz → a 4s audio file will be an array of 88200 amplitude values.

---

## ⚡ Feature Extraction using Librosa
Install libraries:
```bash
pip install librosa
apt-get install ffmpeg
```


Use librosa.load(filepath) to load audio as amplitude array.

Extract features:

MFCCs

Spectral Centroid

Zero Crossing Rate

Energy, etc.

Experiment by combining different features to improve classification accuracy.

🎯 CNN for Sound Classification

Convert audio into spectrogram images → Similar sounds have similar visual patterns.

Train CNN model on spectrograms for classification.

🏋️ Model Training Pipeline
Example: MNIST Dataset

Download & extract MNIST dataset:

wget https://archive.org/download/mnist/mnist.zip
unzip mnist.zip -d mnist/


Load data:
```Python
import pandas as pd
import numpy as np
from keras.utils import to_categorical

df_train = pd.read_csv("mnist/train.csv")
X_train = df_train.iloc[:,1:].values
Y_train = to_categorical(df_train.iloc[:,0].values, num_classes=10)
```

Model creation:
``Python
from keras.models import Sequential
from keras.layers import Dense

def create_model():
    model = Sequential()
    model.add(Dense(30, input_dim=784, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

Save model after each epoch and upload to AWS S3.

🖼️ Part 2: Image Captioning
📚 Dataset Example

Flickr8k, Flick30K, MSCOCO.

Images + Multiple Captions.

✅ Training Sample Structure

Input: Image feature + Partial Sentence.

Output: Next word in sentence.

Special Tokens

startseq, endseq to indicate beginning & end of sentences.

🔧 Model Architecture

Image Encoder:
Pre-trained CNN (e.g., VGG16) used to extract features (4096-d vector per image).

Text Encoder:
Word Embedding + Sequence Input (max length sentence).

Decoder:
Combines encoded image & text features to predict next word.

⚡ Prediction Methods
Greedy Technique

At each step, pick word with max probability.

Beam Search Technique

Beam size (b):

In each iteration, keep top b candidates.

Select next top b words for each candidate sentence.

Continue until endseq.

Intuition: Beam search finds better overall sequences compared to greedy.

✅ Evaluation Metric: BLEU Score

Measures quality of generated captions.

Range: 0 (bad) to 1 (perfect).

Evaluates n-gram matches between predicted and reference captions.

Reference: Andrew NG YouTube Video on BLEU Score

🔗 Useful Resources

Librosa Audio Features

pyAudioAnalysis Feature Extraction

Audio Classification Guide

Image Captioning Paper

🚀 Conclusion

Use PCA for dimensionality reduction when needed.

Combine CNN + RNN architecture for Image Captioning.

Leverage Librosa + Spectrograms + CNN for Audio Classification.

Evaluate caption models using BLEU Score for better results.

📖 Happy Learning and Experimentation!
