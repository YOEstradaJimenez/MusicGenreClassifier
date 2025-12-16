# MusicGenreClassifier
This project provides a script for classifying music genres from .mp3 files using a trained PyTorch neural network. It loads audio files, preprocesses them into spectrograms, and evaluates them with a custom model to predict genres.

## Supported Genres
The classifier currently supports 8 genres:
| ID | Genre       |
|----|-------------|
| 0  | Classical   |
| 1  | Country     |
| 2  | Electronic  |
| 3  | Metal       |
| 4  | Jazz        |
| 5  | Pop         |
| 6  | Hiphop      |
| 7  | Rock        |

## How It Works
- Audio Loading: Uses librosa to load .mp3 files at 22,050 Hz.
- Segmentation: Splits each track into 5‑second chunks.
- Dataset: Wraps segments into CustomAudioDataset with spectrogram transformation.
- Model: Loads pretrained weights (wmusic.pth) into NeuralNet.
- Evaluation: Runs inference with torch.no_grad() and applies sigmoid activation.
- Aggregation: Averages predictions per track and selects the genre with the highest probability.

## Project Structure
MusicGenreClassifier/
│

├── Files/                # Folder containing .mp3 files to classify

├── Dataloader.py         # Custom dataset and spectrogram utilities

├── AudioNet.py           # Neural network architecture

└── wmusic.pth            # Trained PyTorch model weights

