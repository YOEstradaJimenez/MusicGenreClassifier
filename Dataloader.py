from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional 

class CustomAudioDataset(Dataset):
    def __init__(self, dataframe, transform, rate, duration):
        self.df = dataframe
        self.transform = transform
        self.rate = rate
        self.duration = duration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wave = torch.from_numpy(self.df.iloc[idx]['genres']).unsqueeze(0)
        samplerate = self.rate
        wave = self._resample_(wave, samplerate)
        wave = self._mixdown_(wave)
        wave = self._trim_(wave)
        wave = self._padding_(wave)
        wave = self.transform(wave)
        return wave
        
    def _resample_(self, wave, samplerate):
        if samplerate != self.rate:
            resampler = torchaudio.transforms.Resample(samplerate, self.rate)
            wave = resampler(wave)
        return wave

    def _mixdown_(self, wave):
        if wave.shape[0] > 1:
            wave = torch.mean(wave, dim = 0, keepdim = True)
        return wave
        
    def _trim_(self, wave):
        if wave.shape[1] > self.duration:
            wave = wave[:,:self.duration]
        return wave
    
    def _padding_(self, wave):
        if wave.shape[1] < self.duration:
            wave = torch.nn.functional.pad(wave, (0, self.duration - wave.shape[1]))
        return wave

# remenber sample_rate here *sample_rate == sample
spectogram = torchaudio.transforms.MelSpectrogram(sample_rate = 22050, n_fft = 1024, hop_length = 512, n_mels = 64)
