import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import scipy.signal 

from decoder import GreedyDecoder
from model import DeepSpeech

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_mfcc(audio, lengths):
    '''
    Compute MFCC features of input audio,
    by using the method in deepspeech.pytorch.
    In Pytorch, we can differentiate through it.
    '''
    # define the specific set 
    sample_rate = 16000
    window_size = 0.02 
    window_stride = 0.01 
    # window and FFT
    window = scipy.signal.hamming(int(sample_rate * window_size))
    n_fft = int(sample_rate * window_size)
    win_length = n_fft 
    hop_length = int(sample_rate * window_stride)
    #STFT and log(1 + spect), normalize
    window = torch.FloatTensor(window).cuda()
    max_audio_len = int(audio[0].size(0) / 160) + 1
    spects = torch.zeros((audio.size(0), 1, 161, max_audio_len)).cuda()  ### 161 is special
    spects_lengths = torch.IntTensor(audio.size(0))
    for i in range(audio.size(0)):
        stft = torch.stft(audio[i][0:lengths[i]], n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        spect = torch.sqrt(torch.pow(stft[:,:,0], 2) + torch.pow(stft[:,:,1], 2))
        spect = torch.log(1 + spect)
        mean = torch.mean(spect)
        std = torch.std(spect)
        spect = torch.add(spect, -mean)
        spect = torch.div(spect, std)
        length_tmp = spect.size(1)
        spects[i][0].narrow(1, 0, length_tmp).copy_(spect)
        spects_lengths[i] = length_tmp

    return spects, spects_lengths

def get_logits(new_audio, lengths, model):
    '''
    Get the output of deepspeech model.
    '''
    inputs, inputs_sizes = compute_mfcc(new_audio, lengths)
    #print('model_input', inputs.size())
    #print(inputs_sizes)
    _, logits, logit_sizes = model(inputs, inputs_sizes)
    #print('ok')

    return logits, logit_sizes
