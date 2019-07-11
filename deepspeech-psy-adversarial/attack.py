import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import argparse 
import os 
import time 
import json 
import scipy.io.wavfile as wav

from logits import get_logits
from mask import generate_th
from decoder import GreedyDecoder
from model import DeepSpeech
from warpctc_pytorch import CTCLoss

parser = argparse.ArgumentParser(description='deepspeech.pytorch attck')
parser.add_argument('--audio', help='Path for input wav')
parser.add_argument('--target', help='Target transcription')
parser.add_argument('--inputcsv', help='Input csv file for audio and target transcription')
parser.add_argument('--batchsize', type=int, default=10, help='Batch size for attack')
parser.add_argument('--out', default='./adversarial.wav', help='Path for the adversarial example')
parser.add_argument('--model', default='./models/deepspeech_11_266.pth', help='Pretrained model to load')
parser.add_argument('--lr1', type=int, required=False, default=10, help='Learning rate for step one optimization')
parser.add_argument('--iterations1', type=int, required=False, default=1000, help='Maximum number of iterations for step one')
parser.add_argument('--lr2', type=int, required=False, default=1, help='Learning rate for step two optimization')
parser.add_argument('--iterations2', type=int, required=False, default=4000, help='Maximum number of iterations for step two')
parser.add_argument('--l2penalty', type=float, required=False, default=float('inf'), help='Weight for l2 penalty on loss function')

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_audio(path):
    '''
    Load a audio and return data in torch tensor
    '''
    _, sound = wav.read(path)
    sound = np.array(sound, dtype=np.float)
    return sound

def load_transcript(transcript_path):
    '''
    Load a transcript as target phrase
    '''
    with open(transcript_path, 'r', encoding='utf8') as transcript_file:
        transcript = transcript_file.read().replace('\n', '')

    return transcript

def psd_transform(x, psd_max_ori, win_length, win_step):
    '''
    Compute PSD features of delta in pytorch,
    so we can differentiate through it.
    '''
    window = torch.hann_window(win_length).cuda()
    stft = torch.stft(x, n_fft=win_length, hop_length=win_step, win_length=win_length, window=window, center=False)
    spect = torch.sqrt(torch.pow(stft[:,:,0], 2) + torch.pow(stft[:,:,1], 2))
    z = np.sqrt(8. / 3.) * torch.abs(spect / win_length)
    z = torch.transpose(z, 1, 0)
    psd = z * z 
    PSD = ( 10 ** 9.6 ) / torch.reshape(psd_max_ori, [-1, 1]) * psd 

    return PSD

class Attack(nn.Module):
    def __init__(self, lr1, num_iterations1, lr2, num_iterations2, batch_size, l2penalty):

        self.lr1 = lr1
        self.num_iterations1 = num_iterations1 
        self.lr2 = lr2
        self.num_iterations2 = num_iterations2
        self.batch_size = batch_size 
        self.l2penalty = l2penalty

        with open('labels.json') as label_file:
            self.labels = str(''.join(json.load(label_file)))
        self.labels_map = dict([(self.labels[i], i) for i in range(len(self.labels))])
        self.decoder = GreedyDecoder(self.labels, blank_index=self.labels.index('_'))

    def attack1(self, audios, lengths, max_audio_len, targets, model_path):
        self.max_audio_len = max_audio_len
        self.original = torch.FloatTensor(audios).cuda()
        self.lengths = torch.FloatTensor(lengths)
        #define some variables
        self.delta1 = torch.zeros((self.batch_size, self.max_audio_len)).cuda()
        self.delta1.requires_grad=True
        self.rescale = torch.ones((self.batch_size, 1)).cuda()
        self.mask = torch.FloatTensor(np.array([[1 if i<l else 0 for i in range(self.max_audio_len)] for l in self.lengths])).cuda()
        self.final_deltas = [None]*self.batch_size

        self.target_phrase_lengths = torch.IntTensor(self.batch_size)
        self.target_phrase = []
        for x in range(self.batch_size):
            phrase = list(filter(None, [self.labels_map.get(x) for x in list(targets[x].upper())]))
            self.target_phrase_lengths[x] = len(phrase)
            self.target_phrase.extend(phrase)
        self.target_phrase = torch.IntTensor(self.target_phrase)
        #print(self.target_phrase.size(), self.target_phrase_lengths)
        model = DeepSpeech.load_model(model_path)
        model = model.cuda()
        self.optim1 = torch.optim.Adam([self.delta1], lr=self.lr1)

        criterion = CTCLoss()

        MAX = self.num_iterations1
        model.train()
        #self.noise = torch.randn(self.delta1.shape).cuda()  #[batch_szie * max_audio_len]
        for i in range(MAX):

            # print out some debug information every 10 iterations
            apply_delta = torch.clamp(self.delta1, -2000, 2000) * self.rescale  #[batch_size * max_audio_len]
            new_input = apply_delta * self.mask + self.original #[batch_size * max_audio_len]
            #pass_in = torch.clamp(new_input + self.noise, -2**15, 2**15-1) #[batch_szie * max_audio_len]
            pass_in = torch.clamp(new_input, -2**15, 2**15-1)
            pass_in = torch.div(pass_in, 2**15) #[batch_szie * max_audio_len]
            logits, logits_sizes = get_logits(pass_in, self.lengths.int(), model) #[batch_size * T * H]
            logits_ = logits.transpose(0, 1)
            # loss
            if not np.isinf(self.l2penalty):
                loss = torch.mean((new_input - self.original)**2) + self.l2penalty * criterion(logits_, self.target_phrase, logits_sizes, self.target_phrase_lengths).cuda()
            else:
                loss = criterion(logits_, self.target_phrase, logits_sizes, self.target_phrase_lengths).cuda()
            loss_value = loss.item()
            # optimize
            self.optim1.zero_grad()
            loss.backward() 
            # grad sign
            self.delta1.grad = torch.sign(self.delta1.grad)
            self.optim1.step()

            print('loss: ', loss_value)
            if (i+1)%10 == 0:
                decode_out, _ = self.decoder.decode(logits, logits_sizes)
                #print(decode_out, targets)

            for ii in range(self.batch_size):
                # Every 10 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                if ((i+1)%10 == 0 and decode_out[ii] == [targets[ii].upper()]) or (i==MAX-1 and self.final_deltas[ii] is None):
                    # If we're already below the threshold, then
                    # just reduce the threshold to the current
                    # point and save some time.
                    bound_tmp = torch.max(torch.abs(self.delta1[ii])).item()
                    if self.rescale[ii][0] * 2000 > bound_tmp:
                        print("It's way over", bound_tmp / 2000.0)
                        self.rescale[ii][0] = bound_tmp / 2000.0
                    
                    # Otherwise reduce it by some constant. The closer
                    # this number is to 1, the better quality the result
                    # will be. The smaller, the quicker we'll converge
                    # on a result but it will be lower quality.
                    self.rescale[ii][0] *= .8

                    # Adjust the best solution found so far
                    self.final_deltas[ii] = new_input[ii].cpu().detach().numpy()
                    print("bound=%f" % (2000 * self.rescale[ii][0]))

        return self.final_deltas
    
    def attack2(self, init_delta, target, model_path):
        self.delta2 = torch.FloatTensor(init_delta).cuda()
        self.delta2.requires_grad = True 
        self.rescale = torch.ones((self.batch_size, 1)).cuda()
        self.final_deltas = [None]*self.batch_size
        self.alpha = torch.ones((self.batch_size,)).cuda() * 1
        #self.alpha = 1

        model = DeepSpeech.load_model(model_path)
        model = model.cuda()

        self.optim21 = torch.optim.Adam([self.delta2], lr=2)
        self.optim22 = torch.optim.Adam([self.delta2], lr=self.lr2)

        criterion = CTCLoss()

        th_batch = []
        psd_max_batch = []
        for ii in range(self.batch_size):
            th, _, psd_max = generate_th(self.original[ii].cpu().numpy(), fs=16000, window_size=2048)
            th_batch.append(th)
            psd_max_batch.append(psd_max)
        th_batch = np.array(th_batch)
        psd_max_batch = np.array(psd_max_batch)
        th_batch = torch.FloatTensor(th_batch).cuda()
        psd_max_batch = torch.FloatTensor(psd_max_batch).cuda()

        MAX = self.num_iterations2
        model.train()
        deltas = []
        loss_th = [np.inf] *self.batch_size
        for i in range(MAX):
            # print out some debug information every 10 iterations
            #print(self.delta)
            apply_delta = torch.clamp(self.delta2, -2000, 2000) * self.rescale  #[batch_size * max_audio_len]
            new_input = apply_delta * self.mask + self.original #[batch_size * max_audio_len]
            #pass_in = torch.clamp(new_input + self.noise, -2**15, 2**15-1) #[batch_szie * max_audio_len]
            pass_in = torch.clamp(new_input, -2**15, 2**15-1)
            pass_in = torch.div(pass_in, 2**15) #[batch_szie * max_audio_len]
            logits, logits_sizes = get_logits(pass_in, self.lengths.int(), model) #[batch_size * T * H]
            logits_ = logits.transpose(0, 1)
            # loss

            loss2 = criterion(logits_, self.target_phrase, logits_sizes, self.target_phrase_lengths).cuda()
            loss_value_2 = loss2.item()
            self.optim21.zero_grad()
            loss2.backward(retain_graph=True) 
            self.delta2.grad = torch.sign(self.delta2.grad)
            self.optim21.step()

            loss1 = 0
            loss1_each = []
            for ii in range(self.batch_size):
                psd = psd_transform(apply_delta[ii], psd_max_batch[ii], win_length=2048, win_step=512)
                loss1 += self.alpha[ii] * torch.mean(torch.relu(psd - th_batch[ii]))
                loss1_each.append(torch.mean(torch.relu(psd - th_batch[ii])).item())
                #psd_num = psd.cpu().detach().numpy()
                #th_ = th_batch[ii].cpu().detach().numpy()

            loss1 = loss1 / self.batch_size
            loss_value_1 = np.mean(loss1_each) 
            self.optim22.zero_grad()
            loss1.backward()
            for ii in range(self.batch_size):
                self.delta2.grad[ii] = self.alpha[ii] * torch.sign(self.delta2.grad[ii])
            
            grad = np.sum(self.delta2.grad.cpu().numpy())
            if grad != grad:
                print("NaN")
            
            self.optim22.step()

            apply_delta_ = torch.clamp(self.delta2, -2000, 2000) * self.rescale
            deltas.append(apply_delta_.cpu().detach().numpy())
            if 2000. in deltas[i] or -2000. in deltas[i]:
                print("True")
                #break 

            print('loss: ', loss_value_1, loss_value_2)

            if i+1 == 2000:
                param_groups = self.optim21.param_groups
                for g in param_groups:
                    g['lr'] = 0.1
            if i+1 == 3200:
                param_groups = self.optim21.param_groups
                for g in param_groups:
                    g['lr'] = 0.01

            if i+1 == 2000:
                param_groups = self.optim22.param_groups
                for g in param_groups:
                    g['lr'] = 0.1
            if i+1 == 3200:
                param_groups = self.optim22.param_groups
                for g in param_groups:
                    g['lr'] = 0.01

            if (i+1)%10 == 0:
                decode_out, _ = self.decoder.decode(logits, logits_sizes)
                print(i+1, decode_out[0], [target[0]])

            for ii in range(self.batch_size):
                # Every 10 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                if ((i+1)%50 == 0 and decode_out[ii] == [target[ii].upper()]) or (i==MAX-1 and self.final_deltas[ii] is None):
                    self.alpha[ii] = 1.2 * self.alpha[ii]
                    if self.alpha[ii] > 1000:
                        self.alpha[ii] = 1000 
                    # Adjust the best solution found so far
                    if loss1_each[ii] < loss_th[ii]:
                        loss_th[ii] = loss1_each[ii]
                        self.final_deltas[ii] = new_input[ii][0:self.lengths[ii].int()].cpu().detach().numpy()
                    print("up alpha=%f" % (self.alpha[ii]))
                
                if ((i+1)%100 == 0 and decode_out[ii] != [target[ii].upper()]):
                    self.alpha[ii] = 0.6 * self.alpha[ii] 
                    '''
                    if self.alpha <= 100:
                        self.alpha = 100
                    else:
                        # Adjust the best solution found so far
                        print("down alpha=%f" % (self.alpha))
                    '''
                    print("down alpha=%f" % (self.alpha[ii]))
        return self.final_deltas



def main():
    args = parser.parse_args()
    # read audio and target transcript
    with open(args.inputcsv) as f:
        ids = f.readlines()
    ids = [x.strip().split(',') for x in ids]
    audios = []
    targets = []
    for data in ids:
        audio_path, target_path = data[0], data[1]
        #load the input audio
        audio = load_audio(audio_path)
        audios.append(audio)
        #load the target transcript
        target = load_transcript(target_path)
        targets.append(target)
    
    # set up the Attack
    attack = Attack(
                    lr1=args.lr1,
                    num_iterations1=args.iterations1,
                    lr2=args.lr2,
                    num_iterations2=args.iterations2,
                    batch_size=args.batchsize,
                    l2penalty=args.l2penalty
                    )

    #make batch and attack
    num = int(len(ids) / args.batchsize)
    for i in range(num):
        input_audios = audios[i*args.batchsize:(i+1)*args.batchsize]
        input_audios_lengths = [len(x) for x in input_audios]
        input_targets = targets[i*args.batchsize:(i+1)*args.batchsize]
        # decreasing order
        sort_length = np.argsort(-np.array(input_audios_lengths))
        audios_tmp = [input_audios[x] for x in sort_length]
        lengths_tmp = [input_audios_lengths[x] for x in sort_length]
        targets_tmp = [input_targets[x] for x in sort_length]

        input_audios = audios_tmp
        input_audios_lengths = lengths_tmp
        input_targets = targets_tmp

        max_length = max(input_audios_lengths)
        inputs = np.zeros((args.batchsize, max_length))
        for j in range(args.batchsize):
            inputs[j][0:input_audios_lengths[j]] = input_audios[j]

        # do the attack
        final = attack.attack1(
                            audios=inputs,
                            lengths=input_audios_lengths,
                            max_audio_len=max_length,
                            targets=input_targets,
                            model_path=args.model)
        print("attack stage1 is done!")
        # save the final adversarial audio
        #final = torch.reshape(final, (final.size(0), 1))
        delta = final - inputs
        final = attack.attack2(
                            init_delta=delta,
                            target=input_targets,
                            model_path=args.model)
        for ii in range(args.batchsize):
            wav.write('./result/' + str(i*args.batchsize + sort_length[ii]) + '.wav', 16000, np.array(np.clip(np.round(final[ii]), -2**15, 2**15-1), dtype=np.int16))

main()
