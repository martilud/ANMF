from utils import *
from librosa import stft, istft
import numpy as np
from NMF import *

class audio_separation:
    """
    Wrapper class for NMF_separation for audio applciations.
    Main part is applying STFT and ISTFT and metrics for SNR and SDR

    Parameters
    --------
        prob: str, problem formulation. Currently only supports "semi", for standard semi-supervised fitting
            and "semi-adv" for adversarial semi-supervised fitting.
        project: bool, wether or not to use a a basis for the unobserved data during testing
        n_fft: int, size of fft used for STFT. Should be a power of 2.
        hop_length: int, hop length passed to STFT
        win_length: int, window length passed to STFT
        sep_args: arguments passed to NMF_separation
    """

    def __init__(self,prob = "semi", project = False, n_fft = 512, hop_length = None, win_length = None, **sep_args):
        self.prob = prob
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.project = project

        self.sep = NMF_separation(prob = self.prob, **sep_args)

    def fit(self, u, v = None):
        """
        Fits bases given list of audio signals u and list of audio signals v.
        Currently only really works for semi and semi_adv
        """

        U_speech = []
        Z_speech = []
        for i in range(len(u)):
            Z_speech.append(stft(u[i],n_fft=self.n_fft, hop_length = self.hop_length, win_length = self.win_length))
            U_speech.append(np.abs(Z_speech[i]))
        U_speech_concat = np.concatenate(U_speech, axis = -1)

        V = []
        Z_mix =  []
        for i in range(len(v)):
            Z_mix.append(stft(v[i],n_fft=self.n_fft, hop_length = self.hop_length, win_length = self.win_length))
            V.append(np.abs(Z_mix[i]))
        V_concat = np.concatenate(V, axis = -1)

        self.sep.fit(U_r = [U_speech_concat], V = V_concat)


    def separate(self, v = None):
        """
        Separates list of audio signals v. 
        """ 
        out = []
        for i in range(len(v)):
            Z = stft(v[i],n_fft=self.n_fft, hop_length = self.hop_length, win_length = self.win_length)
            if self.project:
                U_reconst = np.dot(self.sep.NMFs[0].W, self.sep.NMFs[0].transform(np.abs(Z)))     
            else:
                U_reconst = self.sep.separate(np.abs(Z))[:,0,:]
            out.append(istft(U_reconst * np.exp(1j * np.angle(Z)), n_fft = self.n_fft, length = len(v[i])))

        return out

    def eval(self, u, v, metric = 'SNR', out = None, aggregate = None):
        """
        Evaluates difference between list of audio signals u and list of audio signals v. 
        """
        
        assert len(u) == len(v)
        
        if out is None:
            out = self.separate(v)

        if metric == 'SNR':
            return np.mean([calculate_snr(u[i], out[i]) for i in range(len(v))])
        elif metric == 'SDR':
            #return np.nanmean(np.concatenate([bss_eval(u[i], out[i], window = self.sdr_window, hop = self.sdr_window)[0] for i in range(len(u))], axis = -1))
            return np.mean([calculate_sdr(u[i], out[i]) for i in range(len(u))])

