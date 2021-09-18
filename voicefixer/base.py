import librosa.display
from voicefixer.tools.pytorch_util import *
from voicefixer.tools.wav import *
from voicefixer.restorer.model import VoiceFixer as voicefixer_fe
import os

EPS=1e-8

class VoiceFixer():
    def __init__(self):
        self._model = voicefixer_fe(channels=2, sample_rate=44100)
        self._model = self._model.load_from_checkpoint(os.path.join(os.path.expanduser('~'), ".cache/voicefixer/analysis_module/checkpoints/epoch=15_trimed_bn.ckpt"))
        self._model.eval()

    def _load_wav_energy(self, path, sample_rate, threshold=0.95):
        wav_10k, _ = librosa.load(path, sr=sample_rate)
        stft = np.log10(np.abs(librosa.stft(wav_10k))+1.0)
        fbins = stft.shape[0]
        e_stft = np.sum(stft, axis=1)
        for i in range(e_stft.shape[0]):
            e_stft[-i-1] = np.sum(e_stft[:-i-1])
        total = e_stft[-1]
        for i in range(e_stft.shape[0]):
            if(e_stft[i] < total*threshold):continue
            else: break
        return wav_10k, int((sample_rate//2) * (i/fbins))

    def _load_wav(self, path, sample_rate, threshold=0.95):
        wav_10k, _ = librosa.load(path, sr=sample_rate)
        return wav_10k

    def _amp_to_original_f(self, mel_sp_est, mel_sp_target, cutoff=0.2):
        freq_dim = mel_sp_target.size()[-1]
        mel_sp_est_low, mel_sp_target_low = mel_sp_est[..., 5:int(freq_dim * cutoff)], mel_sp_target[..., 5:int(freq_dim * cutoff)]
        energy_est, energy_target = torch.mean(mel_sp_est_low, dim=(2, 3)), torch.mean(mel_sp_target_low, dim=(2, 3))
        amp_ratio = energy_target / energy_est
        return mel_sp_est * amp_ratio[..., None, None], mel_sp_target

    def _trim_center(self, est, ref):
        diff = np.abs(est.shape[-1] - ref.shape[-1])
        if (est.shape[-1] == ref.shape[-1]):
            return est, ref
        elif (est.shape[-1] > ref.shape[-1]):
            min_len = min(est.shape[-1], ref.shape[-1])
            est, ref = est[..., int(diff // 2):-int(diff // 2)], ref
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref
        else:
            min_len = min(est.shape[-1], ref.shape[-1])
            est, ref = est, ref[..., int(diff // 2):-int(diff // 2)]
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref

    def _pre(self, model, input, cuda):
        input = input[None, None, ...]
        input = torch.tensor(input)
        if(cuda and torch.cuda.is_available()):
            input = input.cuda()
        sp, _, _ = model.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = model.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
        # return models.to_log(sp), models.to_log(mel_orig)
        return sp, mel_orig

    def restore(self, input, output, cuda=False, mode=0):
        if(cuda and torch.cuda.is_available()):
            self._model = self._model.cuda()
        # metrics = {}
        if(mode == 1):
            self._model.train() # More effective on seriously demaged speech
        elif(mode == 2):
            self._model.generator.denoiser.train() # Another option worth trying
        else:
            self._model.eval()

        with torch.no_grad():
            wav_10k = self._load_wav(input, sample_rate=44100)
            res = []
            seg_length = 44100*60
            break_point = seg_length
            while break_point < wav_10k.shape[0]+seg_length:
                segment = wav_10k[break_point-seg_length:break_point]
                sp,mel_noisy = self._pre(self._model, segment, cuda)
                out_model = self._model(sp, mel_noisy)
                denoised_mel = from_log(out_model['mel'])
                # if(meta["unify_energy"]):
                #    denoised_mel, mel_noisy = self.amp_to_original_f(mel_sp_est=denoised_mel,mel_sp_target=mel_noisy)
                out = self._model.vocoder(denoised_mel)
                # unify energy
                if(torch.max(torch.abs(out)) > 1.0):
                    out = out / torch.max(torch.abs(out))
                    print("Warning: Exceed energy limit,", input)
                # frame alignment
                out, _ = self._trim_center(out, segment)
                res.append(out)
                break_point += seg_length
            out = torch.cat(res,-1)
            save_wave(tensor2numpy(out[0,...]),fname=output,sample_rate=44100)
        # return metrics


