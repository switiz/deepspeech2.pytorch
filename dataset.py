import librosa
import os
from torch.utils.data import Dataset
import torch
import pandas
import numpy as np
from ksponspeech import KsponSpeechVocabulary
from sklearn.model_selection import train_test_split

class dataset(Dataset):
    def __init__(self, opt, vocab, train=True, inference=False):
        self.root = opt['root']
        self.audio_feature = opt['feature']
        self.input_size = opt['n_mels']
        self.script_data_path = opt['script_data_path']
        self.use_npy = opt['use_npy']
        self.balance = opt['split_balance']
        self.train = train
        self.audio_data_path = self.root
        self.audio_paths = list()
        self.transcripts = list()
        self._get_transcript()

        self.collate_fn = self._collate_fn
        if self.use_npy:
            self.input_data = np.load(self.audio_data_path)
        self.vocab = vocab

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio_path, transcript = self.audio_paths[idx], self.transcripts[idx]
        audio_path = os.path.join(self.audio_data_path, audio_path)
        feature = self._get_audio_feature(audio_path)
        script = self._parse_transcript(transcript)
        return feature, script

    def _get_transcript(self):
        with open(self.script_data_path) as f:
            transcript = f.readlines()
        #transcript = [x for x in transcript if 'KsponSpeech_00' in x]
        transcript = self._split_dataset(transcript, balance=self.balance)
        self._post_process(transcript)

    def _post_process(self, transcript):
        for idx, line in enumerate(transcript):
            audio_path, korean_transcript, transcripts = line.split('\t')
            transcripts = transcripts.replace('\n', '')
            self.audio_paths.append(audio_path)
            self.transcripts.append(transcripts)

    def _get_audio_feature(self, path):
        if self.use_npy:
            feature = torch.zeros(10)
        else:
            n_mels = self.input_size
            signal = np.memmap(path, dtype='h', mode='r').astype('float32')
            signal = signal / 32767
            sample_rate = 16000
            frame_length = 20
            frame_shift = 10
            n_fft = int(round(sample_rate * 0.001 * frame_length))
            hop_length = int(round(sample_rate * 0.001 * frame_shift))

            if self.audio_feature == 'melspectrogram':
                feature = librosa.feature.melspectrogram(signal, sample_rate, n_fft=n_fft, n_mels=n_mels,
                                                         hop_length=hop_length)
                feature = librosa.amplitude_to_db(feature, ref=np.max)

            elif self.audio_feature == 'mfcc':
                feature = librosa.feature.mfcc(signal, sample_rate, n_mels, n_fft, hop_length)
                feature = librosa.amplitude_to_db(feature, ref=np.max)

            else:
                feature = signal
            feature = torch.FloatTensor(feature).transpose(0, 1)
        return feature

    def _parse_transcript(self, transcript):
        """ Parses transcript """
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.vocab.sos_id))
        for token in tokens:
            transcript.append(int(token))
        transcript.append(int(self.vocab.eos_id))

        return transcript

    def _collate_fn(self, batch):
        def seq_length_(p):
            return len(p[0])

        def target_length_(p):
            return len(p[1])

        # sort by sequence length for rnn.pack_padded_sequence()
        batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

        seq_lengths = [len(s[0]) for s in batch]
        target_lengths = [len(s[1]) for s in batch]

        max_seq_sample = max(batch, key=seq_length_)[0]
        max_target_sample = max(batch, key=target_length_)[1]

        max_seq_size = max_seq_sample.size(0)
        max_target_size = len(max_target_sample)

        feat_size = max_seq_sample.size(1)
        batch_size = len(batch)

        seqs = torch.zeros(batch_size, max_seq_size, feat_size)

        targets = torch.zeros(batch_size, max_target_size).to(torch.long)
        targets.fill_(self.vocab.pad_id)

        for x in range(batch_size):
            sample = batch[x]
            tensor = sample[0]
            target = sample[1]
            seq_length = tensor.size(0)

            seqs[x].narrow(0, 0, seq_length).copy_(tensor)
            targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

        seq_lengths = torch.IntTensor(seq_lengths)
        target_lengths = target_lengths
        return seqs, targets, seq_lengths, target_lengths

    def _split_dataset(self, transcripts, balance):
        train, val = train_test_split(transcripts, test_size=balance)
        if self.train:
            return train
        else:
            return val