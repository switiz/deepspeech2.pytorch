import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import librosa

from dataset import dataset
from ksponspeech import KsponSpeechVocabulary
from utils import check_envirionment, char_errors, save_result, make_out, load_model, Timer
from model.deepspeech import DeepSpeech2

def load_audio(audio_path, extension='pcm'):
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    if extension == 'pcm':
        signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')
        return signal / 32767  # normalize audio

    elif extension == 'wav' or extension == 'flac':
        signal, _ = librosa.load(audio_path, sr=16000)
        return signal


def parse_audio(audio_path, audio_extension='pcm'):
    signal = load_audio(audio_path, extension=audio_extension)
    sample_rate = 16000
    frame_length = 20
    frame_shift = 10
    n_fft = int(round(sample_rate * 0.001 * frame_length))
    hop_length = int(round(sample_rate * 0.001 * frame_shift))

    if opt['feature'] == 'melspectrogram':
        feature = librosa.feature.melspectrogram(signal, sample_rate, n_fft=n_fft, n_mels=opt['n_mels'],
                                                 hop_length=hop_length)
        feature = librosa.amplitude_to_db(feature, ref=np.max)

    return torch.FloatTensor(feature).transpose(0, 1)


def inference(opt):
    timer = Timer()
    timer.log('Load Data')
    device = check_envirionment(opt['use_cuda'])
    vocab = KsponSpeechVocabulary(opt['vocab_path'])

    if opt['use_val_data']:
        val_dataset = dataset(opt, vocab, train=False)
        custom_loader = DataLoader(dataset=val_dataset, batch_size=opt['batch_size'] * 2, drop_last=True,
                                num_workers=8, collate_fn=val_dataset._collate_fn)
    else:
        #custom_dataset
        feature = parse_audio(opt['audio_path'])
        feature = feature.to(device)
        input_length = torch.LongTensor([len(feature)])

    model = DeepSpeech2(
        input_size=opt['n_mels'],
        num_classes=len(vocab),
        rnn_type=opt['rnn_type'],
        num_rnn_layers=opt['num_encoder_layers'],
        rnn_hidden_dim=opt['hidden_dim'],
        dropout_p=opt['dropout_p'],
        bidirectional=opt['use_bidirectional'],
        activation=opt['activation'],
        device=device,
    ).to(device)

    model, optimizer, criterion, scheduler, start_epoch = load_model(opt, model, vocab)
    print('-'*40)
    print(model)
    print('-'*40)

    timer.startlog('Inference Start')
    model.eval()
    y_hats = model.greedy_search(feature.unsqueeze(0), input_length)
    sentance = vocab.label_to_string(y_hats.cpu().detach().numpy())
    print(sentance)
    timer.endlog('Inference complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, default='', help='audio_path')
    option = parser.parse_args()

    with open('./data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    if option.audio_path != '':
        opt['audio_path'] = option.audio_path
        opt['use_val_data'] = False

    opt['inference'] = True
    inference(opt)