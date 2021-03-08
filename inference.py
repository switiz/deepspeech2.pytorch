import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse

from dataset import dataset
from ksponspeech import KsponSpeechVocabulary
from utils import check_envirionment, char_errors, save_result, make_out, load_model, Timer
from model.deepspeech import DeepSpeech2

def inference(opt):
    timer = Timer()
    timer.log('Load Data')
    device = check_envirionment(opt['use_cuda'])
    vocab = KsponSpeechVocabulary(opt['vocab_path'])
    metric = char_errors(vocab)

    if opt['use_val_data']:
        val_dataset = dataset(opt, vocab, train=False)
        custom_loader = DataLoader(dataset=val_dataset, batch_size=opt['batch_size'] * 2, drop_last=True,
                                num_workers=8, collate_fn=val_dataset._collate_fn)
    else:
        #custom_dataset
        custom_dataset = dataset(opt, vocab, train=False)
        custom_loader = DataLoader(dataset=custom_dataset, batch_size=opt['batch_size'] * 2, drop_last=True,
                                   num_workers=8, collate_fn=custom_dataset._collate_fn)
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
    do_inference(custom_loader, vocab, model, device, metric)
    timer.endlog('Inference complete')

def do_inference(val_loader, vocab, model, device, metric):
    model.eval()
    progress_bar = tqdm(val_loader)
    target_list = list()
    predict_list = list()
    cer = 0.0
    for idx, data in enumerate(progress_bar):
        inputs, targets, input_lengths, target_lengths = data
        inputs = inputs.to(device)
        targets = targets[:, 1:].to(device)
        y_hats = model.greedy_search(inputs, input_lengths)
        for i in range(targets.size(0)):
            predict_list.append(vocab.label_to_string(y_hats[i].cpu().detach().numpy()))
    save_path = make_out()
    save_result(save_path, target_list, predict_list, inference=True)
    return cer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='pretrained.pt', help='initial weights path')
    option = parser.parse_args()

    with open('./data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    if option.weight_path != 'pretrained.pt':
        opt['weight_path'] = option.weight_path

    inference(opt)