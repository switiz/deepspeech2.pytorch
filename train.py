import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse

from dataset import dataset
from ksponspeech import KsponSpeechVocabulary
from utils import check_envirionment, char_errors, save_result, save_model, make_checkpoint, load_model, Timer
from model.deepspeech import DeepSpeech2

def train(opt):
    timer = Timer()
    timer.log('Load Data')
    device = check_envirionment(opt['use_cuda'])
    vocab = KsponSpeechVocabulary(opt['vocab_path'])
    metric = char_errors(vocab)
    train_dataset = dataset(opt, vocab, train=True)
    val_dataset = dataset(opt, vocab, train=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt['batch_size'], drop_last=True,
                              num_workers=8, collate_fn=train_dataset._collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt['batch_size'] * 2, drop_last=True,
                            num_workers=8, collate_fn=val_dataset._collate_fn)

    timer.log("Train data : {} Val data : {}".format(len(train_dataset), len(val_dataset)))

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

    timer.startlog('Train Start')
    print_epoch = 0
    loss = float(1e9)
    cer, val_cer = 0.0, 0.0
    for epoch in range(start_epoch, opt['epochs']):
        print_epoch = epoch
        loss, cer, model_save_path = train_on_epoch(train_loader, model, optimizer, criterion, scheduler, device,
                                                    print_epoch, metric, loss)
        if print_epoch % opt['validation_every'] == 0:
            val_cer = validation_on_epoch(val_loader, vocab, model, device, metric, model_save_path, timer)
    # print(val_cer)
    timer.log('Train : epoch {} loss:{:.2f} train_cer:{:.2f} val_cer:{:.2f}'.format(print_epoch, loss, cer, val_cer))
    timer.endlog('Train Complete')


def train_on_epoch(train_loader, model, optimizer, criterion, scheduler, device, epoch, metric, pre_loss):
    model.train()
    time_stamp = 0
    total_num = 0
    progress_bar = tqdm(train_loader)
    cer = 0.0
    for data in progress_bar:
        inputs, targets, input_lengths, target_lengths = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs, output_lengths = model(inputs, input_lengths)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        loss = criterion(outputs.transpose(0, 1), targets, output_lengths, target_lengths)
        total_num += int(input_lengths.sum())
        optimizer.zero_grad()
        epoch_loss = loss.item()
        time_stamp += 1
        if time_stamp == 10 or time_stamp % opt['cer_every'] == 0 or time_stamp == len(train_loader)-1:
            cer = metric(targets, outputs.max(-1)[1])
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
        progress_bar.set_description(
            'epoch : {}, loss : {:.2f}, cer : {:.2f}, learning_rate : {} '.format(epoch, epoch_loss, cer,
                                                                                  scheduler.get_last_lr()))
    # model save
    if epoch % opt['save_every'] == 0:
        check_point = make_checkpoint(model, epoch, optimizer)
        if epoch_loss < pre_loss:
            #save best model
            model_save_path = save_model(check_point, True)
        else:
            model_save_path = save_model(check_point, False)

    return epoch_loss, cer, model_save_path


def validation_on_epoch(val_loader, vocab, model, device, metric, model_save_path, timer):
    timer.log('validation start')
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
            target_list.append(vocab.label_to_string(targets[i]))
            predict_list.append(vocab.label_to_string(y_hats[i].cpu().detach().numpy()))
        if idx == len(progress_bar):
            cer = metric(targets, y_hats)
    save_result(model_save_path, target_list, predict_list)
    timer.log('validation complete')
    return cer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    option = parser.parse_args()

    with open('./data/config.yaml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    if option.resume:
        opt['resume'] = True
    train(opt)
