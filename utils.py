import torch
import logging
import platform
import pandas as pd
import Levenshtein as Lev
import math
import os
import shutil
from time import strftime, localtime
import time
from datetime import timedelta


def __init__(self, vocab) -> None:
    self.total_dist = 0.0
    self.total_length = 0.0
    self.vocab = vocab


def __call__(self, targets, y_hats):
    """ Calculating character error rate """
    dist, length = self._get_distance(targets, y_hats)
    self.total_dist += dist
    self.total_length += length
    return self.total_dist / self.total_length


class char_errors():
    def __init__(self, vocab):
        self.total_dist = 0.0
        self.total_length = 0.0
        self.vocab = vocab

    def __call__(self, reference, hypothesis):
        dist, length = self._get_distance(reference, hypothesis)
        self.total_dist += dist
        self.total_length += length
        return self.total_dist / self.total_length

    def _get_distance(self, targets, y_hats):
        """
        Provides total character distance between targets & y_hats

        Args:
            targets (torch.Tensor): set of ground truth
            y_hats (torch.Tensor): predicted y values (y_hat) by the model

        Returns: total_dist, total_length
            - **total_dist**: total distance between targets & y_hats
            - **total_length**: total length of targets sequence
        """
        total_dist = 0
        total_length = 0

        for (target, y_hat) in zip(targets, y_hats):
            s1 = self.vocab.label_to_string(target)
            s2 = self.vocab.label_to_string(y_hat)

            dist, length = self._metric(s1, s2)

            total_dist += dist
            total_length += length

        return total_dist, total_length

    def _metric(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to characters.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1 = s1.replace(' ', '')
        s2 = s2.replace(' ', '')

        # if '_' in sentence, means subword-unit, delete '_'
        if '_' in s1:
            s1 = s1.replace('_', '')

        if '_' in s2:
            s2 = s2.replace('_', '')

        dist = Lev.distance(s2, s1)
        length = len(s1.replace(' ', ''))

        return dist, length


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER).
        CER = (Sc + Dc + Ic) / Nc
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


def load_dataset(transcripts_path):
    """
    Provides dictionary of filename and labels

    Args:
        transcripts_path (str): path of transcripts

    Returns: target_dict
        - **target_dict** (dict): dictionary of filename and labels
    """
    audio_paths = list()
    transcripts = list()

    with open(transcripts_path) as f:
        for idx, line in enumerate(f.readlines()):
            audio_path, korean_transcript, transcript = line.split('\t')
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts


def check_envirionment(use_cuda):
    """
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    logger = logging.getLogger()

    if print:
        logger.level = logging.DEBUG
    logger.info(f"Operating System : {platform.system()} {platform.release()}")
    logger.info(f"Processor : {platform.processor()}")

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
        logger.info(f"PyTorch version : {torch.__version__}")

    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"PyTorch version : {torch.__version__}")

    return device


def make_checkpoint(model, epoch, optimizer):
    print('make_checkpoint:', epoch)
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    return checkpoint


def get_last_checkpoint_dir(root):
    checkpoints_path = sorted(os.listdir(root), reverse=True)[-1]
    return os.path.join(root, checkpoints_path) + '/'


def make_out():
    dirname = './out/inference/'
    if os.path.exists(dirname) is False:
        os.makedirs(dirname)
    return dirname


def make_dir(epoch):
    root = './runs/'
    date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    dirname = root + f"{date_time}/"
    if epoch == 0:
        if os.path.exists(dirname) is False:
            os.makedirs(dirname)
    else:
        dirname = get_last_checkpoint_dir(root)
    return dirname


def save_model(checkpoint, is_best, save_pretrined=True):
    dirname = make_dir(checkpoint['epoch'] - 1)
    f_path = "checkpoint.pt"
    pre_path = 'pretrained.pt'
    if os.path.exists(dirname + f_path):
        os.remove(dirname + f_path)
    torch.save(checkpoint, dirname + f_path)
    if save_pretrined:
        torch.save(checkpoint['state_dict'], dirname + pre_path)

    if is_best:
        best_fpath = 'best_model.pt'
        shutil.copyfile(dirname + f_path, dirname + best_fpath)

    print('model save on epoch:{}'.format(checkpoint['epoch']))
    return dirname


def load_model(opt, model, vocab):
    start_epoch = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=1e-05)
    criterion = torch.nn.CTCLoss(blank=vocab.blank_id, reduction='mean', zero_infinity=True)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00007,
                                                  step_size_up=5, max_lr=0.0001,
                                                  mode='triangular', cycle_momentum=False)
    if opt['resume']:
        root = './runs/'
        resume_path = get_last_checkpoint_dir(root) + 'best_model.pt'
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('[resume_model_load] resuming epoch:{}'.format(start_epoch))

    elif opt['inference']:
        weight_path = opt['weight_path']
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint)
        print('[inference_model_load] complete')

    return model, optimizer, criterion, scheduler, start_epoch


def save_result(model_save_path, target_list, predict_list, inference=False):
    if inference:
        results = {
            'predictions': predict_list
        }
    else:
        results = {
            'targets': target_list,
            'predictions': predict_list
        }
    date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    save_path = f"{date_time}-valid.csv"
    save_path = model_save_path + save_path
    results = pd.DataFrame(results)
    results.to_csv(save_path, index=False, encoding='cp949')


class TriStageLRScheduler():
    """
    Tri-Stage Learning Rate Scheduler
    Implement the learning rate scheduler in "SpecAugment"
    """

    def __init__(self, optimizer, init_lr, peak_lr, final_lr, init_lr_scale, final_lr_scale, warmup_steps, total_steps):
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
        assert isinstance(total_steps, int), "total_steps should be inteager type"

        super(TriStageLRScheduler, self).__init__(optimizer, init_lr)
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.init_lr *= init_lr_scale
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.hold_steps = int(total_steps >> 1) - warmup_steps
        self.decay_steps = int(total_steps >> 1)

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_step = 0

    def _decide_stage(self):
        if self.update_step < self.warmup_steps:
            return 0, self.update_step

        offset = self.warmup_steps

        if self.update_step < offset + self.hold_steps:
            return 1, self.update_step - offset

        offset += self.hold_steps

        if self.update_step <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_step - offset

        offset += self.decay_steps

        return 3, self.update_step - offset

    def step(self):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_step += 1

        return self.lr


class Timer():
    def __init__(self):
        self.start = 0

    def startlog(self, message):
        self.start = time.time()
        self.log(message)

    def secondsToStr(self, elapsed=None):
        if elapsed is None:
            return strftime("%Y-%m-%d %H:%M:%S", localtime())
        else:
            return str(timedelta(seconds=elapsed))

    def log(self, s, elapsed=None):
        line = "=" * 40
        print(line)
        print('[{}]'.format(self.secondsToStr()), '-', s)
        if elapsed:
            print("Elapsed time:", elapsed)
        print(line)
        print()

    def endlog(self, message):
        end = time.time()
        elapsed = end - self.start
        self.log(message, self.secondsToStr(elapsed))
