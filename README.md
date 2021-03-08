# DeepSpeech2 
### (Pytorch Implementation using AIHUB data)
#### Paper: https://arxiv.org/abs/1512.02595

----
한국어 음성인식 모델이 종합된 Kospeech의 경우 다양한 모델을 지원함에 따라 코드 복잡도가 높아 단순 모델 학습에는 용의하지 않습니다.

그래서 본 git은 Kospeech의 모델 코드를 이용하여 몇몇 버그 수정 및 단순화에 집중하였습니다.

LAS-> Transformer -> Conformer -> RnnT순으로 추가예정입니다.

코드 사용을 허락해주신 Kospeech Author 김수환 님에게 감사드립니다.

Original git (Kospeech): https://github.com/sooftware/KoSpeech

### Note
 - KospoonSpeech preprocess code import
 - single file inference

--- 
### Step 1 PreProcess (추가예정)
preprocess 과정의 경우 KospoonSpeech방식을 그대로 이용하였기 때문에 아래 git의 코드를 이용하시면 됩니다.

https://github.com/sooftware/KoSpeech/tree/latest/dataset/kspon

### Step 2 Configuration
data/config.yaml 파일의 내용을 load하여 각 코드에서 사용하고 있기 때문에 custom이 필요시 config.yaml을 변경해주어야합니다.

특히 각각의 PC마다 dataset의 위치가 다르기 때문에 해당 부분을 주요하게 변경해주시면됩니다.

- root : dataset root 디렉토리
- script_data_path : script (kospeech style) 디렉토리
- vocab_path : kosponSpeech-preprocess로 생성된 vocab 파일

``` 
root : C:/SpeechRecognitionDataset/Dataset/AI_hub
script_data_path: C:/Users/sanma/PycharmProjects/deepspeech2.pytorch/data/aihub/transcripts.txt
vocab_path: C:/Users/sanma/PycharmProjects/deepspeech2.pytorch/data/aihub/aihub_labels.csv

#input feature
feature: melspectrogram
n_mels : 80
use_npy: False
split_balance : 0.1

#train
batch_size: 8
epochs: 10
use_cuda: True
cer_every: 10000

#model_save_load
resume: False
save_every: 1

#inference
inference: True
use_val_data: True
weight_path: C:/Users/sanma/PycharmProjects/deepspeech2.pytorch/runs/2021_03_04_10_26_57/pretrained.pt

#
#root : C:/SpeechRecognitionDataset/Dataset/AI_hub
#script_data_path: C:/Users/sanma/PycharmProjects/deepspeech2.pytorch/data/aihub/transcripts.txt
#vocab_path: C:/Users/sanma/PycharmProjects/deepspeech2.pytorch/data/aihub/aihub_labels.csv
#model_save_path: C:/Users/sanma/PycharmProjects/deepspeech2.pytorch/data/runs/train.pt

#validation
validation_every: 5

#input
hidden_dim : 1024
activation: hardtanh
use_bidirectional: True
rnn_type: gru
num_encoder_layers: 3
dropout_p: 0.3
```

### Step 3 Train

```
python train.py
```

#### Resume
config.yaml의 resume을 True로 변경하거나 --resume arg를 넣어 동작시켜 줍니다.
가장 마지막모델을 기준으로 load하여 resume이 실행됩니다.

```
python train.py --resume
```

### Step 4 Inference
[pretrained weight](https://drive.google.com/drive/folders/1coWx1pOBFwPnWYShE_h896qH0p_2Z760?usp=sharing)

inference.py를 실행하면 되며 현재는 validation data를 이용하여 inference하도록 되어있습니다.
validation data가 아닌 custom data의 경우 현재는 별도의 dataset를 생성해야합니다.

```
python inference.py
```

### Step 5 Result
데이터셋 : AIHUB 1000h Korean speech data corpus

PC사양 : Windows10, AMD 3600, RAM32, RTX 3080, Pytorch 1.7

소요시간 : EPOCH당 14시간

CER : 0.35 (35%)

| LABEL       | PREDICT     |
|-------------|------------------|
| 금방 가.       | <sos>금방  가.      |
| 네 암.        | <sos>네  암        |
| 그게 너무 멋있었어. | <sos>그게  너무 멋있어. |
| 근데 사람마다 다   | <sos>근데   사람마다 다 |
| 때리진 않았지?    | <sos>때리진  않았지?   |


###Reference

    @ARTICLE{2021-kospeech,
      author    = {Kim, Soohwan and Bae, Seyoung and Won, Cheolhwang},
      title     = {KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition},
      url       = {https://www.sciencedirect.com/science/article/pii/S2665963821000026},
      month     = {February},
      year      = {2021},
      publisher = {ELSEVIER},
      journal   = {SIMPAC},
      pages     = {Volume 7, 100054}
    }