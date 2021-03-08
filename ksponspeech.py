import csv
class KsponSpeechVocabulary():
    def __init__(self, vocab_path, output_unit='character'):
        super(KsponSpeechVocabulary, self).__init__()

        self.vocab_dict, self.id_dict = self.load_vocab(vocab_path, encoding='utf-8')
        self.sos_id = int(self.vocab_dict['<sos>'])
        self.eos_id = int(self.vocab_dict['<eos>'])
        self.pad_id = int(self.vocab_dict['<pad>'])
        self.blank_id = int(self.vocab_dict['<blank>'])

        self.vocab_path = vocab_path
        self.output_unit = output_unit

    def __len__(self):
        return len(self.vocab_dict)

    def label_to_string(self, labels):
        """
        Converts label to string (number => Hangeul)

        Args:
            labels (numpy.ndarray): number label

        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """
        if len(labels.shape) == 1:
            sentence = str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                  continue
                sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                  continue
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def load_vocab(self, label_path, encoding='utf-8'):
        """
        Provides char2id, id2char

        Args:
            label_path (str): csv file with character labels
            encoding (str): encoding method

        Returns: unit2id, id2unit
            - **unit2id** (dict): unit2id[unit] = id
            - **id2unit** (dict): id2unit[id] = unit
        """
        unit2id = dict()
        id2unit = dict()

        try:
            with open(label_path, 'r', encoding=encoding) as f:
                labels = csv.reader(f, delimiter=',')
                next(labels)

                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]

                unit2id['<blank>'] = len(unit2id)
                id2unit[len(unit2id)] = '<blank>'

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(label_path))