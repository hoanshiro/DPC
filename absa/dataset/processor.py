import itertools
import os
import json
import numpy as np
import torch
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import sys
from ..config.config import Config
from pyvi import ViTokenizer
from vncorenlp import VnCoreNLP



class Processor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = None
        self.vocab = None
        self.wv = None

    def load_tokenizer(self):
        is_load_success = False
        if os.path.exists(self.cfg["vn_core_path"]):
            self.tokenizer = VnCoreNLP(self.cfg["vn_core_path"], annotators="wseg", max_heap_size='-Xmx500m')
            is_load_success = True
        else:
            self.tokenizer = None
        if self.tokenizer is None:
            self.tokenizer = ViTokenizer
            self.cfg["token_type"] = "ViTokenizer"
            is_load_success = True
        return is_load_success

    def load_embedding_model(self):
        file_name = self.cfg['word2vec_model_path']
        model = KeyedVectors.load_word2vec_format(file_name, binary=True)
        self.wv = model
        return True

    def word_segment(self, raw: str):
        if self.cfg["token_type"] == "RDRSEGMENTER":
            sentences = self.tokenizer.tokenize(raw)
            return list(itertools.chain(*sentences))
        else:
            return self.tokenizer.tokenize(raw).split()

    def word2vec(self, list_token):
        sentence = []
        for token in list_token:
            try:
                sentence = sentence + self.wv[token].ravel().tolist()
            except:
                sentence += np.zeros(100).tolist()
        return sentence

    def label2idx(self, list_tag, label_type="ASPECT"):
        tag_ids = [self.cfg[label_type][tag] for tag in list_tag]
        return tag_ids


def load_data(processor, data_path):
    if not os.path.exists(data_path):
        is_load_success = False
    print("data_path: Path not exist {}".format(data_path))
    dataset = []
    with open(data_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            raw_text = data['text']
            norm_text = []
            aspect_tags = []
            polarity_tags = []
            last_index = 0
            for span in data["labels"]:
                as_tag, senti_tag = span[-1].split("#")
                # Add prefix tokens
                prefix_span_text = processor.word_segment(raw_text[last_index:span[0]])
                norm_text.extend(prefix_span_text)
                aspect_tags.extend(["O"] * len(prefix_span_text))
                polarity_tags.extend(["O"] * len(prefix_span_text))
                aspect_text = processor.word_segment(raw_text[span[0]:span[1]])
                for idx, _ in enumerate(aspect_text):
                    if idx == 0:
                        aspect_tags.append(f"B-{as_tag.strip()}")
                        polarity_tags.append(f"B-{senti_tag.strip()}")
                        continue
                    aspect_tags.append(f"I-{as_tag.strip()}")
                    polarity_tags.append(f"I-{senti_tag.strip()}")
                norm_text.extend(aspect_text)
                last_index = span[1]
            last_span_text = processor.word_segment(raw_text[last_index:])
            norm_text.extend(last_span_text)
            aspect_tags.extend(["O"] * len(last_span_text))
            polarity_tags.extend(["O"] * len(last_span_text))
            assert len(norm_text) == len(aspect_tags), f"Not match: {line}"
            embed = processor.word2vec(norm_text)
            dataset.append({"tokens": norm_text, 'embed': embed, "a_tags": aspect_tags, "p_tags": polarity_tags})
        is_load_success = True
    return dataset


if __name__=="__main__":
    config = Config()
    config.load_config()
    config = config.cfg
    processor = Processor(config)
    processor.load_tokenizer()
    processor.load_embedding_model()
    data_path = config["dataset"]["path"]["train"]
    raw_data = load_data(processor, data_path)
    print(raw_data[0])
