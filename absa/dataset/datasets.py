import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from dataset.utils import load_data
from config.config import Config

Config = Config()
Config.load_config()
cfg = Config.cfg


class UitAbsaDataset(Dataset):
    def __init__(self, data_type="train"):
        self.dataset = load_data(cfg["dataset"]["path"][data_type])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.aspect_labels = cfg["ASPECT_LABELS"]
        self.polarity_labels = cfg["POLARITY_LABELS"]
        self.max_seq_len = cfg["dataset"]["max_seq_len"]
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        feature = self.convert_example_to_features(sample)
        feature = {key: torch.as_tensor(val, dtype=torch.long).to(self.device) for key, val in feature.items()}
        return feature

    def convert_example_to_features(self, example, use_crf: bool = True):
        data = json.loads(example)
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
        
        a_tag_ids = [self.aspect_labels.index(a_tag) for a_tag in aspect_tags]
        p_tag_ids = [self.polarity_labels.index(p_tag) for p_tag in polarity_tags]
        if self.max_seq_len >= seq_len:
            label_padding_size = (self.max_seq_len - seq_len)
            norm_text.extend(['_'] * label_padding_size)
            a_tag_ids.extend([0] * label_padding_size)
            p_tag_ids.extend([0] * label_padding_size)
        else:
            a_tag_ids = a_tag_ids[:self.max_seq_len]
            p_tag_ids = p_tag_ids[:self.max_seq_len]
            norm_text = norm_text[:self.max_seq_len]
            
        item['a_labels'] = a_tag_ids
        item['p_labels'] = p_tag_ids
        item['embed'] = processor.word2vec(norm_text)
        return item


if __name__ == "__main__":
    dataset = UitAbsaDataset(data_type="dev")
    print(dataset[9])
