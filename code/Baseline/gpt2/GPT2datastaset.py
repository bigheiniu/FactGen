import torch
from torch.utils.data import Dataset
import pickle
import os
import pandas as pd

class GPT2Dataset(Dataset):
    def __init__(self, args, tokenizer, block_size, file_path, mask_task=True):
        super(GPT2Dataset, self).__init__()
        self.mask_task = mask_task
        self.mask_token = "<mask>"
        self.pad_token = "<pad>"
        self.mask_pro = 0.8
        self.tokenizer = tokenizer
        self.max_length = block_size
        cached_features_file = file_path.format("all").replace("txt",'pickle')
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            # logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file+"_mask" if mask_task else "", "rb") as handle:
                result = pickle.load(handle)
                for key, value in result.items():
                    setattr(self, key, value)
        else:
            # logger.info("Creating features from dataset file at %s", cached_features_file)
            src_list = pd.read_csv(file_path, sep="\t")
            src_list = src_list.values.tolist()


            self.examples = []
            self.labels = []
            for src in src_list:

                encode_all, labels = self.mask_handle(src)
                self.examples.append(encode_all)
                self.labels.append(labels)

            # logger.info("Saving features into cached file %s", cached_features_file)
            print("there are {} samples in {}".format(len(self.examples), file_path))
            with open(cached_features_file+"_mask" if mask_task else "" , "wb") as handle:
                result = {"examples": self.examples,
                          "labels": self.labels}
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def mask_handle(self, src_line):

        _, prompt, _, fact = src_line
        prompt = prompt.replace("<t>", "").replace("<\t>", "")
        fact = fact.replace("<t>", "").replace("<\t>", "")
        # mask can be N * 3
        # BERT Mask: Random Mask Part of elements in the sequence

        inputs = torch.ones(len(fact.split("|")), 3)
        padded = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(padded.shape, self.mask_pro)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        padded[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(padded.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = 0
        inputs = inputs.tolist()
        # inputs is the mask and labels is the mask tokens and the pad
        fact_list = [i.split(",") for i in fact.split("|")]
        fact_masked_list = []
        labels = []
        for fact_entry, mask in zip(fact_list, inputs):
            fact_mask = []
            padfact = []
            for entry, index in zip(fact_entry, mask):
                label_entry = " ".join(len(self.tokenizer.encode(entry)) * [self.pad_token])
                if index == 0.:
                    label_entry = entry
                    entry = " ".join(len(self.tokenizer.encode(entry)) * [self.mask_token])

                fact_mask.append(entry)
                padfact.append(label_entry)
            fact_masked_list.append(fact_mask)
            labels.append(" ".join(padfact))


        input_encode = self.tokenizer.encode(prompt + " <c-begin> " + " ".join([" ".join(i) for i in fact_masked_list]), pad_to_max_length=True, max_length=self.max_length)
        labels = " ".join(labels)
        prompt_encode_len = len(self.tokenizer.encode(prompt + " <c-begin> "))
        labels = " ".join(prompt_encode_len * [self.pad_token]) + " " + labels
        label_encode = self.tokenizer.encode(labels, max_length=self.max_length, pad_to_max_length=True)
        return input_encode, label_encode




    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return_tuple = (
            torch.tensor(self.examples[item]), torch.tensor(self.labels[item]))
        return return_tuple


