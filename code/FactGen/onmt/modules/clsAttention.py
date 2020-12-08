import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig, AdamW, get_linear_schedule_with_warmup, GPT2Tokenizer
import pytorch_lightning as pl
import logging
from torch.utils.data import Dataset, DataLoader
logger = logging.getLogger(__name__)
import os
import pandas as pd

class SequenceSummary(nn.Module):
     def __init__(self, hidden_size=768, num_labels=2, dropout_ratio=0.1):
         super(SequenceSummary, self).__init__()
         self.act = nn.Tanh()
         self.dropout = nn.Dropout(dropout_ratio)
         self.clf_head = nn.Linear(hidden_size, num_labels)
         self.loss_fn = nn.CrossEntropyLoss()
     def forward(self, labels=None, hidden=None):
         hidden = self.clf_head(hidden)
         hidden = self.act(hidden)
         if labels is None:
            predict = torch.argmax(hidden, dim=-1)
            return predict
         else:
            loss = self.loss_fn(hidden, labels)
            return loss


class FactReconstructor(nn.Module):
    def __init__(self, word_embedding=None):
        super(FactReconstructor, self).__init__()
        # tiny bert for masked language
        config = BertConfig(num_hidden_layers=2, num_attention_heads=4, intermediate_size=256, pad_token_id=50163, vocab_size=50257)
        self.bert = BertForMaskedLM(config)
        if word_embedding is not None:
            self.bert.set_input_embeddings(word_embedding)
        self.word_embeddings = self.bert.get_input_embeddings()


    def forward(self, input_ids, cls_hidden=None):
        # other_hidden_list batch_size * k * 30
        # mask several tokens from the encoder
        # TODO: mask the name entity
        # padding_idx 50163, unkown_index 49968
        input_ids.squeeze_(-1)
        input_ids = torch.transpose(input_ids, 1, 0)
        not_include_mask = torch.where((input_ids == 50163) * (input_ids == 49968), torch.ones_like(input_ids), torch.zeros_like(input_ids))

        mask_prob = (torch.FloatTensor(input_ids.shape).uniform_() > 0.8).to(input_ids.device)
        # Also Get rid of the padded index
        attention_mask = torch.where(mask_prob, torch.ones_like(input_ids).float(), torch.zeros_like(input_ids).float())
        attention_mask = torch.where(not_include_mask == 1, torch.zeros_like(attention_mask), attention_mask)

        input_embeddings = self.word_embeddings(input_ids)
        masked_lm_labels = torch.where(attention_mask > 0, input_ids, input_ids.new_ones(input_ids.shape) * -100)
        if cls_hidden is not None:
            cls_hidden = cls_hidden.unsqueeze(1)

            input_embeddings = torch.cat([cls_hidden, input_embeddings], dim=1)
            attention_mask = torch.cat([attention_mask.new_ones(attention_mask.shape[0], 1), attention_mask], dim=1)
            masked_lm_labels = torch.cat(
                [masked_lm_labels.new_ones(masked_lm_labels.shape[0], 1) * -100, masked_lm_labels], dim=1)


        return  self.bert(inputs_embeds=input_embeddings, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels)[0]



class clsDataset(Dataset):
    def __init__(self, file_path, overwrite_cache=False):
        catched_path = file_path+".torch"
        if os.path.exists(catched_path) and overwrite_cache is False:
            self.example = torch.load(catched_path)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if "tsv" in file_path:
                data = pd.read_csv(file_path, sep="\t")
                data = data['title'].values.tolist()
            else:
                data = open(file_path,'r').readlines()
            data = [i.strip() for i in data]
            data = [tokenizer.encode(i) for i in data]
            # pad idx is 50163
            data = [i[:100] if len(i) > 100 else i + [50163] * (100-len(i)) for i in data ]
            self.example = data
            torch.save(
                data, catched_path)

    def __len__(self):
        return len(self.example)

    def __getitem__(self, item):
        return torch.tensor(self.example[item])



class clsAttenTrain(pl.LightningModule):
    def __init__(self, hparams):
        super(clsAttenTrain, self).__init__()
        self.hparams = hparams
        # cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        # load the vocab file from the encoder and the embeddings from other file
        self.model = FactReconstructor()

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def load_dataset(self, type, batch_size):
        if type == "trian":
            shuffle = True
        else:
            shuffle = False

        file_path = self.hparams.file_path.format(type)

        dataset = clsDataset(file_path)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return data_loader


    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            second_order_closure = None,
    ):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        avg_loss = getattr(self.trainer, "avg_loss", 0.0)
        tqdm_dict = {"loss": "{:.3f}".format(avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def train_dataloader(self):
        train_batch_size = self.hparams.train_batch_size
        dataloader = self.load_dataset("train", train_batch_size)

        t_total = (
            (len(dataloader.dataset) // (train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        return self.load_dataset("val", self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.load_dataset("test", self.hparams.eval_batch_size)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default="fact_clf",
            type=str,
            # required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument("--file_path",type=str)
        parser.add_argument("--overwrite_cache", action='store_true')



        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=1000, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument(
            "--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform."
        )

        parser.add_argument("--train_batch_size", default=100, type=int)
        parser.add_argument("--eval_batch_size", default=100, type=int)
        return parser



def add_generic_args(parser, root_dir):
    parser.add_argument(
        "--output_dir",
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--n_tpu_cores", type=int, default=0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("-seed", type=int, default=42, help="random seed for initialization")



def generic_train(model, args):
    # init model
    #
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback
    )

    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_gpu > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(**train_params)

    # if args.do_train:
    trainer.fit(model)

    return trainer


def trainFactReconstructor(args):
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}", )
        os.makedirs(args.output_dir)
    model = clsAttenTrain(args)
    trainer = generic_train(model, args)


if __name__ == '__main__':
    import argparse
    import time
    from collections import OrderedDict
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = clsAttenTrain.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    trainFactReconstructor(args)

    # Extract the model statedict from the checkpoint
    file_path = "../checkpointcheckpoint_ckpt_epoch_4.ckpt"
    state_dic = torch.load(file_path)['state_dict']
    state_dic = OrderedDict([(key.replace("model.",""), value) for key, value in state_dic.items()])
    torch.save(state_dic,"../fact_clf_4.torch")