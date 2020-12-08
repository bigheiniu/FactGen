#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import time
import numpy as np
from itertools import count

import torch
from itertools import repeat

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.decoders.ensemble
from onmt.translate.beam_search import BeamSearch
from onmt.translate.random_sampling import RandomSampling
from onmt.utils.misc import tile, set_random_seed
from onmt.modules.copy_generator import collapse_copy_scores


def build_translator(opt, report_score=False, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')
        # out_file = codecs.open("./result.output", 'w+', 'utf-8')
    
    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)
    
    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)
    
    translator = Translator.from_opt(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_score=report_score,
        logger=logger
    )
    return translator


class Translator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        fields (dict[str, torchtext.data.Field]): A dict
            mapping each side to its list of name-Field pairs.
        src_reader (onmt.inputters.DataReaderBase): Source reader.
        tgt_reader (onmt.inputters.TextDataReader): Target reader.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.random_sampling.RandomSampling`.
        random_sampling_temp (int): See
            :class:`onmt.translate.random_sampling.RandomSampling`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_bleu (bool): Print/log Bleu metric.
        report_rouge (bool): Print/log Rouge metric.
        report_time (bool): Print/log total time/frequency.
        copy_attn (bool): Use copy attention.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """
    
    def __init__(
            self,
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=-1,
            n_best=1,
            min_length=0,
            max_length=100,
            beam_size=30,
            random_sampling_topk=1,
            random_sampling_temp=1,
            stepwise_penalty=None,
            dump_beam=False,
            block_ngram_repeat=0,
            ignore_when_blocking=frozenset(),
            replace_unk=False,
            data_type="text",
            verbose=False,
            report_bleu=False,
            report_rouge=False,
            report_time=False,
            copy_attn=False,
            simple_fusion=False,
            gpt_tgt=False,
            global_scorer=None,
            out_file=None,
            report_score=True,
            logger=None,
            seed=-1):
        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        # self._tgt_bos_idx = self._tgt_vocab.stoi['Ġsee']
        print(self._tgt_bos_idx)
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)
        
        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) \
            if self._use_cuda else torch.device("cpu")
        
        self.n_best = n_best
        self.max_length = max_length
        
        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking}
        self.src_reader = src_reader
        self.tgt_reader = tgt_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError(
                "replace_unk requires an attentional decoder.")
        self.data_type = data_type
        self.verbose = verbose
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.report_time = report_time
        
        self.copy_attn = copy_attn
        self.simple_fusion = simple_fusion
        self.gpt_tgt = gpt_tgt
        
        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and \
                not self.model.decoder.attentional:
            raise ValueError(
                "Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.report_score = report_score
        self.logger = logger
        
        self.use_filter_pred = False
        self._filter_pred = None
        
        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}
        
        set_random_seed(seed, self._use_cuda)
    
    @classmethod
    def from_opt(
            cls,
            model,
            fields,
            opt,
            model_opt,
            global_scorer=None,
            out_file=None,
            report_score=True,
            logger=None):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """
        
        if opt.data_type == 'none':
            src_reader = None
        else:
            src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)
        return cls(
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            dump_beam=opt.dump_beam,
            block_ngram_repeat=opt.block_ngram_repeat,
            ignore_when_blocking=set(opt.ignore_when_blocking),
            replace_unk=opt.replace_unk,
            data_type=opt.data_type,
            verbose=opt.verbose,
            report_bleu=opt.report_bleu,
            report_rouge=opt.report_rouge,
            report_time=opt.report_time,
            copy_attn=model_opt.copy_attn,
            simple_fusion=model_opt.simple_fusion,
            gpt_tgt=model_opt.GPT_representation_mode != 'none' and model_opt.GPT_representation_loc in ['tgt', 'both'],
            global_scorer=global_scorer,
            out_file=out_file,
            report_score=report_score,
            logger=logger,
            seed=opt.seed)
    
    def _gold_score(self, batch, memory_bank, src_lengths, src_vocabs,
                    use_src_map, enc_states, batch_size, src):
        if "tgt" in batch.__dict__:
            gs = self._score_target(
                batch, memory_bank, src_lengths, src_vocabs,
                batch.src_map if use_src_map else None)
            self.model.decoder.init_state(src, memory_bank, enc_states)
            if self.simple_fusion:
                self.model.lm_decoder.init_state(src, None, None)
        else:
            gs = [0] * batch_size
        return gs
    
    def build_data_iter(self,
                        opt,
                        batch_size=1
                        ):
        if batch_size is None:
            raise ValueError("batch_size must be set")

        
        def read_file(path):
            priv_str = "r"
            priv_str += "b"
            with open(path, priv_str) as f:
                return f.readlines()
        src = read_file(opt.src)
        tgt = read_file(opt.tgt)  if opt.tgt is not None else None
        
        
        readers, data, dirs = [], [], []
        if self.src_reader:
            readers += [self.src_reader]
            data += [("src", src)]
            dirs += [None]
        if tgt:
            readers += [self.tgt_reader]
            data += [("tgt", tgt)]
            dirs += [None]
    
        data = inputters.Dataset(
            self.fields,
            readers=readers,
            data=data,
            dirs=dirs,
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred
        )
    
        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )
        
        self.src_vocabs = data.src_vocabs
        
        self._build_xlation(data, tgt)
        self.all_scores = []
        self.all_predictions = []

        self.pred_score_total, self.pred_words_total = 0, 0
        self.gold_score_total, self.gold_words_total = 0, 0
        self.counter = count(1)
        return data_iter
        
        
    def set_encoder_state(self,
                          batch,
                          tags=None,
                          temperature=1.0
                          ):
    
        if self.beam_size != 1:
            self.beam_size = 1
        if self.block_ngram_repeat != 0:
            self.block_ngram_repeat = 0
    
        # Encoder forward.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)
        if self.simple_fusion:
            self.model.lm_decoder.init_state(src, None, None)
    
        use_src_map = self.copy_attn
        memory_lengths = src_lengths
        src_map = batch.src_map if use_src_map else None
        
        # set for the decoder usage
        self.enc_states = enc_states
        self.src = src
        self.src_lengths = src_lengths
        self.memory_bank = memory_bank
        self.memory_lengths = memory_lengths
        self.src_map = src_map
        self.batch = batch
        self.batch_size = batch.batch_size
        self.set_random_sampler(temperature=temperature)
        self._build_result()
        

    def forward_pass(
            self,
            decoder_in,
            step,
            past=None,
            input_embeds=None,
            tags=None,
            use_copy=True
            
    ):
        memory_bank = self.memory_bank
        src_vocabs = self.src_vocabs
        memory_lengths = self.memory_lengths
        src_map = self.src_map
        batch = self.batch
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )
            
        decoder = self.model.decoder
        

        dec_out, all_hidden_states, past, dec_attn = decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step, past=past, input_embeds=input_embeds, pplm_return=True
        )
        
        # Generator forward.
        if not self.copy_attn:
            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None
        
            if self.simple_fusion:
                lm_dec_out, _ = self.model.lm_decoder(decoder_in, memory_bank.new_zeros(1, 1, 1), step=step)
                probs = self.model.generator(dec_out.squeeze(0), lm_dec_out.squeeze(0))
            else:
                probs = self.model.generator(dec_out.squeeze(0))
                # print(log_probs)
                # returns [(batch_size x beam_size) , vocab ] when 1 step
                # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
        
            scores, p_copy = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                             attn.view(-1, attn.size(2)),
                                             src_map, tags=tags)
            
    
            scores = scores.view(batch.batch_size, -1, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=None
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            # log_probs = scores.squeeze(0).log()
            probs = scores.squeeze(0)
            if use_copy is False:
                probs = probs[:, :50257]
                return probs, attn, all_hidden_states, past
            return probs, attn, all_hidden_states, past, p_copy
        return probs, attn, all_hidden_states, past
        
    def set_random_sampler(self,
                           return_attention=False,
                           temperature=1.0
                           ):
        if isinstance(self.memory_bank, tuple) or isinstance(self.memory_bank, list):
            if isinstance(self.memory_bank[0], dict):
                mb_device = self.memory_bank[0][list(self.memory_bank[0].keys())[0]].device
            else:
                mb_device = self.memory_bank[0].device
        else:
            mb_device = self.memory_bank.device

        if self.max_length < 400:
            self.max_length = 400
        if self.min_length < 300:
            self.min_length = 300
        self.random_sampler = RandomSampling(
            self._tgt_pad_idx, self._tgt_bos_idx, self._tgt_eos_idx,
            self.batch_size, mb_device, self.min_length, self.block_ngram_repeat,
            self._exclusion_idxs, return_attention, self.max_length,
            temperature, self.sample_from_topk, self.memory_lengths)
        
    def generate_tokens(self, log_probs, attn=None):
        
        self.random_sampler.advance(log_probs, attn)
        any_batch_is_finished = self.random_sampler.is_finished.any()
        # ATTENTION: The batch size is one
        if any_batch_is_finished:
            self.random_sampler.update_finished()
            if self.random_sampler.done:
                # Finish the generation, set the resulf for generation
                self._finalize_result()
                self.generate_sentence_batch()
                return False
        else:
            return self.random_sampler.alive_seq[:, -1].view(1, 1)
    
    def generate_sentence_batch(self):
        batch = self.results
        translations = self.xlation_builder.from_batch(batch)
        for trans in translations:
            self.all_scores += [trans.pred_scores[:self.n_best]]
            self.pred_score_total += trans.pred_scores[0]
            self.pred_words_total += len(trans.pred_sents[0])
    
            n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:self.n_best]]
            self.all_predictions += [n_best_preds]
            self.out_file.write('\n'.join(n_best_preds) + '\n')
            self.out_file.flush()
    
            if self.verbose:
                sent_number = next(self.counter)
                output = trans.log(sent_number)
                if self.logger:
                    self.logger.info(output)
                else:
                    os.write(1, output.encode('utf-8'))
        
        
            
    def _build_result(self):
        use_src_map = self.copy_attn
        self.results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": self.batch,
            "gold_score": self._gold_score(
                self.batch, self.memory_bank, self.src_lengths, self.src_vocabs, use_src_map,
                self.enc_states, self.batch_size, self.src)}
        
    def _build_xlation(self, data, tgt):
        self.xlation_builder = onmt.translate.TranslationBuilder(
            data, self.fields, self.n_best, self.replace_unk, tgt
        )
        
    
    def _finalize_result(self):
        self.results["scores"] = self.random_sampler.scores
        self.results["predictions"] = self.random_sampler.predictions
        self.results["attention"] = self.random_sampler.attention
        
    def _run_encoder(self, batch):
        if hasattr(batch, 'src'):
            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            
            enc_states, memory_bank, src_lengths = self.model.encoder(
                src, src_lengths)
            if src_lengths is None:
                assert not isinstance(memory_bank, tuple), \
                    'Ensemble decoding only supported for text data'
                src_lengths = torch.Tensor(batch.batch_size) \
                    .type_as(memory_bank) \
                    .long() \
                    .fill_(memory_bank.size(0))
        else:
            src = None
            enc_states = None
            memory_bank = torch.zeros((1, batch.tgt[0].shape[1], 1), dtype=torch.float, device=batch.tgt[0].device)
            src_lengths = torch.ones((batch.tgt[0].shape[1],), dtype=torch.long, device=batch.tgt[0].device)
            # src_lengths = None
        return src, enc_states, memory_bank, src_lengths
    
    def freeze_parameter(self):
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad=False
        for parameter in self.model.decoder.parameters():
            parameter.requires_grad=False
        for parameter in self.model.generator.parameters():
            parameter.requires_grad=False
    

