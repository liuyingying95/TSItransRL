#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import math


class MoleGPT:

    def __init__(self, tokenizer, d_model, nhead, num_decoder_layers,
                 dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
        self.decoder = Decoder(len(tokenizer.vocab), d_model, nhead,
                               num_decoder_layers, dim_feedforward,
                               max_seq_length, pos_dropout, trans_dropout)
        self.decoder.cuda()
        self.tokenizer = tokenizer
        self._nll_loss = nn.NLLLoss(ignore_index=0, reduction="none")

    def likelihood(self, target):
        batch_size, seq_length = target.size()
        con_token = target[:, 0:3]
        x = torch.cat((target[:, 0].unsqueeze(1), target[:, 3:-1]), 1)
        y_target = target[:, 3:].contiguous().view(-1)
        seq_length = seq_length - len(con_token[0])

        logits = prior_train_forward(self.decoder, x, con_token)
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        log_probs = criterion(logits.view(-1, len(self.tokenizer.vocab)), y_target)
        
        mean_log_probs = log_probs.mean()
        log_probs = log_probs.view(-1, seq_length)
        log_probs_each_molecule = log_probs.sum(dim=1)
        return mean_log_probs, log_probs_each_molecule
    
    @torch.no_grad()
    def sample(self, batch_size, max_length=256, con_token_list=["<s>positive[SEP]"]):
        con_token_list = self.tokenizer(con_token_list)["input_ids"][0]
        con_tokens = torch.zeros(batch_size, len(con_token_list)).long().cuda()
        for ind, token in enumerate(con_token_list):
            con_tokens[:, ind] = token
        start_token = torch.zeros(batch_size, 1).long().cuda()
        start_token[:] = self.tokenizer.vocab["<s>"]
        input_vector = start_token
        sequences = start_token
        log_probs = torch.zeros(batch_size)
        finished = torch.zeros(batch_size).byte().cuda()
        for step in range(max_length):
            logits = sample_forward_model(self.decoder, input_vector, con_tokens)
            logits_step = logits[:, step, :]
            prob = F.softmax(logits_step, dim=1)
            log_prob = F.log_softmax(logits_step, dim=1)
            input_vector = torch.multinomial(prob, 1)
            sequences = torch.cat((sequences, input_vector), 1)
            log_probs += self._nll_loss(log_prob, input_vector.view(-1)).cpu()
            EOS_sampled = (input_vector.view(-1) == self.tokenizer.vocab["</s>"]).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                break

            input_vector = sequences

        return sequences[:, 1:].data, log_probs

    def generate(self, batch_size, max_length=256, con_token_list=["<s>positive[SEP]"]):
        con_token_list = self.tokenizer(con_token_list)["input_ids"][0]
        con_tokens = torch.zeros(batch_size, len(con_token_list)).long().cuda()
        for ind, token in enumerate(con_token_list):
            con_tokens[:, ind] = token
        start_token = torch.zeros(batch_size, 1).long().cuda()
        start_token[:] = self.tokenizer.vocab["<s>"]
        input_vector = start_token
        sequences = start_token
        finished = torch.zeros(batch_size).byte().cuda()

        for step in range(max_length):
            logits = sample_forward_model(self.decoder, input_vector, con_tokens)
            logits_step = logits[:, step, :]
            prob = F.softmax(logits_step, dim=1)
            log_prob = F.log_softmax(logits_step, dim=1)
            input_vector = torch.multinomial(prob, 1)
            sequences = torch.cat((sequences, input_vector), 1)
            EOS_sampled = (input_vector.view(-1) == self.tokenizer.vocab["</s>"]).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                break

            input_vector = sequences

        return sequences[:, 1:].data

        
class Decoder(nn.Module):

    def __init__(self, vocab_size, d_model, nhead,
                 num_decoder_layers, dim_feedforward,
                 max_seq_length, pos_dropout,
                 trans_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnedPositionEncoding(d_model, pos_dropout,
                                               max_seq_length)
        decoder_layers = GPTDecoderLayer(d_model, nhead,
                                         dim_feedforward, trans_dropout)
        self.transformer_encoder = GPTDecoder(decoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, con_token, tgt_key_padding_mask, tgt_mask):
        con_token = con_token.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt = self.pos_enc(
            (self.embed_tgt(tgt) + self.embed_tgt(con_token).sum(dim=0).unsqueeze(0)) * math.sqrt(self.d_model)
        )
        output = self.transformer_encoder(
            tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        output = output.transpose(0, 1)
        return self.fc(output)


class LearnedPositionEncoding(nn.Embedding):

    def __init__(self, d_model, dropout=0.1, max_len=256):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        a = weight[:x.size(0), :]
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class GPTDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512,
                 dropout=0.1, activation="relu"):
        super(GPTDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(GPTDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


class GPTDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(GPTDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_activation_fn(act):
    if act == "relu":
        return F.relu
    elif act == "gelu":
        return F.gelu


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def prior_train_forward(model, tgt, con_token):
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    output = model(tgt, con_token, tgt_key_padding_mask=None,
                   tgt_mask=tgt_mask)
    del tgt, tgt_mask
    return output.squeeze(1)


def gen_nopeek_mask(length):
    mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask.cuda()


class ScheduledOptim:

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        ])

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


def sample_forward_model(model, tgt, con_token):
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    output = model(tgt, con_token, tgt_key_padding_mask=None,
                   tgt_mask=tgt_mask)
    return output


