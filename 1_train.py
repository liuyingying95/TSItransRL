#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from model import MoleGPT, ScheduledOptim
from data import myDataset
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader
from torch.optim import Adam
from pytorchtools import EarlyStopping
from tqdm import tqdm
from data import get_dict

def train(train_data, valid_data, batch_size, d_model,
          nhead, num_decoder_layers, dim_feedforward, n_warmup_steps,
          max_seq_length, pos_dropout, trans_dropout,
          num_epochs, save_path):
    tokenizer = GPT2TokenizerFast.from_pretrained("./liuyingv3", max_len=256)
    train_data = get_dict(train_data)
    valid_data = get_dict(valid_data)
    train_ds = myDataset(train_data, tokenizer)
    valid_ds = myDataset(valid_data, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=12,
                              collate_fn=myDataset.collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                              num_workers=12,
                              shuffle=False, drop_last=True)
    net = MoleGPT(tokenizer, d_model, nhead, num_decoder_layers,
                  dim_feedforward, max_seq_length, pos_dropout, trans_dropout)

    optim = ScheduledOptim(
        Adam(net.decoder.parameters(), betas=(0.9, 0.98), eps=1e-09),
        d_model * 8, n_warmup_steps
    )
    train_losses, val_losses = pre_train(train_loader, valid_loader,
                                         net, optim, num_epochs, save_path)
    torch.cuda.empty_cache()
    

def pre_train(train_data, valid_data, model, optim,
              num_epochs, save_path):
    #model.decoder.load_state_dict(torch.load("./drdres/molegpt.ckpt"))
    #model.decoder.load_state_dict(torch.load("./drdres/molegpt3.ckpt"))
    #model.decoder.load_state_dict(torch.load("./drdres/molegpt4.ckpt"))
    #model.decoder.load_state_dict(torch.load("./drdres/molegpt5.ckpt"))
    #model.decoder.load_state_dict(torch.load("./drdres/molegpt6.ckpt"))
    # jnk_gsk task
    #model.decoder.load_state_dict(torch.load("./jnk_gsk_res/molegpt_jnk_gsk2.ckpt"))
    #model.decoder.load_state_dict(torch.load("./jnk_gsk_res/molegpt_jnk_gsk3.ckpt"))
    #model.decoder.load_state_dict(torch.load("./jnk_gsk_res/molegpt_jnk_gsk4.ckpt"))
    #model.decoder.load_state_dict(torch.load("./jnk_gsk_res/molegpt_jnk_gsk5.ckpt"))
    model.decoder.load_state_dict(torch.load("./jnk_gsk_res/molegpt_jnk_gsk6.ckpt"))
    model.decoder.cuda()
    model.decoder.train()
    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0

    early_stopping = EarlyStopping(patience=5, verbose=False)

    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in tqdm(enumerate(train_data), total=len(train_data)):
            seqs = batch.long().cuda()
            loss, each_molecule_loss = model.likelihood(seqs)
            optim.zero_grad()
            loss.backward()
            optim.step_and_update_lr()
            total_loss += loss.item()

            if step % 200 == 0 and step != 0:
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}    step {:3d}    loss: {:5.2f}\n".format(
                    epoch, step, loss.data
                ))

        print("average epoch loss:", total_loss / len(train_data))
        val_loss = validate(valid_data, model)
        val_losses.append((total_step, val_loss))

        early_stopping(val_loss, model.decoder, "Ying1")

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model.decoder.state_dict(), save_path)

        print(f"Val Loss: {val_loss}")

    return train_losses, val_losses


def validate(valid_data, model):
    model.decoder.cuda()
    model.decoder.eval()
    total_loss = 0

    for step, batch in tqdm(enumerate(valid_data), total=len(valid_data)):
        with torch.no_grad():
            seqs = batch.long().cuda()
            loss, each_molecule_loss = model.likelihood(seqs)
            total_loss += loss.item()

    return total_loss / len(valid_data)


if __name__ == "__main__":
    max_seq_length = 256
    d_model = 128
    num_decoder_layers = 12
    dim_feedforward = 512
    nhead = 8
    pos_dropout = 0.1
    trans_dropout = 0.1
    n_warmup_steps = 500
    num_epoch = 100
    #batch_size = 256
    batch_size = 128
    #train_path = "./drd2data/drd_train.csv"
    #train_path = "./drd2data/drd_train_new2.csv"
    #train_path = "./drd2data/drd_train_new3.csv"
    #train_path = "./drd2data/drd_train_new4.csv"
    #train_path = "./drd2data/drd_train_new5.csv"
    #train_path = "./drd2data/drd_train_new6.csv"
    #train_path = "./drd2data/drd_train_new7.csv"
    # jnk_gsk3 task
    #train_path = "./drd2data/jnk_gsk/jnk_gsk_train.csv"
    #train_path = "./jnk_gsk_res/jnk_gsk_train_new2.csv"
    #train_path = "./jnk_gsk_res/jnk_gsk_train_new3.csv"
    #train_path = "./jnk_gsk_res/jnk_gsk_train_new4.csv"
    #train_path = "./jnk_gsk_res/jnk_gsk_train_new5.csv"
    #train_path = "./jnk_gsk_res/jnk_gsk_train_new6.csv"
    train_path = "./jnk_gsk_res/jnk_gsk_train_new7.csv"
    #valid_path = "./drd2data/drd_test.csv"
    #valid_path = "./drd2data/drd_test_new2.csv"
    #valid_path = "./drd2data/drd_test_new3.csv"
    #valid_path = "./drd2data/drd_test_new4.csv"
    #valid_path = "./drd2data/drd_test_new5.csv"
    #valid_path = "./drd2data/drd_test_new6.csv"
    #valid_path = "./drd2data/drd_test_new7.csv"
    # jnk_gsk3 task
    #valid_path = "./drd2data/jnk_gsk/jnk_gsk_test.csv"
    #valid_path = "./jnk_gsk_res/jnk_gsk_test_new2.csv"
    #valid_path = "./jnk_gsk_res/jnk_gsk_test_new3.csv"
    #valid_path = "./jnk_gsk_res/jnk_gsk_test_new4.csv"
    #valid_path = "./jnk_gsk_res/jnk_gsk_test_new5.csv"
    #valid_path = "./jnk_gsk_res/jnk_gsk_test_new6.csv"
    valid_path = "./jnk_gsk_res/jnk_gsk_test_new7.csv"
    #save_path = "./drdres/molegpt_init.ckpt"
    #save_path = "./drdres/molegpt.ckpt"
    #save_path = "./drdres/molegpt3.ckpt"
    #save_path = "./drdres/molegpt4.ckpt"
    #save_path = "./drdres/molegpt5.ckpt"
    #save_path = "./drdres/molegpt6.ckpt"
    #save_path = "./drdres/molegpt7.ckpt"
    # jnk_gsk3 task
    #save_path = "./jnk_gsk_res/molegpt_init_jnk_gsk.ckpt"
    #save_path = "./jnk_gsk_res/molegpt_jnk_gsk2.ckpt"
    #save_path = "./jnk_gsk_res/molegpt_jnk_gsk3.ckpt"
    #save_path = "./jnk_gsk_res/molegpt_jnk_gsk4.ckpt"
    #save_path = "./jnk_gsk_res/molegpt_jnk_gsk5.ckpt"
    #save_path = "./jnk_gsk_res/molegpt_jnk_gsk6.ckpt"
    save_path = "./jnk_gsk_res/molegpt_jnk_gsk7.ckpt"

    train(train_path, valid_path, batch_size, d_model, nhead,
          num_decoder_layers, dim_feedforward, n_warmup_steps,
          max_seq_length, pos_dropout, trans_dropout, num_epoch,
          save_path)
