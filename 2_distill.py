from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer

from data import Distilldata


#tokenizer = GPT2TokenizerFast.from_pretrained("entropy/gpt2_zinc_87m", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('entropy/gpt2_zinc_87m')
# distill
#tokenizer = GPT2TokenizerFast.from_pretrained("./liuyingdistill", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./liuyingdistill')
#tokenizer = GPT2TokenizerFast.from_pretrained("./liuyingdistill2", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./liuyingdistill2')
#tokenizer = GPT2TokenizerFast.from_pretrained("./liuyingdistill3", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./liuyingdistill3')
#tokenizer = GPT2TokenizerFast.from_pretrained("./liuyingdistill4", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./liuyingdistill4')
#tokenizer = GPT2TokenizerFast.from_pretrained("./liuyingdistill5", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./liuyingdistill5')
# jnk_gsk task
#tokenizer = GPT2TokenizerFast.from_pretrained("entropy/gpt2_zinc_87m", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('entropy/gpt2_zinc_87m')
#tokenizer = GPT2TokenizerFast.from_pretrained("./jnk_gsk_res/liuyingdistill", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./jnk_gsk_res/liuyingdistill')
#tokenizer = GPT2TokenizerFast.from_pretrained("./jnk_gsk_res/liuyingdistill2", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./jnk_gsk_res/liuyingdistill2')
#tokenizer = GPT2TokenizerFast.from_pretrained("./jnk_gsk_res/liuyingdistill3", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./jnk_gsk_res/liuyingdistill3')
#tokenizer = GPT2TokenizerFast.from_pretrained("./jnk_gsk_res/liuyingdistill4", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./jnk_gsk_res/liuyingdistill4')
#tokenizer = GPT2TokenizerFast.from_pretrained("./jnk_gsk_res/liuyingdistill5", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./jnk_gsk_res/liuyingdistill5')
tokenizer = GPT2TokenizerFast.from_pretrained("./jnk_gsk_res/liuyingdistill6", max_len=256)
model = GPT2LMHeadModel.from_pretrained('./jnk_gsk_res/liuyingdistill6')

tokenizer.add_special_tokens({"sep_token":"[SEP]"})
model.resize_token_embeddings(len(tokenizer))


#df = pd.read_csv("./drdres/gen_smiles_unique.csv")
#df = pd.read_csv("./drdres/for_gpt_2.csv")
#df = pd.read_csv("./drdres/for_gpt_3.csv")
#df = pd.read_csv("./drdres/for_gpt_4.csv")
#df = pd.read_csv("./drdres/for_gpt_5.csv")
#df = pd.read_csv("./drdres/for_gpt_6.csv")
# jnk_gsk task
#df = pd.read_csv("./jnk_gsk_res/for_gpt_1.csv")
#df = pd.read_csv("./jnk_gsk_res/for_gpt_2.csv")
#df = pd.read_csv("./jnk_gsk_res/for_gpt_3.csv")
#df = pd.read_csv("./jnk_gsk_res/for_gpt_4.csv")
#df = pd.read_csv("./jnk_gsk_res/for_gpt_5.csv")
#df = pd.read_csv("./jnk_gsk_res/for_gpt_6.csv")
df = pd.read_csv("./jnk_gsk_res/for_gpt_7.csv")
df_train, df_val = train_test_split(df, test_size=0.1, random_state=1)

data_train = dict()
id = 1
for row in df_train.itertuples():
    t0 = time.time()
    smiles = getattr(row, "smiles")
    data_train[id] = [smiles]

    id += 1
    print(f"Number of train smiles: {len(data_train) :,}")

data_val = dict()
id = 1
for row in df_val.itertuples():
    t0 = time.time()
    smiles = getattr(row, "smiles")
    data_val[id] = [smiles]

    id += 1
    print(f"Number of val smiles: {len(data_val) :,}")

train_dataset = Distilldata(data_train, tokenizer)
val_dataset = Distilldata(data_val, tokenizer)

UNFREEZE_LAST_N = 6

# - Freeze selective layers:
# - Freeze all layers except last n:
for parameter in model.parameters():
    parameter.requires_grad = False
for i, m in enumerate(model.transformer.h):        
#Only un-freeze the last n transformer blocks
    if i+1 > 12 - UNFREEZE_LAST_N:
        for parameter in m.parameters():
            parameter.requires_grad = True 
        
for parameter in model.transformer.ln_f.parameters():
    parameter.requires_grad = True

for parameter in model.lm_head.parameters():        
    parameter.requires_grad = True


APEX_OPT_LEVEL  = 'O1'
EPOCHS          = 10
LR              = 5e-4
EPS             = 1e-8
WARMUP_STEPS    = 1e2
TRAIN_BATCHSIZE = 32
BATCH_UPDATE    = 16
#output_dir = "./liuyingdistill/"
#output_dir = "./liuyingdistill2/"
#output_dir = "./liuyingdistill3/"
#output_dir = "./liuyingdistill4/"
#output_dir = "./liuyingdistill5/"
#output_dir = "./liuyingdistill6/"
# jnk_gsk task
#output_dir = "./jnk_gsk_res/liuyingdistill/"
#output_dir = "./jnk_gsk_res/liuyingdistill2/"
#output_dir = "./jnk_gsk_res/liuyingdistill3/"
#output_dir = "./jnk_gsk_res/liuyingdistill4/"
#output_dir = "./jnk_gsk_res/liuyingdistill5/"
#output_dir = "./jnk_gsk_res/liuyingdistill6/"
output_dir = "./jnk_gsk_res/liuyingdistill7/"


training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=TRAIN_BATCHSIZE,
            per_device_eval_batch_size=TRAIN_BATCHSIZE,
            gradient_accumulation_steps=BATCH_UPDATE,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            fp16_opt_level=APEX_OPT_LEVEL,
            warmup_steps=WARMUP_STEPS,    
            learning_rate=LR,
            adam_epsilon=EPS,
            weight_decay=0.01,        
            save_total_limit=1,
            load_best_model_at_end=True)

#---------------------------------------------------#
trainer = Trainer(
            model=model,
            args=training_args,    
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer)

#---------------------------------------------------#
trainer.train()
trainer.save_model()
