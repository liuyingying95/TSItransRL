from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from tqdm import tqdm
import pandas as pd
import torch

from eval_script import valid_metric
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#tokenizer = GPT2TokenizerFast.from_pretrained("entropy/gpt2_zinc_87m", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('entropy/gpt2_zinc_87m')
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
#tokenizer = GPT2TokenizerFast.from_pretrained("./liuyingdistill6", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./liuyingdistill6')
#tokenizer = GPT2TokenizerFast.from_pretrained("./YingGPTModel3", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./YingGPTModel3')
#tokenizer = GPT2TokenizerFast.from_pretrained("./YingGPTModel8", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./YingGPTModel8')
#tokenizer = GPT2TokenizerFast.from_pretrained("./YingGPTModel9", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./YingGPTModel9')
# jnk_gsk task
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
#tokenizer = GPT2TokenizerFast.from_pretrained("./jnk_gsk_res/liuyingdistill6", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./jnk_gsk_res/liuyingdistill6')
#tokenizer = GPT2TokenizerFast.from_pretrained("./jnk_gsk_res/liuyingdistill7", max_len=256)
#model = GPT2LMHeadModel.from_pretrained('./jnk_gsk_res/liuyingdistill7')
tokenizer = GPT2TokenizerFast.from_pretrained("./jnk_gsk_res/YingGPTModel", max_len=256)
model = GPT2LMHeadModel.from_pretrained('./jnk_gsk_res/YingGPTModel')

#import ipdb;ipdb.set_trace()


inputs = torch.tensor([[tokenizer.bos_token_id]]).cuda()
inputs


model.cuda()

smiles = []
with torch.no_grad():
    for i in tqdm(range(20)):
        #import ipdb;ipdb.set_trace()
        gen = model.generate(inputs, do_sample=True, max_length=256, 
                     temperature=.8, early_stopping=True, pad_token_id=tokenizer.pad_token_id,
                     num_return_sequences=500)
        #gen = model.generate(inputs, min_length=-1, top_k=0.0, top_p=1.0,
        #        do_sample=True, max_length=256, pad_token_id=tokenizer.pad_token_id,
        #        num_return_sequences=1000)
        gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
        smiles.extend(gen)

smiles = pd.DataFrame({"SMILES": smiles})
#smiles.to_csv("./smiles.csv", index=False)
#smiles.to_csv("./smiles2.csv", index=False)
#smiles.to_csv("./smiles3.csv", index=False)
#smiles.to_csv("./smiles_for_scaffold.csv", index=False)
# jnk_gsk task
#smiles.to_csv("./jnk_gsk_res/smilesdistill.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/smilesdistill2.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/smilesdistill3.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/smilesdistill4.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/smilesdistill5.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/smilesdistill6.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/smilesdistill7.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/lunwenres/smilesdistillinit.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/lunwenres/smilesdistilltsiinit.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/lunwenres/studentinit.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/lunwenres/student2.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/lunwenres/student7.csv", index=False)
#smiles.to_csv("./jnk_gsk_res/lunwenres/studentgpt.csv", index=False)
smiles.to_csv("./jnk_gsk_res/lunwenres/for_scaffold_analysis.csv", index=False)

#gen_csv = "./smiles.csv"
#gen_csv = "./smiles2.csv"
#gen_csv = "./smiles3.csv"
#gen_csv = "./smiles_for_scaffold.csv"
# jnk_gsk task
#gen_csv = "./jnk_gsk_res/smilesdistill.csv"
#gen_csv = "./jnk_gsk_res/smilesdistill2.csv"
#gen_csv = "./jnk_gsk_res/smilesdistill3.csv"
#gen_csv = "./jnk_gsk_res/smilesdistill4.csv"
#gen_csv = "./jnk_gsk_res/smilesdistill5.csv"
#gen_csv = "./jnk_gsk_res/smilesdistill6.csv"
#gen_csv = "./jnk_gsk_res/smilesdistill7.csv"
#gen_csv = "./jnk_gsk_res/lunwenres/smilesdistilltsiinit.csv"
#gen_csv = "./jnk_gsk_res/lunwenres/studentinit.csv"
#gen_csv = "./jnk_gsk_res/lunwenres/student7.csv"
#gen_csv = "./jnk_gsk_res/lunwenres/studentgpt.csv"
gen_csv = "./jnk_gsk_res/lunwenres/for_scaffold_analysis.csv"
n_jobs = 10
#ref_path = "./drd2data/drd2_activity.smi"
# jnk_gsk task
ref_path = "./drd2data/jnk_gsk/jnk_gsk_activity.csv"
#gen_smiles_csv = "./gen_smiles.csv"
#gen_smiles_csv = "./gen_smiles2.csv"
#gen_smiles_csv = "./gen_smiles_init_S.csv"
#gen_smiles_csv = "./gen_smiles_init_S1_2.csv"
#gen_smiles_csv = "./gen_smiles_S1_3.csv"
#gen_smiles_csv = "./gen_smiles_S2_1.csv"
#gen_smiles_csv = "./gen_smiles_S2_2.csv"
#gen_smiles_csv = "./gen_smiles_S2_3.csv"
#gen_smiles_csv = "./gen_smiles_S3_1.csv"
#gen_smiles_csv = "./gen_smiles_S3_2.csv"
#gen_smiles_csv = "./gen_smiles_S3_3.csv"
#gen_smiles_csv = "./gen_smiles_init_TS.csv"
#gen_smiles_csv = "./gen_smiles_init_TS.csv"
#gen_smiles_csv = "./gen_smiles_init_S1.csv"
#gen_smiles_csv = "./gen_smiles_init_S4.csv"
#gen_smiles_csv = "./gen_smiles_S4_2.csv"
#gen_smiles_csv = "./gen_smiles_S4_3.csv"
#gen_smiles_csv = "./gen_smiles_S4_4.csv"
#gen_smiles_csv = "./gen_smiles_S5_1.csv"
#gen_smiles_csv = "./gen_smiles_S5_2.csv"
#gen_smiles_csv = "./gen_smiles_S5_3.csv"
#gen_smiles_csv = "./gen_smiles_S6_1.csv"
#gen_smiles_csv = "./gen_smiles_S6_2.csv"
#gen_smiles_csv = "./gen_smiles_S6_3.csv"
#gen_smiles_csv = "./gen_smiles_init_S7.csv"
#gen_smiles_csv = "./gen_smiles_S7_2.csv"
#gen_smiles_csv = "./gen_smiles_S7_3.csv"
#gen_smiles_csv = "./gen_smiles_S7_4.csv"
#gen_smiles_csv = "./gen_smiles_match.csv"
#gen_smiles_csv = "./rl_gen.csv"
#gen_smiles_csv = "./plots/rl_gen1.csv"
#gen_smiles_csv = "./plots/rl_gen_for_scaffold.csv"
# jnk_gsk task
#gen_smiles_csv = "./jnk_gsk_res/jnk_gsk_total_new2.csv"
#gen_smiles_csv = "./jnk_gsk_res/jnk_gsk_total_new3.csv"
#gen_smiles_csv = "./jnk_gsk_res/jnk_gsk_total_new4.csv"
#gen_smiles_csv = "./jnk_gsk_res/jnk_gsk_total_new5.csv"
#gen_smiles_csv = "./jnk_gsk_res/jnk_gsk_total_new6.csv"
#gen_smiles_csv = "./jnk_gsk_res/jnk_gsk_total_new7.csv"
#gen_smiles_csv = "./jnk_gsk_res/jnk_gsk_total_new8.csv"
#gen_smiles_csv = "./jnk_gsk_res/lunwenres/jnk_gsk_studentinit.csv"
#gen_smiles_csv = "./jnk_gsk_res/lunwenres/jnk_gsk_tsiinit.csv"
#gen_smiles_csv = "./jnk_gsk_res/lunwenres/jnk_gsk_sinit.csv"
#gen_smiles_csv = "./jnk_gsk_res/lunwenres/jnk_gsk_s7.csv"
#gen_smiles_csv = "./jnk_gsk_res/lunwenres/jnk_gsk_gpt.csv"
gen_smiles_csv = "./jnk_gsk_res/lunwenres/gsk_scaffold_analysis.csv"
#res_path = "./res.txt"
# jnk_gsk task
res_path = "./jnk_gsk_res/res.txt"

valid_metric(gen_csv, n_jobs, ref_path, gen_smiles_csv, res_path)
