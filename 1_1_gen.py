#!/usr/bin/env python
# coding=utf-8
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from moses.metrics import fraction_valid
from moses.metrics.metrics import novelty, fraction_unique, remove_invalid
from moses.dataset import get_dataset, get_statistics
from moses.metrics.metrics import SNNMetric, FragMetric, internal_diversity
from moses.metrics.metrics import compute_intermediate_statistics, FCDMetric
from moses.metrics.metrics import fraction_passes_filters, ScafMetric
from moses.metrics.metrics import WassersteinMetric
from moses.utils import mapper, get_mol
from moses.metrics.utils import canonic_smiles
from moses.metrics.utils import logP, QED, SA, weight
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import RDLogger, Chem

from multiprocessing import Pool

from utils import get_scoring_func, multi_score_funcs_one_hot
from transformers import GPT2TokenizerFast
from model import MoleGPT
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

RDLogger.DisableLog("rdApp.*")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_conf = {"d_model": 128, "nhead": 8, "num_decoder_layers": 12,
            "dim_feedforward": 512, "max_seq_length": 140,
            "pos_dropout": 0.1, "trans_dropout": 0.1}


def unique_smiles(gen, n_jobs=1, check_validity=True):
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return canonic


class BaseMetric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)
    def __call__(self, gen=None, pgen=None):
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pgen is None:
            pgen = self.precalc(gen)
        return pgen

    def precalc(self, moleclues):
        return NotImplementedError


class PropMetric(BaseMetric):
    def __init__(self, func=None, **kwargs):
        self.func = func
        super().__init__(**kwargs)

    def precalc(self, mols):
        if self.func is not None:
            values = mapper(self.n_jobs)(self.func, mols)
        else:
            values = mols
        mean_value = np.mean(values)
        std_value = np.std(values)
        return {"mean": mean_value, "std": std_value}, values


class property_drd2_func:
    def __call__(self, smiles_list):
        qeds = []
        sas = []
        logps = []
        smiles = []
        drd2_func = get_scoring_func("drd2")
        for sm in smiles_list:
            mol = Chem.MolFromSmiles(sm)
            smiles.append(sm)
            if mol is None:
                qeds.append(0.0)
                sas.append(100.0)
                logps.append(100.0)
            else:
                try:
                    qed = QED(mol)
                    sa = SA(mol)
                    logp = logP(mol)
                except:
                    qed = 0.0
                    sa = 100.0
                    logp = 100.0
                qeds.append(qed)
                sas.append(sa)
                logps.append(logp)

        drd2s = drd2_func(smiles_list)

        df = pd.DataFrame({"smiles": smiles, "qed": qeds,
                           "sa": sas, "drd2": drd2s, "logp": logps})
        return df


class property_jnk_gsk_func:
    def __call__(self, smiles_list):
        qeds = []
        sas = []
        logps = []
        smiles = []
        jnk3_func = get_scoring_func("jnk3")
        gsk3_func = get_scoring_func("gsk3")
        for sm in smiles_list:
            mol = Chem.MolFromSmiles(sm)
            smiles.append(sm)
            if mol is None:
                qeds.append(0.0)
                sas.append(100.0)
                logps.append(100.0)
            else:
                try:
                    qed = QED(mol)
                    sa = SA(mol)
                    logp = logP(mol)
                except:
                    qed = 0.0
                    sa = 100.0
                    logp = 100.0
                qeds.append(qed)
                sas.append(sa)
                logps.append(logp)

        jnk3s = jnk3_func(smiles_list)
        gsk3s = gsk3_func(smiles_list)

        df = pd.DataFrame({"smiles": smiles, "qed": qeds,
            "sa": sas, "jnk3": jnk3s, "gsk3": gsk3s, "logp": logps})
        return df


@torch.no_grad()
def gen_metric(n_to_generate, model_path, batch_size, n_jobs, ref_path, gen_smiles_csv, res_path):
    gen_model_conf = {
            "d_model": 128, "nhead": 8, "num_decoder_layers": 12,
            "dim_feedforward": 512, "max_seq_length": 256,
            "pos_dropout": 0.1, "trans_dropout": 0.1}
    voc = GPT2TokenizerFast.from_pretrained("./liuyingv3", max_len=256)
    model = MoleGPT(voc, gen_model_conf["d_model"], gen_model_conf["nhead"],
                gen_model_conf["num_decoder_layers"], gen_model_conf["dim_feedforward"],
                gen_model_conf["max_seq_length"], gen_model_conf["pos_dropout"],
                gen_model_conf["trans_dropout"])
    model.decoder.eval()
    model.decoder.load_state_dict(torch.load(model_path))
    model.decoder.cuda()
    smiles = []
    nlls = []

    for i in tqdm(range(n_to_generate)):
        #seqs = model.generate(batch_size, max_length=256, con_token_list=["<s>positive[SEP]"])
        seqs, nll = model.sample(batch_size, max_length=256, con_token_list=["<s>positive[SEP]"])
        nlls.extend(nll.tolist())
        gen = voc.batch_decode(seqs, skip_special_tokens=True)
        smiles.extend(gen)

    nllsdf = pd.DataFrame({"Average NLL": nlls})

    sns.kdeplot(data=nllsdf, x="Average NLL")
    plt.savefig("tmp.png")
    nllsdf.to_csv("average_nll_teacher_prior3.csv", index=False)
    
    #smiles_df = pd.read_csv(gen_csv)
    #smiles = smiles_df["SMILES"].tolist()
    frac_valid = fraction_valid(smiles, n_jobs=6)
    print("Generated SMILES:", len(smiles))
    print("valid SMILES proportion:", frac_valid)

    smiles_valid = remove_invalid(smiles, canonize=True)

    #with open(ref_path) as f:
    #    next(f)
    #    train_smiles = [line.split('\n')[0] for line in f]
    #print('number of active reference', len(train_smiles))
    # jnk_gsk task
    with open(ref_path) as f:
        train_smiles = [line.split('\n')[0] for line in f]
    print('number of active reference', len(train_smiles))

    frac_novelty = novelty(smiles_valid, train_smiles, n_jobs=n_jobs)
    print("novel SMILES proportion:", frac_novelty)

    frac_unique = fraction_unique(smiles_valid, k=1000, n_jobs=n_jobs)
    print("SMILES unique@1000:", frac_novelty)

    close_pool = False
    if n_jobs != 1:
        pool = Pool(n_jobs)
        close_pool = True
    else:
        pool = 1

    batch_size = 500
    ptest = compute_intermediate_statistics(train_smiles,
                                            n_jobs=n_jobs,
                                            device=device,
                                            batch_size=batch_size,
                                            pool=pool)
    
    mols = mapper(pool)(get_mol, smiles_valid)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}

    FCD_Test = FCDMetric(**kwargs_fcd)(gen=smiles_valid, pref=ptest['FCD'])

    SNN_Test = SNNMetric(**kwargs)(gen=mols, pref=ptest['SNN'])

    Frag_Test = FragMetric(**kwargs)(gen=mols, pref=ptest['Frag'])

    IntDiv = internal_diversity(mols, pool, device=device)
    IntDiv2 = internal_diversity(mols, pool, device=device, p=2)

    Filters = fraction_passes_filters(mols, pool)

    Scaf_Test = ScafMetric(**kwargs)(gen=mols, pref=ptest['Scaf'])

    # Properties
    property_metrics = {}
    property_sim = {}
    for_success = {}
    for name, func in [('logP', logP), ('SA', SA),
                       ('QED', QED),
                       ('weight', weight)]:
        property_metrics[name], values = PropMetric(func, **kwargs)(
                        gen=mols)
        if name in ["QED", "SA"]:
            for_success[name] = values
        property_sim[name] = WassersteinMetric(func, **kwargs)(
                        gen=mols, pref=ptest[name]
        )

    if close_pool:
        pool.close()
        pool.join()

    print("SMILES SNN:", SNN_Test)
    print("SMILES Frag:", Frag_Test)
    print("SMILES IntDiv:", IntDiv)
    print("SMILES IntDiv2:", IntDiv2)
    print("SMILES Filters:", Filters)
    print("SMILES Scaf:", Scaf_Test)
    print("SMILES FCD:", FCD_Test)
    print("SMILES Property metric:", property_metrics)
    print("SMILES Property sim:", property_sim)

    #df = property_drd2_func()(smiles_valid)
    # jnk_gsk task
    df = property_jnk_gsk_func()(smiles_valid)
    #df_success = df[(df["qed"]>=0.6) & (df["sa"]<=4) & (df["drd2"] >=0.5)]
    # jnk_gsk task
    df_success = df[(df["qed"]>=0.6) & (df["sa"]<=4) & (df["jnk3"] >=0.5) & (df["gsk3"] >= 0.5)]
    df_unique_sms = df.drop_duplicates()
    #df_real_success = df_unique_sms[
    #    (df_unique_sms["qed"]>=0.6) & (df_unique_sms["sa"]<=4) & (df_unique_sms["drd2"] >=0.5)]
    # jnk_gsk task
    df_real_success = df_unique_sms[
        (df_unique_sms["qed"]>=0.6) & (df_unique_sms["sa"]<=4) & (df_unique_sms["jnk3"] >=0.5) & (df_unique_sms["gsk3"] >= 0.5)]
    #print("Success:", len(df_success)/len(smiles)
    print("Success:", len(df_success)/len(smiles))
    #print("Real Success:", len(df_real_success)/len(smiles)
    print("Real Success:", len(df_real_success)/len(smiles))
    df.to_csv(gen_smiles_csv, index=False)
    #df_unique_sms.to_csv("teacher_init.csv", index=False)
    # jnk_gsk task
    #df_real_success.to_csv("./jnk_gsk_res/for_gpt_1.csv", index=False)
    #df_real_success.to_csv("./jnk_gsk_res/for_gpt_2.csv", index=False)
    #df_real_success.to_csv("./jnk_gsk_res/for_gpt_3.csv", index=False)
    #df_real_success.to_csv("./jnk_gsk_res/for_gpt_4.csv", index=False)
    #df_real_success.to_csv("./jnk_gsk_res/for_gpt_5.csv", index=False)
    #df_real_success.to_csv("./jnk_gsk_res/for_gpt_6.csv", index=False)
    df_real_success.to_csv("./jnk_gsk_res/for_gpt_7.csv", index=False)
    #df_unique_sms.to_csv("gen_for_rl_unique2.csv", index=False)
    #df_unique_sms.to_csv("gen_for_rl_unique3.csv", index=False)
    #df_unique_sms.to_csv("gen_for_rl_unique4.csv", index=False)
    #df_unique_sms.to_csv("gen_for_rl_unique5.csv", index=False)
    #df_unique_sms.to_csv("gen_for_rl_unique6.csv", index=False)
    #df_unique_sms.to_csv("gen_for_rl_unique7.csv", index=False)
    #df_real_success.to_csv("./gen_for_T7_scaffold.csv", index=False)
    # jnk_gsk task
    df_real_success.to_csv("./jnk_gsk_res/gen_for_T1_scaffold.csv", index=False)

    res = {"SMILES SNN": SNN_Test, "SMILES Frag": Frag_Test,
           "SMILES IntDiv": IntDiv, "SMILES IntDiv2": IntDiv2,
           "SMILES Filters": Filters, "SMILES Scaf": Scaf_Test,
           "SMILES FCD": FCD_Test, "SMILES Property metric": property_metrics,
           "SMILES Property sim": property_sim,
           #"Success": len(df_success)/len(smiles),
           "Success": len(df_success)/len(smiles),
           #"Real Success": len(df_real_success)/len(smiles),
           "Real Success": len(df_real_success)/len(smiles),
           }

    with open(res_path, "w") as f:
        f.write(str(res))


if __name__ == "__main__":
    n_jobs = 10
    #ref_path = "./drd2data/drd2_activity.smi"
    # jnk_gsk task
    ref_path = "./drd2data/jnk_gsk/jnk_gsk_activity.csv"
    #gen_smiles_csv = "./gen_smiles4.csv"
    #gen_smiles_csv = "./gen_smiles_init_T.csv"
    #gen_smiles_csv = "./gen_smiles_init_T2.csv"
    #gen_smiles_csv = "./gen_smiles_init_T3.csv"
    #gen_smiles_csv = "./gen_smiles_init_T4.csv"
    #gen_smiles_csv = "./gen_smiles_init_T5.csv"
    #gen_smiles_csv = "./gen_smiles_init_T6.csv"
    #gen_smiles_csv = "./gen_smiles_init_T7.csv"
    #gen_smiles_csv = "./gen_smiles_init_T2_1.csv"
    #gen_smiles_csv = "./gen_smiles_init_T2_2.csv"
    #gen_smiles_csv = "./gen_smiles_init_T2_3.csv"
    #gen_smiles_csv = "./gen_smiles_init_T3_1.csv"
    #gen_smiles_csv = "./gen_smiles_init_T3_2.csv"
    #gen_smiles_csv = "./gen_smiles_init_T3_3.csv"
    #gen_smiles_csv = "./gen_smiles_init_T4_1.csv"
    #gen_smiles_csv = "./gen_smiles_init_T4_2.csv"
    #gen_smiles_csv = "./gen_smiles_init_T4_3.csv"
    #gen_smiles_csv = "./gen_smiles_init_T5_1.csv"
    #gen_smiles_csv = "./gen_smiles_init_T5_2.csv"
    #gen_smiles_csv = "./gen_smiles_init_T5_3.csv"
    #gen_smiles_csv = "./gen_smiles_init_T6_1.csv"
    #gen_smiles_csv = "./gen_smiles_init_T6_2.csv"
    #gen_smiles_csv = "./gen_smiles_init_T6_3.csv"
    #gen_smiles_csv = "./gen_smiles_init_T7_1.csv"
    #gen_smiles_csv = "./gen_smiles_init_T7_2.csv"
    #gen_smiles_csv = "./gen_smiles_init_T7_3.csv"
    #gen_smiles_csv = "./gen_smiles_T7_scaffold.csv"
    # jnk_gsk task
    #gen_smiles_csv = "./jnk_gsk_res/gen_smiles_T1_scaffold.csv"
    #gen_smiles_csv = "./jnk_gsk_res/lunwenres/gen_smiles_Tinit.csv"
    gen_smiles_csv = "./jnk_gsk_res/lunwenres/gen_smiles_Tinit2.csv"
    #gen_smiles_csv = "./jnk_gsk_res/lunwenres/gen_smiles_T2.csv"
    #gen_smiles_csv = "./jnk_gsk_res/lunwenres/gen_smiles_T3.csv"
    #gen_smiles_csv = "./jnk_gsk_res/lunwenres/gen_smiles_T4.csv"
    #gen_smiles_csv = "./jnk_gsk_res/lunwenres/gen_smiles_T5.csv"
    #gen_smiles_csv = "./jnk_gsk_res/lunwenres/gen_smiles_T6.csv"
    #gen_smiles_csv = "./jnk_gsk_res/lunwenres/gen_smiles_T7.csv"
    #res_path = "./res4.txt"
    # jnk_gsk task
    res_path = "./jnk_gsk_res/res1.txt"
    n_to_generate = 500
    #n_to_generate = 5000
    #model_path = "./drdres/molegpt_init.ckpt"
    #model_path = "./drdres/molegpt.ckpt"
    #model_path = "./drdres/molegpt3.ckpt"
    #model_path = "./drdres/molegpt4.ckpt"
    #model_path = "./drdres/molegpt5.ckpt"
    #model_path = "./drdres/molegpt6.ckpt"
    #model_path = "./drdres/molegpt7.ckpt"
    # jnk_gsk task
    model_path = "./jnk_gsk_res/molegpt_init_jnk_gsk.ckpt"
    #model_path = "./jnk_gsk_res/molegpt_jnk_gsk2.ckpt"
    #model_path = "./jnk_gsk_res/molegpt_jnk_gsk3.ckpt"
    #model_path = "./jnk_gsk_res/molegpt_jnk_gsk4.ckpt"
    #model_path = "./jnk_gsk_res/molegpt_jnk_gsk5.ckpt"
    #model_path = "./jnk_gsk_res/molegpt_jnk_gsk6.ckpt"
    #model_path = "./jnk_gsk_res/molegpt_jnk_gsk7.ckpt"
    batch_size = 10
    #batch_size = 20
    #batch_size = 128 # drd2 128
    #valid_metric(n_jobs, ref_path, gen_smiles_csv, res_path)
    gen_metric(n_to_generate, model_path, batch_size, n_jobs, ref_path, gen_smiles_csv, res_path)

