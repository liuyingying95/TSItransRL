"""
_author: Liuying Wang
_date: 2023/05/23
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import rdkit.Chem.QED as QED
from rdkit.Chem import rdchem
from rdkit import Chem
from rdkit import DataStructs
#from dgl.dataloading import DataLoader
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import roc_auc_score
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch

import os
from collections import defaultdict
import random
import json
import pickle
import math


_fscores = None


def readFragmentScores(name="fpscores"):
    import gzip
    global _fscores
    if name == "fpscores":
        name = os.path.join(os.path.dirname(__file__), name)
    _fscores = pickle.load(gzip.open("%s.pkl.gz" % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def load_json_config(path):
    return json.load(open(path, "r"))

def rdchem_enum_to_list(values):
    return [values[i] for i in range(len(values))]


def index_tool(alist, elem):
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist) - 1


def unique(arr):
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

    
def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    if nf == 0:
        nf = 1
    score1 /= nf

    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.

    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


def canonical_smiles(smiles, sanitize=True, throw_warning=False):
    new_smiles = []
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            new_smiles.append(Chem.MolToSmiles(mol))
        except:
            if throw_warning:
                warnings.warn(sm + " can not be canonized: invalid SMILES string",
                              UserWarning)
                new_smiles.append('')
    return new_smiles


def valid_smile_proportion(voc, generated_smiles):
    sanitized = canonical_smiles(generated_smiles, sanitize=False)
    if '' in sanitized:
        unique_smiles = list(np.unique(sanitized))[1:]
    else:
        unique_smiles = list(np.unique(sanitized))
    print("Proportion of valid SMILES:", len(unique_smiles) / len(generated_smiles))


class drd2_model:
    """scores based on edfp classifier for activity"""
    #clf_path = "../data/drd2/drd2.pkl"
    clf_path = "./drd2data/drd2.pkl"

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append(int(mol is not None))
            fp = drd2_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


class jnk3_model():
    """Scores based on an ECFP classifier for activity."""
    kwargs = ["clf_path"]
    clf_path = './drd2data/jnk_gsk/jnk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append(int(mol is not None))
            fp = jnk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)
    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


class gsk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = './drd2data/jnk_gsk/gsk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append(int(mol is not None))
            fp = gsk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


class qed_func:
    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0)
            else:
                try:
                    qed = QED.qed(mol)
                except:
                    qed = 0
                scores.append(qed)
        return np.float32(scores)


class sa_func:
    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                scores.append(calculateScore(mol))
        return np.float32(scores)


def get_scoring_func(prop_name):
    if prop_name == "drd2":
        return drd2_model()
    elif prop_name == "jnk3":
        return jnk3_model()
    elif prop_name == "gsk3":
        return gsk3_model()
    elif prop_name == "qed":
        return qed_func()
    elif prop_name == "sa":
        return sa_func()


def multi_score_funcs_one_hot(data, func_list):
    funcs = [get_scoring_func(prop) for prop in func_list]
    props = np.array([func(data) for func in funcs])
    props = pd.DataFrame(props).T
    props.columns = func_list
    scoring_sum = condition_convert(props).values.sum(1)
    return scoring_sum

 

#def condition_convert(con_df):
    # convert to 0, 1
#    con_df['drd2'][con_df['drd2'] >= 0.5] = 1
#    con_df['drd2'][con_df['drd2'] < 0.5] = 0
#    con_df['qed'][con_df['qed'] >= 0.6] = 1
#    con_df['qed'][con_df['qed'] < 0.6] = 0
#    con_df['sa'][con_df['sa'] <= 4.0] = 1
#    con_df['sa'][con_df['sa'] > 4.0] = 0
#    return con_df


def condition_convert(con_df):
    # convert to 0, 1
    con_df['jnk3'][con_df['jnk3'] >= 0.5] = 1
    con_df['jnk3'][con_df['jnk3'] < 0.5] = 0
    con_df['gsk3'][con_df['gsk3'] >= 0.5] = 1
    con_df['gsk3'][con_df['gsk3'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df


# if __name__ == "__main__":
    # smiles = "OCc1ccccc1CN"
    # # smiles = r"[H]/[NH+]=C(\N)C1=CC(=O)/C(=C\C=c2ccc(=C(N)[NH3+])cc2)C=C1"
    # mol = AllChem.MolFromSmiles(smiles)
    # print(len(smiles))
    # print(mol)
    # data = mol_to_geognn_graph_data_MMFF3d(mol)
    # dataset_name = "bbbp"
    # data_path = "../data/finetune_datasets/bbbp"
    # dataset = build_dataset(dataset_name, data_path)
