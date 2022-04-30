from rdkit.Chem import Descriptors, Crippen
from multiprocessing import Process
from rdkit.Chem import rdmolops
from rdkit.Chem import Draw
import pubchempy as pcp
from rdkit import Chem
from tqdm import tqdm
import json

data = {}
TotalCompounds = 163000000
TotalIteration = 163183509
MaxFileEntry = 10000000
FileCount = 0


def WriteToFile(data, filecount):
    with open(f"./data/data_filter_{filecount}.json", 'w+') as f:
        json.dump(data, f, indent=4)


def GetBiggestFragment(molecule):
    molFrag = rdmolops.GetMolFrags(molecule, asMols=True)
    LargestMolSmile = max(molFrag, default=molecule,
                          key=lambda m: m.GetNumAtoms())
    return LargestMolSmile


def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts(
        "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def GetNumHydrogen(molecule):
    moleculewH = Chem.AddHs(molecule)
    NumAtoms = moleculewH.GetNumAtoms()
    NumHeavyAtom = moleculewH.GetNumHeavyAtoms()
    NumHydrogen = NumAtoms-NumHeavyAtom
    return NumHydrogen


def GetNumCarbon(molecule):
    NumCarbon = 0
    CarbonAtomNumber = 6
    for atom in molecule.GetAtoms():
        if(atom.GetAtomicNum() == CarbonAtomNumber):
            NumCarbon += 1
    return NumCarbon


if __name__ == '__main__':
    with tqdm(total=163183509) as pbar:
        with open("./CID-SMILES", 'r') as f:
            for line in f:
                line_data = (line.strip()).split('\t')
                i = int(line_data[0])
                smile = line_data[1]
                try:
                    molecule = Chem.MolFromSmiles(smile)
                    molecule = GetBiggestFragment(molecule)
                    molecule = neutralize_atoms(molecule)
                    molecular_weight = Descriptors.ExactMolWt(molecule)
                    num_heavy_atoms = molecule.GetNumHeavyAtoms()
                    num_hydrogen = GetNumHydrogen(molecule)
                    num_carbon = GetNumCarbon(molecule)
                    logp = Crippen.MolLogP(molecule)
                    if ((molecular_weight > 12) and (molecular_weight < 600) and (logp > -7) and (logp < 5) and (num_heavy_atoms > 3)
                            and (smile not in data) and num_carbon > 0 and num_hydrogen > 0):
                        data[smile] = i
                except:
                    pass
                pbar.update()
                if(i % MaxFileEntry == 0 or i == (TotalIteration-1)):
                    FileCount += 1
                    WriteToFile(data, FileCount)
                    data = {}
