import os
import itertools
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
import copy
import networkx as nx
import math
import random
import time
import csv
from contextlib import contextmanager
import sys
from mflow.utils.sascorer import calculateScore
"""
Environment Usage
plogp: penalized_logp(mol)
logp: MolLogP(mol)
qed: qed(mol)
sa: calculateScore(mol)
mw: rdMolDescriptors.CalcExactMolWt(mol)
"""


def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:
        return m
    else:
        print('converting radical electrons to H')
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def check_chemical_validity(mol):
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified. Radicals pass this test.
    :return: True if chemically valid, False otherwise
    """
    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)
    if m:
        return True
    else:
        return False


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.
            SANITIZE_PROPERTIES)
        return True
    except ValueError:
        return False


def get_final_smiles(mol):
    """
    Returns a SMILES of the final molecule. Converts any radical
    electrons into hydrogens. Works only if molecule is valid
    :return: SMILES
    """
    m = convert_radical_electrons_to_hydrogens(mol)
    return Chem.MolToSmiles(m, isomericSmiles=True)


def get_final_mol(mol):
    """
    Returns a rdkit mol object of the final molecule. Converts any radical
    electrons into hydrogens. Works only if molecule is valid
    :return: SMILES
    """
    m = convert_radical_electrons_to_hydrogens(mol)
    return m


def add_hydrogen(mol):
    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    return Chem.MolFromSmiles(s)


def calculate_min_plogp(mol):
    p1 = penalized_logp(mol)
    s1 = Chem.MolToSmiles(mol, isomericSmiles=True)
    s2 = Chem.MolToSmiles(mol, isomericSmiles=False)
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s2)
    p2 = penalized_logp(mol1)
    p3 = penalized_logp(mol2)
    final_p = min(p1, p2)
    final_p = min(final_p, p3)
    return final_p


def penalized_logp(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455
    log_p = Chem.Descriptors.MolLogP(mol)
    SA = -calculateScore(mol)
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        )
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle


def steric_strain_filter(mol, cutoff=0.82, max_attempts_embed=20,
    max_num_iters=200):
    """
    Flags molecules based on a steric energy cutoff after max_num_iters
    iterations of MMFF94 forcefield minimization. Cutoff is based on average
    angle bend strain energy of molecule
    :param mol: rdkit mol object
    :param cutoff: kcal/mol per angle . If minimized energy is above this
    threshold, then molecule fails the steric strain filter
    :param max_attempts_embed: number of attempts to generate initial 3d
    coordinates
    :param max_num_iters: number of iterations of forcefield minimization
    :return: True if molecule could be successfully minimized, and resulting
    energy is below cutoff, otherwise False
    """
    if mol.GetNumAtoms() <= 2:
        return True
    m = copy.deepcopy(mol)
    m_h = Chem.AddHs(m)
    try:
        flag = AllChem.EmbedMolecule(m_h, maxAttempts=max_attempts_embed)
        if flag == -1:
            return False
    except:
        return False
    AllChem.MMFFSanitizeMolecule(m_h)
    if AllChem.MMFFHasAllMoleculeParams(m_h):
        mmff_props = AllChem.MMFFGetMoleculeProperties(m_h)
        try:
            ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
        except:
            return False
    else:
        return False
    try:
        ff.Minimize(maxIts=max_num_iters)
    except:
        return False
    mmff_props.SetMMFFBondTerm(False)
    mmff_props.SetMMFFAngleTerm(True)
    mmff_props.SetMMFFStretchBendTerm(False)
    mmff_props.SetMMFFOopTerm(False)
    mmff_props.SetMMFFTorsionTerm(False)
    mmff_props.SetMMFFVdWTerm(False)
    mmff_props.SetMMFFEleTerm(False)
    ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
    min_angle_e = ff.CalcEnergy()
    num_atoms = m_h.GetNumAtoms()
    atom_indices = range(num_atoms)
    angle_atom_triplets = itertools.permutations(atom_indices, 3)
    double_num_angles = 0
    for triplet in list(angle_atom_triplets):
        if mmff_props.GetMMFFAngleBendParams(m_h, *triplet):
            double_num_angles += 1
    num_angles = double_num_angles / 2
    avr_angle_e = min_angle_e / num_angles
    if avr_angle_e < cutoff:
        return True
    else:
        return False


def zinc_molecule_filter(mol):
    """
    Flags molecules based on problematic functional groups as
    provided set of ZINC rules from
    http://blaster.docking.org/filtering/rules_default.txt.
    :param mol: rdkit mol object
    :return: Returns True if molecule is okay (ie does not match any of
    therules), False if otherwise
    """
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(mol)


def test_mol_score():
    mol_smiles = 'COC1=CC=C(C2=CC(C3=CC=CC=C3)=CC(C3=CC=C(Br)C=C3)=[O+]2)C=C1'
    print(len(mol_smiles))
    mol = Chem.MolFromSmiles(mol_smiles)
    plogp = penalized_logp(mol)
    q = qed(mol)
    print(mol_smiles)
    print(plogp, q)


if __name__ == '__main__':
    test_mol_score()
