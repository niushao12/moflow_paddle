import paddle
import numpy as np
from data.smile_to_graph import GGNNPreprocessor
from rdkit import Chem
from data import transform_qm9
from data.transform_zinc250k import one_hot_zinc250k, transform_fn_zinc250k
from mflow.models.model import MoFlow as Model


def load_model(snapshot_path, model_params, debug=False):
    print('loading snapshot: {}'.format(snapshot_path))
    if debug:
        print('Hyper-parameters:')
        model_params.print()
    model = Model(model_params)
    device = str('cpu').replace('cuda', 'gpu')
    model.set_state_dict(state_dict=paddle.load(path=snapshot_path))
    return model


def smiles_to_adj(mol_smiles, data_name='qm9'):
    out_size = 9
    transform_fn = transform_qm9.transform_fn
    if data_name == 'zinc250k':
        out_size = 38
        transform_fn = transform_fn_zinc250k
    preprocessor = GGNNPreprocessor(out_size=out_size, kekulize=True)
    canonical_smiles, mol = preprocessor.prepare_smiles_and_mol(Chem.
        MolFromSmiles(mol_smiles))
    atoms, adj = preprocessor.get_input_features(mol)
    atoms, adj, _ = transform_fn((atoms, adj, None))
    adj = np.expand_dims(adj, axis=0)
    atoms = np.expand_dims(atoms, axis=0)
    adj = paddle.to_tensor(data=adj)
    atoms = paddle.to_tensor(data=atoms)
    return adj, atoms


def get_latent_vec(model, mol_smiles, data_name='qm9'):
    adj, atoms = smiles_to_adj(mol_smiles, data_name)
    with paddle.no_grad():
        z = model(adj, atoms)
    z = np.hstack([z[0][0].cpu().numpy(), z[0][1].cpu().numpy()]).squeeze(0)
    return z
