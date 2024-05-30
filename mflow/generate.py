import sys
import paddle
import os
import argparse
import sys
sys.path.insert(0, '..')
from distutils.util import strtobool
import numpy as np
import pandas as pd
from orderedset import OrderedSet
import numpy as np
from rdkit.Chem import Draw, AllChem
from rdkit import Chem
from rdkit import Chem, DataStructs
from data import transform_qm9, transform_zinc250k
from data.transform_zinc250k import zinc250_atomic_num_list, transform_fn_zinc250k
from mflow.models.hyperparams import Hyperparameters
from mflow.models.utils import check_validity, adj_to_smiles, check_novelty, valid_mol, construct_mol, _to_numpy_array, correct_mol, valid_mol_can_with_seg
from mflow.utils.model_utils import load_model, get_latent_vec
from mflow.models.model import MoFlow, rescale_adj
import mflow.utils.environment as env
import cairosvg
from data.data_loader import NumpyTupleDataset
import time
from mflow.utils.timereport import TimeReport
import mflow.utils.paddle_aux
import functools
print = functools.partial(print, flush=True)


def generate_mols(model, temp=0.7, z_mu=None, batch_size=20, true_adj=None,
    device=-1):
    """

    :param model: Moflow model
    :param z_mu: latent vector of a molecule
    :param batch_size:
    :param true_adj:
    :param gpu:
    :return:
    """
    if isinstance(device, str):
        pass
    elif isinstance(device, int):
        if device >= 0:
            device = ':'.join([('cuda' if paddle.device.cuda.device_count() >=
                1 else 'cpu').replace('cuda', 'gpu'), str(int(device))])
        else:
            device = str('cpu').replace('cuda', 'gpu')
    else:
        raise ValueError(
            "only 'paddle.device' or 'int' are valid for 'device', but '%s' is given"
             % str(device))
    z_dim = model.b_size + model.a_size
    mu = np.zeros(z_dim)
    sigma_diag = np.ones(z_dim)
    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())
                ) * sigma_diag[:model.b_size]
            sigma_diag[model.b_size + 1:] = np.sqrt(np.exp(model.ln_var[1].
                item())) * sigma_diag[model.b_size + 1:]
    sigma = temp * sigma_diag
    with paddle.no_grad():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * np.eye(z_dim)
        z = np.random.normal(mu, sigma, (batch_size, z_dim))
        z = paddle.to_tensor(data=z).astype(dtype='float32').to(device)
        adj, x = model.reverse(z, true_adj=true_adj)
    return adj, x


def generate_mols_interpolation_grid(model, z0=None, true_adj=None, seed=0,
    mols_per_row=13, device=None, delta=1.0):
    np.random.seed(seed)
    latent_size = model.b_size + model.a_size
    if z0 is None:
        mu = np.zeros([latent_size], dtype=np.float32)
        sigma = 0.02 * np.eye(latent_size, dtype=np.float32)
        z0 = np.random.multivariate_normal(mu, sigma).astype(np.float32)
    x = np.random.randn(latent_size)
    x /= np.linalg.norm(x)
    y = np.random.randn(latent_size)
    y -= y.dot(y=x) * x
    y /= np.linalg.norm(y)
    num_mols_to_edge = mols_per_row // 2
    z_list = []
    p_center = -1
    for dx in range(-num_mols_to_edge, num_mols_to_edge + 1):
        for dy in range(-num_mols_to_edge, num_mols_to_edge + 1):
            z = z0 + x * delta * dx + y * delta * dy
            z_list.append(z)
            if dx == 0 and dy == 0:
                p_center = len(z_list) - 1
    z_array = paddle.to_tensor(data=z_list).astype(dtype='float32')
    if device:
        z_array = z_array.to(device)
        if true_adj:
            true_adj = true_adj.to(device)
    adj, xf = model.reverse(z_array, true_adj=true_adj)
    return adj, xf


def visualize_interpolation(filepath, model, mol_smiles=None, mols_per_row=
    13, delta=0.1, seed=0, atomic_num_list=[6, 7, 8, 9, 0], true_data=None,
    device=None, data_name='qm9', keep_duplicate=False, correct=True):
    z0 = None
    if mol_smiles is not None:
        raise NotImplementedError
    else:
        with paddle.no_grad():
            np.random.seed(seed)
            mol_index = np.random.randint(0, len(true_data))
            adj = np.expand_dims(true_data[mol_index][1], axis=0)
            x = np.expand_dims(true_data[mol_index][0], axis=0)
            adj = paddle.to_tensor(data=adj)
            x = paddle.to_tensor(data=x)
            smile0 = adj_to_smiles(adj, x, atomic_num_list)[0]
            mol0 = Chem.MolFromSmiles(smile0)
            fp0 = AllChem.GetMorganFingerprint(mol0, 2)
            print('seed smile: {}'.format(smile0))
            adj_normalized = rescale_adj(adj)
            if device:
                adj = adj.to(device)
                x = x.to(device)
                adj_normalized = adj_normalized.to(device)
            z0, _ = model(adj, x, adj_normalized)
            z0[0] = z0[0].reshape(tuple(z0[0].shape)[0], -1)
            z0[1] = z0[1].reshape(tuple(z0[1].shape)[0], -1)
            z0 = paddle.concat(x=(z0[0], z0[1]), axis=1).squeeze(axis=0)
            z0 = _to_numpy_array(z0)
    adjm, xm = generate_mols_interpolation_grid(model, z0=z0, mols_per_row=
        mols_per_row, delta=delta, seed=seed, device=device)
    adjm = _to_numpy_array(adjm)
    xm = _to_numpy_array(xm)
    if correct:
        interpolation_mols = []
        for x_elem, adj_elem in zip(xm, adjm):
            mol = construct_mol(x_elem, adj_elem, atomic_num_list)
            cmol = correct_mol(mol)
            vcmol = valid_mol_can_with_seg(cmol)
            interpolation_mols.append(vcmol)
    else:
        interpolation_mols = [valid_mol(construct_mol(x_elem, adj_elem,
            atomic_num_list)) for x_elem, adj_elem in zip(xm, adjm)]
    valid_mols = [mol for mol in interpolation_mols if mol is not None]
    valid_mols_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
    if keep_duplicate:
        valid_mols_smiles_unique = valid_mols_smiles
    else:
        valid_mols_smiles_unique = list(OrderedSet(valid_mols_smiles))
    valid_mols_unique = [Chem.MolFromSmiles(s) for s in
        valid_mols_smiles_unique]
    valid_mols_smiles_unique_label = []
    print(
        'len(interpolation_mols):{}, len(valid_mols):{}, len(valid_mols_smiles_unique):{}'
        .format(len(interpolation_mols), len(valid_mols), len(
        valid_mols_smiles_unique)))
    for s, m in zip(valid_mols_smiles_unique, valid_mols_unique):
        fp = AllChem.GetMorganFingerprint(m, 2)
        sim = DataStructs.TanimotoSimilarity(fp, fp0)
        s = ' {:.2f}'.format(sim)
        valid_mols_smiles_unique_label.append(s)
    if keep_duplicate:
        molsPerRow = mols_per_row
    else:
        molsPerRow = 9
    k = len(valid_mols_smiles_unique)
    print('interpolation_mols valid {} / {}'.format(len(valid_mols), len(
        interpolation_mols)))
    if data_name == 'qm9':
        psize = 150, 150
    else:
        psize = 150, 150
    img = Draw.MolsToGridImage(valid_mols_unique[:k], molsPerRow=molsPerRow,
        legends=valid_mols_smiles_unique_label[:k], subImgSize=psize)
    img.save(filepath + '_.png')
    svg = Draw.MolsToGridImage(valid_mols_unique[:k], molsPerRow=molsPerRow,
        legends=valid_mols_smiles_unique_label[:k], subImgSize=psize,
        useSVG=True)
    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to=filepath + '.pdf'
        )
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=filepath + '.png'
        )
    print('Dump {}.png/pdf done'.format(filepath))


def visualize_interpolation_between_2_points(filepath, model, mol_smiles=
    None, mols_per_row=15, n_interpolation=100, seed=0, atomic_num_list=[6,
    7, 8, 9, 0], true_data=None, device=None, data_name='qm9'):
    z0 = None
    if mol_smiles is not None:
        raise NotImplementedError
    else:
        with paddle.no_grad():
            np.random.seed(seed)
            mol_index = np.random.randint(0, len(true_data), 2)
            adj0 = np.expand_dims(true_data[mol_index[0]][1], axis=0)
            x0 = np.expand_dims(true_data[mol_index[0]][0], axis=0)
            adj0 = paddle.to_tensor(data=adj0)
            x0 = paddle.to_tensor(data=x0)
            smile0 = adj_to_smiles(adj0, x0, atomic_num_list)[0]
            mol0 = Chem.MolFromSmiles(smile0)
            fp0 = AllChem.GetMorganFingerprint(mol0, 2)
            adj1 = np.expand_dims(true_data[mol_index[1]][1], axis=0)
            x1 = np.expand_dims(true_data[mol_index[1]][0], axis=0)
            adj1 = paddle.to_tensor(data=adj1)
            x1 = paddle.to_tensor(data=x1)
            smile1 = adj_to_smiles(adj1, x1, atomic_num_list)[0]
            mol1 = Chem.MolFromSmiles(smile1)
            print('seed smile0: {}'.format(smile0))
            print('seed smile1: {}'.format(smile1))
            adj_normalized0 = rescale_adj(adj0)
            if device:
                adj0 = adj0.to(device)
                x0 = x0.to(device)
                adj_normalized0 = adj_normalized0.to(device)
            z0, _ = model(adj0, x0, adj_normalized0)
            z0[0] = z0[0].reshape(tuple(z0[0].shape)[0], -1)
            z0[1] = z0[1].reshape(tuple(z0[1].shape)[0], -1)
            z0 = paddle.concat(x=(z0[0], z0[1]), axis=1).squeeze(axis=0)
            z0 = _to_numpy_array(z0)
            adj_normalized1 = rescale_adj(adj1)
            if device:
                adj1 = adj1.to(device)
                x1 = x1.to(device)
                adj_normalized1 = adj_normalized1.to(device)
            z1, _ = model(adj1, x1, adj_normalized1)
            z1[0] = z1[0].reshape(tuple(z1[0].shape)[0], -1)
            z1[1] = z1[1].reshape(tuple(z1[1].shape)[0], -1)
            z1 = paddle.concat(x=(z1[0], z1[1]), axis=1).squeeze(axis=0)
            z1 = _to_numpy_array(z1)
    d = z1 - z0
    z_list = [(z0 + i * 1.0 / (n_interpolation + 1) * d) for i in range(
        n_interpolation + 2)]
    z_array = paddle.to_tensor(data=z_list).astype(dtype='float32')
    if device:
        z_array = z_array.to(device)
    adjm, xm = model.reverse(z_array)
    adjm = _to_numpy_array(adjm)
    xm = _to_numpy_array(xm)
    interpolation_mols = [valid_mol(construct_mol(x_elem, adj_elem,
        atomic_num_list)) for x_elem, adj_elem in zip(xm, adjm)]
    valid_mols = [mol for mol in interpolation_mols if mol is not None]
    valid_mols_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
    valid_mols_smiles_unique = list(OrderedSet(valid_mols_smiles))
    valid_mols_unique = [Chem.MolFromSmiles(s) for s in
        valid_mols_smiles_unique]
    valid_mols_smiles_unique_label = []
    for s, m in zip(valid_mols_smiles_unique, valid_mols_unique):
        fp = AllChem.GetMorganFingerprint(m, 2)
        sim = DataStructs.TanimotoSimilarity(fp, fp0)
        s = '{:.2f}\n'.format(sim) + s
        if s == smile0:
            s = '***[' + s + ']***'
        valid_mols_smiles_unique_label.append(s)
    print('interpolation_mols valid {} / {}'.format(len(valid_mols), len(
        interpolation_mols)))
    if data_name == 'qm9':
        psize = 200, 200
    else:
        psize = 200, 200
    img = Draw.MolsToGridImage(valid_mols_unique, legends=
        valid_mols_smiles_unique_label, molsPerRow=mols_per_row, subImgSize
        =psize)
    img.save(filepath + '_.png')
    svg = Draw.MolsToGridImage(valid_mols_unique, legends=
        valid_mols_smiles_unique_label, molsPerRow=mols_per_row, subImgSize
        =psize, useSVG=True)
    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to=filepath + '.pdf'
        )
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=filepath + '.png'
        )
    print('Dump {}.png/pdf done'.format(filepath))


def plot_colormap():
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()
    a = [[0.42, 0.41, 0.38, 0.32, 0.44, 0.37, 0.37, 0.37, 0.31], [0.53, 
        0.53, 0.53, 0.68, 0.68, 0.57, 0.57, 0.43, 0.32], [1, 0.52, 0.68, 
        0.68, 0.68, 0.68, 0.57, 0.47, 0.39], [1, 1, 1, 0.68, 0.68, 0.68, 1,
        1, 0.57], [1, 1, 1, 1, 1, 1, 1, 1, 0.68], [0.59, 0.59, 0.59, 0.59, 
        0.59, 0.59, 0.79, 0.79, 0.53], [0.59, 0.59, 0.59, 0.59, 0.59, 0.59,
        0.59, 0.53, 0.71], [0.34, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.54,
        0.71], [0.32, 0.34, 0.59, 0.59, 0.59, 0.59, 0.32, 0.54, 0.54]]
    am = np.array(a)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax = sns.heatmap(am, annot=True, fmt='.2f', linewidths=0.5, ax=ax,
        annot_kws={'size': 18}, vmin=0, vmax=1)
    plt.show()
    fig.savefig('a_all.pdf')
    idx = 0
    for idx in range(9):
        n = len(a[idx])
        aa = np.array(a[idx])
        aa = np.expand_dims(aa, axis=0)
        fig, ax = plt.subplots(figsize=(14.4, 0.8))
        ax = sns.heatmap(aa, annot=True, fmt='.2f', linewidths=0.5, ax=ax,
            annot_kws={'size': 18}, cbar=False, xticklabels=False,
            yticklabels=False, vmin=0, vmax=1)
        plt.show()
        fig.savefig('a{}.pdf'.format(idx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./results')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--data_name', type=str, default='qm9', choices=[
        'qm9', 'zinc250k'], help='dataset name')
    parser.add_argument('--snapshot-path', '-snapshot', type=str, required=True
        )
    parser.add_argument('--hyperparams-path', type=str, default=
        'moflow-params.json', required=True)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--additive_transformations', type=strtobool,
        default='false', help='apply only additive coupling layers')
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--n_experiments', type=int, default=1, help=
        'number of times generation to be run')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--temperature', type=float, default=1.0, help=
        'temperature of the gaussian distribution')
    parser.add_argument('--save_fig', type=strtobool, default='true')
    parser.add_argument('--save_score', type=strtobool, default='true')
    parser.add_argument('-r', '--reconstruct', action='store_true', default
        =False)
    parser.add_argument('--int2point', action='store_true', default=False)
    parser.add_argument('--intgrid', action='store_true', default=False)
    parser.add_argument('--inter_times', type=int, default=5)
    parser.add_argument('--correct_validity', type=strtobool, default=
        'true', help='if apply validity correction after the generation')
    args = parser.parse_args()
    start = time.time()
    print('Start at Time: {}'.format(time.ctime()))
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    print('loading hyperparamaters from {}'.format(hyperparams_path))
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True)
    if len(model.ln_var) == 1:
        print('model.ln_var: {:.2f}'.format(model.ln_var.item()))
    elif len(model.ln_var) == 2:
        print('model.ln_var[0]: {:.2f}, model.ln_var[1]: {:.2f}'.format(
            model.ln_var[0].item(), model.ln_var[1].item()))
    if args.gpu >= 0:
        device = str('cuda:' + str(args.gpu) if paddle.device.cuda.
            device_count() >= 1 else 'cpu').replace('cuda', 'gpu')
    else:
        device = str('cpu').replace('cuda', 'gpu')
    model.to(device)
    model.eval()
    if args.data_name == 'qm9':
        atomic_num_list = [6, 7, 8, 9, 0]
        transform_fn = transform_qm9.transform_fn
        valid_idx = transform_qm9.get_val_ids()
        molecule_file = 'qm9_relgcn_kekulized_ggnp.npz'
    elif args.data_name == 'zinc250k':
        atomic_num_list = zinc250_atomic_num_list
        transform_fn = transform_zinc250k.transform_fn_zinc250k
        valid_idx = transform_zinc250k.get_val_ids()
        molecule_file = 'zinc250k_relgcn_kekulized_ggnp.npz'
    batch_size = args.batch_size
    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir,
        molecule_file), transform=transform_fn)
    assert len(valid_idx) > 0
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
    n_train = len(train_idx)
    train = paddle.io.Subset(dataset=dataset, indices=train_idx)
    test = paddle.io.Subset(dataset=dataset, indices=valid_idx)
    print(
        '{} in total, {}  training data, {}  testing data, {} batchsize, train/batchsize {}'
        .format(len(dataset), len(train), len(test), batch_size, len(train) /
        batch_size))
    if args.reconstruct:
        train_dataloader = paddle.io.DataLoader(dataset=train, batch_size=
            batch_size)
        reconstruction_rate_list = []
        max_iter = len(train_dataloader)
        for i, batch in enumerate(train_dataloader):
            x = batch[0].to(device)
            adj = batch[1].to(device)
            adj_normalized = rescale_adj(adj).to(device)
            z, sum_log_det_jacs = model(adj, x, adj_normalized)
            z0 = z[0].reshape(tuple(z[0].shape)[0], -1)
            z1 = z[1].reshape(tuple(z[1].shape)[0], -1)
            adj_rev, x_rev = model.reverse(paddle.concat(x=[z0, z1], axis=1))
            reverse_smiles = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(),
                atomic_num_list)
            train_smiles = adj_to_smiles(adj.cpu(), x.cpu(), atomic_num_list)
            lb = np.array([int(a != b) for a, b in zip(train_smiles,
                reverse_smiles)])
            idx = np.where(lb)[0]
            if len(idx) > 0:
                for k in idx:
                    print(i * batch_size + k, 'train: ', train_smiles[k],
                        ' reverse: ', reverse_smiles[k])
            reconstruction_rate = 1.0 - lb.mean()
            reconstruction_rate_list.append(reconstruction_rate)
            print('iter/total: {}/{}, reconstruction_rate:{}'.format(i,
                max_iter, reconstruction_rate))
        reconstruction_rate_total = np.array(reconstruction_rate_list).mean()
        print('reconstruction_rate for all the train data:{} in {}'.format(
            reconstruction_rate_total, len(train)))
        exit(0)
    if args.int2point:
        mol_smiles = None
        gen_dir = os.path.join(args.model_dir, 'generated')
        print('Dump figure in {}'.format(gen_dir))
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        for seed in range(args.inter_times):
            filepath = os.path.join(gen_dir,
                '2points_interpolation-2point_molecules_seed{}'.format(seed))
            visualize_interpolation_between_2_points(filepath, model,
                mol_smiles=mol_smiles, mols_per_row=15, n_interpolation=50,
                atomic_num_list=atomic_num_list, seed=seed, true_data=train,
                device=device, data_name=args.data_name)
        exit(0)
    if args.intgrid:
        mol_smiles = None
        gen_dir = os.path.join(args.model_dir, 'generated')
        print('Dump figure in {}'.format(gen_dir))
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        for seed in range(args.inter_times):
            filepath = os.path.join(gen_dir,
                'generated_interpolation-grid_molecules_seed{}'.format(seed))
            print('saving {}'.format(filepath))
            visualize_interpolation(filepath, model, mol_smiles=mol_smiles,
                mols_per_row=9, delta=args.delta, atomic_num_list=
                atomic_num_list, seed=seed, true_data=train, device=device,
                data_name=args.data_name, keep_duplicate=True)
            filepath = os.path.join(gen_dir,
                'generated_interpolation-grid_molecules_seed{}_unique'.
                format(seed))
            visualize_interpolation(filepath, model, mol_smiles=mol_smiles,
                mols_per_row=9, delta=args.delta, atomic_num_list=
                atomic_num_list, seed=seed, true_data=train, device=device,
                data_name=args.data_name, keep_duplicate=False)
        exit(0)
    train_x = [a[0] for a in train]
    train_adj = [a[1] for a in train]
    train_smiles = adj_to_smiles(train_adj, train_x, atomic_num_list)
    print('Load trained model and data done! Time {:.2f} seconds'.format(
        time.time() - start))
    save_fig = args.save_fig
    valid_ratio = []
    unique_ratio = []
    novel_ratio = []
    abs_unique_ratio = []
    abs_novel_ratio = []
    for i in range(args.n_experiments):
        adj, x = generate_mols(model, batch_size=batch_size, true_adj=None,
            temp=args.temperature, device=device)
        val_res = check_validity(adj, x, atomic_num_list, correct_validity=
            args.correct_validity)
        novel_r, abs_novel_r = check_novelty(val_res['valid_smiles'],
            train_smiles, tuple(x.shape)[0])
        novel_ratio.append(novel_r)
        abs_novel_ratio.append(abs_novel_r)
        unique_ratio.append(val_res['unique_ratio'])
        abs_unique_ratio.append(val_res['abs_unique_ratio'])
        valid_ratio.append(val_res['valid_ratio'])
        n_valid = len(val_res['valid_mols'])
        if args.save_score:
            assert len(val_res['valid_smiles']) == len(val_res['valid_mols'])
            smiles_qed_plogp = [(sm, env.qed(mol), env.penalized_logp(mol)) for
                sm, mol in zip(val_res['valid_smiles'], val_res['valid_mols'])]
            smiles_qed_plogp.sort(key=lambda tup: tup[2], reverse=True)
            gen_dir = os.path.join(args.model_dir, 'generated')
            os.makedirs(gen_dir, exist_ok=True)
            filepath = os.path.join(gen_dir,
                'smiles_qed_plogp_{}_RankedByPlogp.csv'.format(i))
            df = pd.DataFrame(smiles_qed_plogp, columns=['Smiles', 'QED',
                'Penalized_logp'])
            df.to_csv(filepath, index=None, header=True)
            smiles_qed_plogp.sort(key=lambda tup: tup[1], reverse=True)
            filepath2 = os.path.join(gen_dir,
                'smiles_qed_plogp_{}_RankedByQED.csv'.format(i))
            df2 = pd.DataFrame(smiles_qed_plogp, columns=['Smiles', 'QED',
                'Penalized_logp'])
            df2.to_csv(filepath2, index=None, header=True)
        if save_fig:
            gen_dir = os.path.join(args.model_dir, 'generated')
            os.makedirs(gen_dir, exist_ok=True)
            filepath = os.path.join(gen_dir, 'generated_mols_{}.png'.format(i))
            img = Draw.MolsToGridImage(val_res['valid_mols'], legends=
                val_res['valid_smiles'], molsPerRow=20, subImgSize=(300, 300))
            img.save(filepath)
    print('validity: mean={:.2f}%, sd={:.2f}%, vals={}'.format(np.mean(
        valid_ratio), np.std(valid_ratio), valid_ratio))
    print('novelty: mean={:.2f}%, sd={:.2f}%, vals={}'.format(np.mean(
        novel_ratio), np.std(novel_ratio), novel_ratio))
    print('uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}'.format(np.mean(
        unique_ratio), np.std(unique_ratio), unique_ratio))
    print('abs_novelty: mean={:.2f}%, sd={:.2f}%, vals={}'.format(np.mean(
        abs_novel_ratio), np.std(abs_novel_ratio), abs_novel_ratio))
    print('abs_uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}'.format(np.
        mean(abs_unique_ratio), np.std(abs_unique_ratio), abs_unique_ratio))
    print('Task random generation done! Time {:.2f} seconds, Data: {}'.
        format(time.time() - start, time.ctime()))
