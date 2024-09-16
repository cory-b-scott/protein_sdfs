import numpy as np
import torch

from torch_sdf.sdfs.sdf import TorchSDF
from torch_sdf.sdfs.geom import SphereSDF
from torch_sdf.sdfs.compound import UnionSDF, IntersectionSDF
from torch_sdf.sdfs.transform import RoundSDF, InvertSDF
from torch_sdf.sdfs.affine import TranslatedSDF
from torch_sdf.ops import binary_ops as binops
from torch_sdf.ops.unary_ops import *
from torch_sdf.functional.sdfs import *

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgb
from skimage.measure import find_contours

from skimage.transform import resize as skresize
from skimage.io import imread, imsave
from scipy.ndimage import distance_transform_edt,zoom

from tqdm import tqdm

from skimage.measure import marching_cubes
import meshplot as mp
mp.offline()

from sklearn.preprocessing import StandardScaler

from protein_sdfs import ProteinUnionSDF,ProteinKDTreeUnionSDF,ProteinBVH,ProteinContact

vdw_radii = {'H':1.10, 'HE':1.4, 'C':1.7, 'O':1.52, 'N':1.55, 'S':1.8}



def get_grid_coords(K):
    t = (np.mgrid[:K, :K, :K])
    X = t[0].reshape(K**3, 1)
    Y = t[1].reshape(K**3, 1)
    Z = t[2].reshape(K**3, 1)
    coords = np.concatenate([X,Y,Z],1)
    return coords

def parse_atom(string):
    string=" "+string
    radius = vdw_radii[string[77:79].strip()]
    coords = [float(string[31:39]),float(string[39:47]),float(string[47:55])]
    return radius,coords

def pdb_to_sdf(pdb_file, dev, chain, method='flat'):
    sdf_list = []
    with open(pdb_file) as f:
        for line in f.readlines():
            if "ATOM" in line[:4] and (line[21]==chain or chain == "ALL"):
                radius,coords = parse_atom(line)
                sdf =  TranslatedSDF(torch.tensor(coords).to(dev), SphereSDF(torch.tensor([radius]).to(dev)))
                sdf_list.append(sdf)

    if method == 'flat':
        combined = ProteinUnionSDF(sdf_list, method='smooth_exp')
    elif method == 'kdtree':
        combined = ProteinKDTreeUnionSDF(sdf_list, method='smooth_exp')
    else:
        combined = ProteinBVH(sdf_list)
    return combined, len(sdf_list)

def get_bbox_coords(K, sdf):
    coords = get_grid_coords(K)
    coords_scaler = StandardScaler()
    coords_scaler.fit(coords)

    pdb_scaler = StandardScaler()

    pdb_scaler.fit(torch.stack(sdf.bbox()).detach().cpu().numpy())

    bbcoords = pdb_scaler.inverse_transform(coords_scaler.transform(coords))
    return bbcoords

def pdf_sdf_to_mesh(K, dev, sdf, bbcoords):

    dist_lists = []
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(bbcoords).float()), batch_size=BSIZE)
    for batch in loader:
        bXYZ = batch[0].to(dev)
        dist_lists.append(sdf(bXYZ).detach().cpu().numpy())

    dist_array = np.concatenate(dist_lists).reshape(K,K,K)
    dist_array = np.clip(dist_array, a_min=-1e6,a_max=1e6)

    #print("Starting Marching Cubes... ", dist_array.min(), dist_array.max())


    # marching_cubes produces vertex coordinates all in the range 0..K-1
    #vert, face, _, vals = marching_cubes(dist_array, level=0.0, mask = abs(dist_array)<5.0)

    #vert = pdb_scaler.inverse_transform(coords_scaler.transform(vert))
    #return vert, face, vals


if __name__ == '__main__':


    import sys, os
    import trimesh

    # Re
    K = int(sys.argv[1])
    pdb_id = sys.argv[2]
    dev='cuda'
    directory = "../ProtSDFInterface/raw_pdbs"
    round_rad = 4.0
    BSIZE = 2**15

    fname1 = os.path.join(directory,os.path.join(pdb_id, pdb_id+"_r_b_cleaned.pdb"))
    fname2 = os.path.join(directory,os.path.join(pdb_id, pdb_id+"_l_b_cleaned.pdb"))
    #print(fname1)
    #quit()

    coords = get_grid_coords(K)
    coords_scaler = StandardScaler()
    coords_scaler.fit(coords)

    m=torch.rand(32,32).to(dev)
    torch.matmul(m,m)

    import time
    times = [time.time()]
    memvalues = []
    for method in ["flat", "bvh", "kdtree"]:
        #print(method)
        torch.cuda.reset_peak_memory_stats()
        sdfA, countA = pdb_to_sdf(fname1, dev, "A", method=method)
        sdfB, countB = pdb_to_sdf(fname2, dev, "A", method=method)

        sdf = ProteinContact(
            torch.tensor(round_rad).to(dev),
            sdfA,
            sdfB
        )


        pdb_scaler = StandardScaler()

        pdb_scaler.fit(
            torch.cat(RoundSDF(torch.tensor(.05).to(dev),sdf).bbox()).detach().cpu().numpy()
            )
        bbcoords = pdb_scaler.inverse_transform(coords_scaler.transform(coords))

        try:
            pdf_sdf_to_mesh(K, dev, sdf, bbcoords)
            times.append(time.time())
            memvalues.append(torch.cuda.max_memory_allocated())
        except torch.cuda.OutOfMemoryError:


            times.append(time.time())
            memvalues.append(np.inf)

    time_diffs = [t1 - t0 for t0, t1 in zip(times[:-1], times[1:])]
    print(pdb_id, countA, countB,  *time_diffs, *memvalues)
