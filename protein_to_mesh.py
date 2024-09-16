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

from protein_sdfs import ProteinUnionSDF,ProteinKDTreeUnionSDF

vdw_radii = {'H':1.10, 'HE':1.4, 'C':1.7, 'O':1.52, 'N':1.55, 'S':1.8}

BSIZE = 2**20

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

def pdb_to_sdf(pdb_file, dev, chain):
    sdf_list = []
    with open(pdb_file) as f:
        for line in f.readlines():
            if "ATOM" in line[:4] and (line[21]==chain or chain == "ALL"):
                radius,coords = parse_atom(line)
                sdf =  TranslatedSDF(torch.tensor(coords).to(dev), SphereSDF(torch.tensor([radius]).to(dev)))
                sdf_list.append(sdf)
                for i in range(0):

                    sdf =  TranslatedSDF(torch.tensor(np.array(coords) + np.random.random((3,))).to(dev), SphereSDF(torch.tensor([radius]).to(dev)))
                    sdf_list.append(sdf)

    #combined = ProteinUnionSDF(sdf_list, method='smooth_exp')
    combined = ProteinKDTreeUnionSDF(sdf_list, method='smooth_exp')
    return combined

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
    for batch in tqdm(loader):
        bXYZ = batch[0].to(dev)
        dist_lists.append(sdf(bXYZ).detach().cpu().numpy())

    dist_array = np.concatenate(dist_lists).reshape(K,K,K)
    dist_array = np.clip(dist_array, a_min=-1e6,a_max=1e6)

    print("Starting Marching Cubes... ", dist_array.min(), dist_array.max())


    # marching_cubes produces vertex coordinates all in the range 0..K-1
    vert, face, _, vals = marching_cubes(dist_array, level=0.0, mask = abs(dist_array)<5.0)

    vert = pdb_scaler.inverse_transform(coords_scaler.transform(vert))
    return vert, face, vals


if __name__ == '__main__':


    import sys, os
    import trimesh

    # Re
    K = int(sys.argv[1])
    pdb_id = sys.argv[2]
    dev='cuda'
    directory = "raw_pdbs"
    meshdir = "meshes"
    round_rad = 4.0

    fname1 = os.path.join(directory,os.path.join(pdb_id, pdb_id+"_r_b_cleaned.pdb"))
    fname2 = os.path.join(directory,os.path.join(pdb_id, pdb_id+"_l_b_cleaned.pdb"))
    #print(fname1)
    #quit()

    sdfA = pdb_to_sdf(fname1, dev, "A")
    sdfB = pdb_to_sdf(fname2, dev, "A")

    if False:
        sdf = IntersectionSDF([RoundSDF(torch.tensor(round_rad).to(dev), sdfA), RoundSDF(torch.tensor(round_rad).to(dev), sdfB)], device=dev)

    else:
        sdf = UnionSDF(
            [IntersectionSDF([RoundSDF(torch.tensor(round_rad).to(dev), sdfA),sdfB], device=dev),
             IntersectionSDF([sdfA, RoundSDF(torch.tensor(round_rad).to(dev), sdfB)], device=dev)
             ],
             device=dev,
             method='sharp'
        )
    #sdf = Intersection

    coords = get_grid_coords(K)
    coords_scaler = StandardScaler()
    coords_scaler.fit(coords)


    pdb_scaler = StandardScaler()
    pdb_scaler.fit(
        torch.cat(RoundSDF(torch.tensor(.05).to(dev),sdfA).bbox()).detach().cpu().numpy()
        )
    bbcoords = pdb_scaler.inverse_transform(coords_scaler.transform(coords))


    vertA, faceA, valsA = pdf_sdf_to_mesh(K, dev, sdfA, bbcoords)
    mesh = trimesh.Trimesh(vertices=vertA,
                       faces=faceA)
    mesh.visual = trimesh.visual.ColorVisuals()
    mesh.export(os.path.join(os.path.join(meshdir, pdb_id), pdb_id+"_r.obj"))


    pdb_scaler = StandardScaler()
    pdb_scaler.fit(
        torch.cat(RoundSDF(torch.tensor(.05).to(dev),sdfB).bbox()).detach().cpu().numpy()
        )
    bbcoords = pdb_scaler.inverse_transform(coords_scaler.transform(coords))

    vertB, faceB, valsB = pdf_sdf_to_mesh(K, dev, sdfB, bbcoords)
    mesh = trimesh.Trimesh(vertices=vertB,
                       faces=faceB)
    mesh.visual = trimesh.visual.ColorVisuals()
    mesh.export(os.path.join(os.path.join(meshdir, pdb_id), pdb_id+"_l.obj"))

    pdb_scaler = StandardScaler()
    pdb_scaler.fit(
        torch.cat(RoundSDF(torch.tensor(.05).to(dev),sdf).bbox()).detach().cpu().numpy()
        )
    bbcoords = pdb_scaler.inverse_transform(coords_scaler.transform(coords))

    vert, face, vals = pdf_sdf_to_mesh(K, dev, sdf, bbcoords)
    mesh = trimesh.Trimesh(vertices=vert,
                       faces=face)
    mesh.visual = trimesh.visual.ColorVisuals()
    mesh.export(os.path.join(os.path.join(meshdir, pdb_id), pdb_id+"_rl.obj"))
    #p = mp.plot(vertA, faceA, c=np.array(to_rgb('orange')))
    #p.save("gt_intersection.html")
