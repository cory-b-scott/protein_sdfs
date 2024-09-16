import numpy as np
import torch

from torch_sdf.sdfs.sdf import TorchSDF
from torch_sdf.sdfs.geom import SphereSDF
from torch_sdf.sdfs.compound import UnionSDF, IntersectionSDF
from torch_sdf.sdfs.transform import RoundSDF, InvertSDF
from torch_sdf.sdfs.affine import TranslatedSDF
from torch_sdf.ops import binary_ops as binops
from torch_sdf.ops import unary_ops as unops
from torch_sdf.functional.sdfs import *

from sklearn.neighbors import KDTree

BVH_CUTOFF = 20

class ProteinUnionSDF(UnionSDF):

    def forward(self, query):
        dists = torch.stack([sd(query) for sd in self.sdfs])
        #dmax = dists.max(0)[0]
        #print(query[dmax > 1e6])
        if self.method == 'sharp':
            return binops.nary_sharp_union(dists)
        elif self.method == 'smooth_exp':
            return binops.nary_smooth_union_exp(dists, k=1.0/2.0)

class ProteinKDTreeUnionSDF(UnionSDF):

    def __init__(self, sdfs, k=12, device='cpu', method='sharp'):
        super(ProteinKDTreeUnionSDF, self).__init__(sdfs, device=device, method=method)
        self.k = k
        locs = torch.stack([item.offset for item in self.sdfs])
        rads = torch.stack([item.child.rad for item in self.sdfs])

        self.tree = KDTree(locs.detach().cpu().numpy(), leaf_size=3)
        #self.tree = build_kd_tree(locs)

    def forward(self, query):

        locs = torch.stack([item.offset for item in self.sdfs])
        rads = torch.stack([item.child.rad for item in self.sdfs])

        center_dists, indices = self.tree.query(query.detach().cpu().numpy(), k=self.k)
        #conter_dists, indices = self.tree.query(points_query, nr_nns_searches=k)
        chosen_locs = locs[indices]
        chosen_rads = rads[indices].squeeze(-1)
        #print(chosen_rads.shape, chosen_locs.shape, (chosen_locs - query.unsqueeze(1)).shape)
        #quit()
        dists = torch.linalg.norm( chosen_locs - query.unsqueeze(1), axis=-1) - chosen_rads

        dists = dists.T
        #print(dists.shape)
        #dmax = dists.max(0)[0]
        #print(query[dmax > 1e6])
        if self.method == 'sharp':
            return binops.nary_sharp_union(dists)
        elif self.method == 'smooth_exp':
            return binops.nary_smooth_union_exp(dists, k=1.0/2.0)

class BVHNode(TorchSDF):

    def __init__(self, sdf_list, device='cpu'):
        super(BVHNode, self).__init__()
        if len(sdf_list) <= BVH_CUTOFF:
            self.lc = None
            self.rc = None
            self.sdfs = torch.nn.ModuleList(sdf_list)
        else:
            #splitpt = len(sdf_list) // 2
            locs = [item.offset.detach().cpu().numpy() for item in sdf_list]
            nlocs = np.stack(locs)
            am = np.argmax(nlocs.std(0))
            #idxs = list(np.argsort(nlocs[:, am]))
            positions = nlocs[:, am]
            margin = .5
            m_pos = np.median(positions)
            left_idx = np.where(positions - margin <= m_pos)[0]
            right_idx = np.where(positions - margin > m_pos)[0]

            #print(left_idx, right_idx)
            if len(left_idx) == 0 or len(right_idx) == 0:
                self.lc = None
                self.rc = None
                self.sdfs = torch.nn.ModuleList(sdf_list)
            else:
                rc_list = []
                lc_list = []
                for i in left_idx:
                    lc_list.append(sdf_list[i])
                for i in right_idx:
                    rc_list.append(sdf_list[i])


                self.rc = BVHNode(rc_list)
                self.lc = BVHNode(lc_list)
                self.sdfs = None
        self.static_bbox = None

    def forward(self, query):
        if self.sdfs is None:
            rightDists = axis_aligned_bounding_box_sdf(query, self.rc.bbox())
            leftDists = axis_aligned_bounding_box_sdf(query, self.lc.bbox())

            rightSelect = (rightDists -.5) < 0
            leftSelect = (leftDists - .5) < 0

            leftQuery = self.lc(query[leftSelect])
            rightQuery = self.rc(query[rightSelect])#.shape, .shape)

            rightDists[rightSelect] = rightQuery
            leftDists[leftSelect] = leftQuery


            #dists = torch.stack([leftDists, rightDists])
            return torch.minimum(rightDists, leftDists)
        else:
            dists = torch.stack([sd(query) for sd in self.sdfs])#   print("ding", len(self.sdfs))
            return binops.nary_smooth_union_exp(dists, k=1.0/2.0)

    def bbox(self):
        if self.static_bbox is not None:
            return self.static_bbox
        if self.sdfs is None:
            child_bboxes = [self.rc.bbox(), self.lc.bbox()]
        else:
            child_bboxes = [item.bbox() for item in self.sdfs]

        child_bboxes = [( (item[0].unsqueeze(0), item[1].unsqueeze(0)) if len(item[0].shape) == 1 else item) for item in child_bboxes]

        lowers = torch.stack([item[0] for item in child_bboxes])
        uppers = torch.stack([item[1] for item in child_bboxes])
        all_pts = torch.cat([lowers,uppers])
        self.static_bbox = (all_pts.min(0)[0], all_pts.max(0)[0])
        return self.static_bbox

class ProteinBVH(TorchSDF):

    def __init__(self, sdfs, device='cpu'):
        super(ProteinBVH, self).__init__()

        self.device = device
        self.sdfs = torch.nn.ModuleList(sdfs)
        self.root = self.build_bvh()

    def build_bvh(self):
        return BVHNode(self.sdfs, device=self.device)

    def rebuild(self):
        self.root = self.build_bvh()
        self.root.bbox()

    def forward(self, query):
        return self.root.forward(query)

    def bbox(self):
        return self.root.bbox()


class ProteinContact(TorchSDF):

    def __init__(self, rad_radius, sdfA, sdfB, device='cpu'):
        super(ProteinContact, self).__init__()

        self.device = device
        self.sdfA = sdfA
        self.sdfB = sdfB
        self.r = rad_radius

    def forward(self, query):
        distsA = self.sdfA(query)
        distsB = self.sdfB(query)

        return binops.sharp_union(
            binops.sharp_intersection(distsA,unops.round(distsB, self.r)),
            binops.sharp_intersection(distsB,unops.round(distsA, self.r))
        )


    def bbox(self):
        child_bboxes = [self.sdfA.bbox(), self.sdfB.bbox()]
        lowers = torch.stack([item[0] for item in child_bboxes])
        uppers = torch.stack([item[1] for item in child_bboxes])
        all_pts = torch.cat([lowers,uppers])
        self.static_bbox = (all_pts.min(0)[0], all_pts.max(0)[0])
        return self.static_bbox
