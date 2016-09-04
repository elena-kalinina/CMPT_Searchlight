# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:20:25 2016

@author: elena
"""

import numpy as np
import sklearn
from sklearn import neighbors
from sklearn.externals.joblib import Memory
from distutils.version import LooseVersion

from nilearn.image.resampling import coord_transform
from nilearn._utils import CacheMixin
from nilearn._utils.niimg_conversions import check_niimg_4d, check_niimg_3d
from nilearn._utils.class_inspect import get_params
from nilearn import image
from nilearn import masking
from nilearn import input_data
from nilearn.input_data.base_masker import filter_and_extract, BaseMasker
import nibabel as nib


def apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap,
                                 mask_img):
    seeds = list(seeds)
    affine = niimg.get_affine()
  #  print affine
    # Compute world coordinates of all in-mask voxels.
  #  print mask_img
    if mask_img is not None:
        mask_img = check_niimg_3d(mask_img)
        mask_img = image.resample_img(mask_img, target_affine=affine,
                                      target_shape=niimg.shape[:3],
                                      interpolation='nearest')
        mask, _ = masking._load_mask_img(mask_img)
        mask_coords = list(zip(*np.where(mask != 0)))
      #  print "1", len(mask_coords)
     #   X = masking._apply_mask_fmri(niimg, mask_img)
    else:
    #    print niimg.shape[:3]
        mask_coords = list(np.ndindex(niimg.shape[:3]))
     #   print "2", len(mask_coords)
      #  X = niimg.get_data().reshape([-1, niimg.shape[3]]).T

    # For each seed, get coordinates of nearest voxel
    nearests = []
    for sx, sy, sz in seeds:
        nearest = np.round(coord_transform(sx, sy, sz, np.linalg.inv(affine)))
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
    #    print nearest
        try:
            nearests.append(mask_coords.index(nearest))
        except ValueError:
            nearests.append(None)
  #      print nearests
    mask_coords = np.asarray(list(zip(*mask_coords)))
    mask_coords = coord_transform(mask_coords[0], mask_coords[1],
                                  mask_coords[2], affine)
    mask_coords = np.asarray(mask_coords).T
  #  print mask_coords
    if (radius is not None and
            LooseVersion(sklearn.__version__) < LooseVersion('0.16')):
        # Fix for scikit learn versions below 0.16. See
        # https://github.com/scikit-learn/scikit-learn/issues/4072
        radius += 1e-6

    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
    A = A.tolil()
  #  print A
    for i, nearest in enumerate(nearests):
        if nearest is None:
            continue
        A[i, nearest] = True
 #   print A
    # Include the voxel containing the seed itself if not masked
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        try:
            A[i, mask_coords.index(seed)] = True
        except ValueError:
            # seed is not in the mask
            pass
    
    if not allow_overlap:
        if np.any(A.sum(axis=0) >= 2):
            raise ValueError('Overlap detected between spheres')
  #  print len(A.rows[0])
  #  print "A=", A[:, 1]
    return A.rows[0] #A[:, 1]

if __name__ == '__main__':

    gm_mask = nib.load('/home/elena/ATTEND/MASKS/mynewgreymask.nii.gz')
    niimg = nib.load('/home/elena/ATTEND/cross_modal/results/whole_brain/whole_brain_ADDL_Im_fts/ADDL_1_15_0.0.nii')
    current_mask=np.array((gm_mask.get_data()==1))
    print current_mask.shape[0], current_mask.shape[1],current_mask.shape[2]  
    mask_index=np.array(np.where(current_mask))
    for vox in range(0, 1): # len(mask_index[0])):
        coord=[(mask_index[0][vox], mask_index[1][vox], mask_index[2][vox])]
        print coord
        mysphere=apply_mask_and_get_affinity(coord, niimg, 8, allow_overlap=1, mask_img=gm_mask)
#print mysphere

    print np.asarray(mysphere).shape
    main_masker=input_data.NiftiMasker()
    mm=main_masker.fit_transform(niimg)
    print len(mm.T[mysphere])
#print b.shape
