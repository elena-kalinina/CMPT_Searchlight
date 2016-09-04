# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:17:27 2016

@author: elena
"""
import time

from mySMasker import myNiftiSpheresMasker
import nilearn.image as image
from nilearn.image import *
import numpy as np
import nibabel as nib
import os
from nilearn import input_data
import re
import CMPT
from CMPT import *
import random
from joblib import Parallel, delayed
import fnmatch
from pearson_vectorized import pearsonr_vectorized
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
from get_sphere import apply_mask_and_get_affinity

def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest    
    
def my_cmpt_searchlight(n_of_subj, subj_data_all_1, subj_data_all_2, labels_all_1, labels_all_2, mask_index, i, gm_mask):
        counter=0
   # for i in range(0, len(mask_index[0])):
        print "index", i
        alert=0
        stat=0
        coord=[(mask_index[0][i], mask_index[1][i], mask_index[2][i])]
        mysphere=apply_mask_and_get_affinity(coord, niimg, 8, allow_overlap=1, mask_img=gm_mask)
      #  print len(mysphere)
        if len(mysphere)==0:            
            alert=1
            print "sphere empty, will skip"
           # break
        if len(mysphere)<50:
                    print "sphere too small, will skip"
                    alert=1
                 #   break
        else:
 
            
            for subj in range(0, n_of_subj): 
                
                img_cond_modality_1=[]
                img_cond_modality_2=[]
                counter=counter+1
                if counter%10000==0:
                    print ('We did %d voxels' %counter)
  
                maps1=subj_data_all_1[subj][:, mysphere]
                maps2=subj_data_all_1[subj][:, mysphere]
           #     print maps1.shape  
           #     print maps2.shape
    
            
                for cond in np.unique(labels_all_1[subj]):
           
                        img_cond_modality_1.append(np.array(maps1[labels_all_1[subj]==cond]).mean(axis=0))
                        img_cond_modality_2.append(np.array(maps2[labels_all_2[subj]==cond]).mean(axis=0))
                stat+=test_stat(np.asarray(img_cond_modality_1), np.asarray(img_cond_modality_2))
 
                    
        return alert, coord, stat
        

           

def sl_cmpt_permutations_routine(n_of_subj, subj_data_all_1, subj_data_all_2, labels_all_1, labels_all_2, mask_index, vox_group, gm_mask, stat_map, sign_map, split_counter):
    for vox in vox_group:
     #   print vox#range(0, 20): # len(mask_index[0])):
        os.chdir(cwd)
        alert, coord, stat_map[vox] = my_cmpt_searchlight(n_of_subj, subj_data_all_1, subj_data_all_2, labels_all_1, labels_all_2, mask_index, vox, gm_mask)
    #    print stat_map[vox]
    
        if alert==0:
            
            sign_map[vox]=sl_cmpt_permutations(vox, coord, niimg, 8, gm_mask, subj_data_perm_1_1, subj_data_perm_1_2, subj_data_perm_2_1, subj_data_perm_2_2, stat_map[vox], split_counter)
         #   print sign_map.shape
        #    print sign_map[vox]
    return sign_map
        
def sl_cmpt_permutations(vox, coord, niimg, radius, mask, subj_data_perm_1_1, subj_data_perm_1_2, subj_data_perm_2_1, subj_data_perm_2_2, true_stat, split_counter):
    stat_perm=[0]*n_of_subj
    mysphere=apply_mask_and_get_affinity(coord, niimg, 8, allow_overlap=1, mask_img=gm_mask)
     #   perms=[0]*n_of_subj
      #  for perm in range(0, n_perm):
    for s in range(0, n_of_subj):
                img_cond_modality_1=[0]*2
                img_cond_modality_2=[0]*2
                      
                
                img_cond_modality_1[0]=subj_data_perm_1_1[s][:, mysphere]
                img_cond_modality_1[1]=subj_data_perm_1_2[s][:, mysphere]
                img_cond_modality_2[0]=subj_data_perm_2_1[s][:, mysphere]
                img_cond_modality_2[1]=subj_data_perm_2_2[s][:, mysphere]
           #     print img_cond_modality_1[0].shape
                #upd_stat
                stat_perm[s]=pearsonr_vectorized(img_cond_modality_1[0], img_cond_modality_2[0])+pearsonr_vectorized(img_cond_modality_1[1], img_cond_modality_2[1]) - \
                    (pearsonr_vectorized(img_cond_modality_1[0], img_cond_modality_2[1])+pearsonr_vectorized(img_cond_modality_1[1], img_cond_modality_2[0]))
            #    print stat_perm[s]
 
#                if split_counter==0:
#                    stat_perm[s]=upd_stat
#                #    print len(stat_perm[s])
#            #    print stat_perm[s]
#                else:
#                    stat_perm[s]=np.hstack([stat_perm[s], upd_stat])
                 #   print stat_perm[s].shape
                
                
                
                
   # print np.sum(stat_perm, axis=1)
    sign_map[vox]=np.sum(np.sum(stat_perm, axis=0)< true_stat)
  #  print sign_map[vox]
    return   sign_map[vox]

# if __name__ == '__main__':
start = time.time()
cwd="/home/elena/ATTEND/cross_modal/code/"
results_dir="/home/elena/ATTEND/cross_modal/sl_res"
if os.path.isdir(results_dir)==False:
                os.mkdir(results_dir)
SubjID=["19881016MCBL", "19890126ANPS", "19901103GBTE", "19900422ADDL", "19850630IAAD", "19851030DNGL", "19750827RNPL", "19830905RBMS", "19861104GGBR"]
n_of_subj=len(SubjID)

counter=0
#n_perm=100
n_perm=10000
perm_count=np.zeros([n_perm])
threshold=9500


all_perms=np.arange(0, n_perm)  #np.arange(0, len(mask_index[0]))
n_splits=n_perm/100
all_perms=np.array_split(all_perms, n_splits)  


#here we load the grey matter mask and create a masker that will transform
#the resulting significance map into a whole brain image

gm_mask = nib.load('/home/elena/ATTEND/MASKS/mynewgreymask.nii.gz')
print gm_mask.shape
main_masker=input_data.NiftiMasker(mask_img=gm_mask)
mm=main_masker.fit_transform(gm_mask)
print gm_mask.shape
#We also create an index through the mask 
current_mask=np.array((gm_mask.get_data()==1))
print current_mask.shape[0], current_mask.shape[1],current_mask.shape[2]  
mask_index=np.array(np.where(current_mask)) #np.unravel_index(current_mask, 3)

#This one will hold the real group statistic
stat_map=np.zeros_like(mm).T

#This one will hold significances. Both need to have the same shape
#as the transformed gm mask to be transformed back to .nii
new_sign_map=np.zeros_like(mm).T
#new_sign_map=np.zeros_like(mm).T
print stat_map.shape


#Next lists will hold the data from all subjects - first,
#for the permutations, next, as the overarching loop is going through voxels
subj_maps_all_1=[0]*n_of_subj
labels_all_1=[0]*n_of_subj

subj_maps_all_2=[0]*n_of_subj
labels_all_2=[0]*n_of_subj

subj_data_all_1=[0]*n_of_subj
subj_data_all_2=[0]*n_of_subj
#subj_data_perm_1_1=[0]*n_of_subj
#subj_data_perm_2_1=[0]*n_of_subj
#subj_data_perm_1_2=[0]*n_of_subj
#subj_data_perm_2_2=[0]*n_of_subj
SubjName=[0]*n_of_subj
Mod1='Perc'
Mod2='Im'
datapath='/home/elena/ATTEND/cross_modal/results/whole_brain/'
permdatapath='/home/elena/ATTEND/cross_modal/sl_res/test_4'
#Here, we have to stack together each subjects beta maps into a 
#single 4D nifti image; then, each subject's 4D goes into the list
#Labels - because I included them into the file name, here you are free to change
for subj in range(0, n_of_subj): 
        SubjName[subj]=SubjID[subj][-4:]
       
        subj_data_all_1[subj]=np.load(os.path.join(datapath, SubjName[subj]+'_'+Mod1+'.npy'))
  #  print subj_data_all_1[subj].shape
        subj_data_all_2[subj]=np.load(os.path.join(datapath, SubjName[subj]+'_'+Mod2+'.npy'))

labels_all_1=np.load(os.path.join('/home/elena/ATTEND/cross_modal/results/whole_brain/', 'labelsPerc.npy'))
labels_all_2=np.load(os.path.join('/home/elena/ATTEND/cross_modal/results/whole_brain/', 'labelsIm.npy'))
#print len(mask_index[0])
mylabels=np.load(os.path.join('/home/elena/ATTEND/cross_modal/results/whole_brain/', 'labelsPerc.npy'))

n_jobs=-1
niimg = nib.load('/home/elena/ATTEND/cross_modal/sl_res/test_4/beta_ANPS_2_Im_cond1.nii.gz')
all_voxels=np.arange(0, 2000)  #np.arange(0, len(mask_index[0]))
n_splits=6 #n_jobs
#if n_splits<0:
#     n_splits=cpu.count
all_voxels=np.array_split(all_voxels, n_splits)  




split_counter=0
for mysplit in all_perms:
    print "split_counter=", split_counter
    result=[]
    
    
    sign_map=np.zeros_like(mm).T
    subj_data_perm_1_1=[] #[0]*n_of_subj
    subj_data_perm_2_1=[] #[0]*n_of_subj
    subj_data_perm_1_2=[] #[0]*n_of_subj
    subj_data_perm_2_2=[] #[0]*n_of_subj
    
 ########################################################################   
   
   # print subj_data_all_2[subj].shape
#    
#        subj_data_perm_1_1[subj]=np.load(os.path.join(permdatapath, '100_perm_'+ SubjName[subj]+str(asplit[0])+'_1_1'+'.npy'))
#        subj_data_perm_1_2[subj]=np.load(os.path.join(permdatapath, '100_perm_'+str(asplit[0])+ SubjName[subj]+'_1_2'+'.npy'))
#        subj_data_perm_2_1[subj]=np.load(os.path.join(permdatapath, '100_perm_'+str(asplit[0])+ SubjName[subj]+'_2_1'+'.npy'))
#        subj_data_perm_2_2[subj]=np.load(os.path.join(permdatapath, '100_perm_'+str(asplit[0])+ SubjName[subj]+'_2_2'+'.npy'))
#    
###################################################################
  
    subj_data_perm_1_1=np.load(os.path.join(permdatapath,'100_perm_'+str(mysplit[0])+'_1_1.npy'))
    subj_data_perm_2_1=np.load(os.path.join(permdatapath,'100_perm_'+str(mysplit[0])+'_2_1.npy'))
    subj_data_perm_1_2=np.load(os.path.join(permdatapath,'100_perm_'+str(mysplit[0])+'_1_2.npy'))
    subj_data_perm_2_2=np.load(os.path.join(permdatapath,'100_perm_'+str(mysplit[0])+'_2_2.npy'))
    
    

    result = Parallel(n_jobs=n_jobs)(delayed(sl_cmpt_permutations_routine)(n_of_subj, subj_data_all_1, subj_data_all_2, labels_all_1, labels_all_2,  mask_index, mysplit,gm_mask, stat_map, sign_map, split_counter) for mysplit in all_voxels) # len(mask_index[0]))) #  20)) len(mask_index[0])))
    
    print result[0].shape
    print len(result)
    for z in range(0, len(result)):
   #     print z
        xxx=result[z].nonzero()[0]
    #    print len(xxx)
     #   print result[z][xxx]
        new_sign_map[xxx]+=result[z][xxx]
        
    
    
    split_counter+=1
#print len(result)
sign_niimg=main_masker.inverse_transform(new_sign_map.T)
os.chdir(results_dir)        
nib.save(sign_niimg, 'sl_test_Perc_Im1.nii')
print "yeah, done !"
end = time.time()
print (end - start)

#if __name__ == "__main__":
    # execute only if run as a script
 #   main()