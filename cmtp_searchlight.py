# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:42:21 2016

@author: elena kalinina
"""

import nilearn
from mySMasker import myNiftiSpheresMasker
import nilearn.image as image
from nilearn.image import *
import numpy as np
import nibabel as nib
import os
from nilearn import input_data
import re
from CMTP import *
import random
from joblib import Parallel, delayed


def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest    


#######################Concatenate images
#joint_mni_image = concat_imgs([datasets.load_mni152_template(),
#...                                datasets.load_mni152_template()])
#single_mni_image = index_img(joint_mni_image, 1)
########################################

cwd="/home/elena/ATTEND/cross_modal/code/"
results_dir="/home/elena/ATTEND/cross_modal/sl_res/"
if os.path.isdir(results_dir)==False:
                os.mkdir(results_dir)
SubjID=["19900422ADDL", "19850630IAAD"]
n_of_subj=len(SubjID)

counter=0
perm=10000

#here we load the grey matter mask and create a masker that will transform
#the resulting significance map into a whole brain image

gm_mask = nib.load('/home/elena/ATTEND/MASKS/mynewgreymask.nii.gz')
print gm_mask.shape
main_masker=input_data.NiftiMasker()
mm=main_masker.fit_transform(gm_mask)
print gm_mask.shape
#We also create an index through the mask 
current_mask=np.array((gm_mask.get_data()==1))
print current_mask.shape[0], current_mask.shape[1],current_mask.shape[2]  
mask_index=np.array(np.where(current_mask)) #np.unravel_index(current_mask, 3)

#This one will hold the real group statistic
stat_map=np.zeros_like(mm)

#This one will hold significances. Both need to have the same shape
#as the transformed gm mask to be transformed back to .nii
sign_map=np.zeros_like(mm)
print stat_map.shape


#Next lists will hold the data from all subjects - first,
#for the permutations, next, as the overarching loop is going through voxels
subj_maps_all_1=[0]*n_of_subj
labels_all_1=[0]*n_of_subj

subj_maps_all_2=[0]*n_of_subj
labels_all_2=[0]*n_of_subj

subj_data_all_1=[0]*n_of_subj
subj_data_all_2=[0]*n_of_subj


#Here, we have to stack together each subjects beta maps into a 
#single 4D nifti image; then, each subject's 4D goes into the list
#Labels - because I included them into the file name, here you are free to change
for subj in range(0, n_of_subj): 
    SubjName=SubjID[subj][-4:]
    print SubjName
    datapath1='/home/elena/ATTEND/cross_modal/results/whole_brain/whole_brain_' +SubjName + '_Perc_fts/'
    os.chdir(datapath1)
    labels1=[]
    for file1 in os.listdir(datapath1):
        labels1.append(int(re.findall('\d+', file1)[2]))
    print len(labels1)
    labels_all_1[subj]=labels1
   
    subj_data_all_1[subj]=concat_imgs(nib.load(os.listdir(datapath1)[k]) for k in range(0, len(os.listdir(datapath1))))
    print subj_data_all_1[subj].shape
    
    datapath2='/home/elena/ATTEND/cross_modal/results/whole_brain/whole_brain_'+SubjName + '_VS_fts/'
    os.chdir(datapath2)
    labels2=[]
    for file2 in os.listdir(datapath2):
        labels2.append(int(re.findall('\d+', file2)[2]))
    print len(labels2)
    labels_all_2[subj]=labels2

    subj_data_all_2[subj]=concat_imgs(nib.load(os.listdir(datapath2)[k]) for k in range(0, len(os.listdir(datapath2))))
    print subj_data_all_2[subj].shape
    
    
#here we loop through coordinates creating a sphere around each. if it is to small, the 
#loop breaks
os.chdir(cwd)
print len(mask_index[0])
for i in range(0, len(mask_index[0])):
        print "index", i
        coord=[(mask_index[0][i], mask_index[1][i], mask_index[2][i])]
        print coord
        
        #This variable is needed to break out of the loop
        #if the sphere is too small (smaller than 50 )
        alert=0
      #  if gm_mask.get_data()[mask_index[0][i], mask_index[1][i], mask_index[2][i]]==1:
      #      print gm_mask.get_data()[mask_index[0][i], mask_index[1][i], mask_index[2][i]]
            
       
        for subj in range(0, n_of_subj): 
                
                img_cond_modality_1=[]
                img_cond_modality_2=[]
                counter=counter+1
                if counter%10000==0:
                    print ('We did %d voxels' %counter)
                my_sphere_masker=myNiftiSpheresMasker(coord, radius=8, mask_img=gm_mask)
            #    for img in image.iter_img(data1):
                  #  print img
                try:
                    maps1=my_sphere_masker.fit_transform(subj_data_all_1[subj])
                except ValueError:
                    alert=1
                    print "sphere empty, will skip"
                    break
                 #   print ts.shape
                #    maps1.append(ts)
                print maps1.shape  
                if maps1.shape[1]<50:
                    print "sphere too small, will skip"
                    alert=1
                    break
                else:
                    maps1=np.squeeze(maps1)
                    
                    #here we append together the data from the subject's spheres
                    #because we will run permutations on them  
                    subj_maps_all_1[subj]=maps1#maps1.reshape((maps1.shape[0], maps1.shape[1])) #=maps1.squeeze(axis=2)
                    print 'data1 created'
           
                    maps2=np.squeeze(my_sphere_masker.fit_transform(subj_data_all_2[subj]))
              
                    print maps2.shape
                    subj_maps_all_2[subj]=maps2
                    print 'data2 created'
            
                    for cond in np.unique(labels1):
                        print cond
               
                        print np.array(maps1[labels1==cond]).mean(axis=0).shape
                    
                        img_cond_modality_1.append(np.array(maps1[labels1==cond]).mean(axis=0))
                        img_cond_modality_2.append(np.array(maps2[labels2==cond]).mean(axis=0))
                    t0=test_stat(np.asarray(img_cond_modality_1), np.asarray(img_cond_modality_2))
                    print 'test stat computed'
                    stat_map[:, i]+=t0
        if alert==0:
                stat_perm=0        
                for p in range(0, perm):
                    print p
                    if p%1000==0:
                        print ('permutation' +str(p))
                        
                    #Here I scramble the labels starting with the labels
                    #of a random subject it really does not matter which
                    labels1_perm=scrambled(labels_all_1[1])
                 #   print len(labels1_perm)
                    labels2_perm=scrambled(labels_all_2[1])
                    for subj in range(0, n_of_subj): 
                 
                        maps1=subj_maps_all_1[subj]
                        maps2=subj_maps_all_2[subj]            
                        for cond in np.unique(labels1_perm):
              
                            img_cond_modality_1.append(np.array(maps1[labels1_perm==cond]).mean(axis=0))
                            img_cond_modality_2.append(np.array(maps2[labels2_perm==cond]).mean(axis=0))
                        stat_perm+=test_stat(np.asarray(img_cond_modality_1), np.asarray(img_cond_modality_2))
                           
                    if stat_perm>stat_map[:, i]:
                            sign_map[:, i]+=1
                  
#Here I am saving only significance map but the map 
#with the original statistic can be saved as well

result = Parallel(n_jobs=n_jobs)(delayed(regression_scores)(data_timeseries[trial_i,:,[i,j,z]].T, time_window_size=time_window_size, reg=reg, cv=n_folds, scoring=scoring, timeseriesZ=data_timeseries[trial_i,:,conditionCh].T) for trial_i in range(nTrial))


niimg=main_masker.inverse_transform(sign_map)  
os.chdir(results_dir)        
nib.save(niimg, 'my_sl_test.nii')
print "yeah, done !"



#