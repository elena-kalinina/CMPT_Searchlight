# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:50:29 2016

@author: elena
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
import CMPT
from CMPT import *
import random
from joblib import Parallel, delayed
import fnmatch


def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest    
    
 
        
cwd="/home/elena/ATTEND/cross_modal/code/"
results_dir="/home/elena/ATTEND/cross_modal/sl_res/test_4"
if os.path.isdir(results_dir)==False:
                os.mkdir(results_dir)
SubjID=["19881016MCBL", "19890126ANPS", "19901103GBTE", "19900422ADDL", "19850630IAAD", "19851030DNGL", "19750827RNPL", "19830905RBMS", "19861104GGBR"]
n_of_subj=len(SubjID)

counter=0
n_perm=10000
all_perms=np.arange(0, n_perm)  #np.arange(0, len(mask_index[0]))
n_splits=n_perm/100
all_perms=np.array_split(all_perms, n_splits)  
threshold=9500
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



labels_all_1=[0]*n_of_subj
labels_all_2=[0]*n_of_subj

subj_data_all_1=[0]*n_of_subj
subj_data_all_2=[0]*n_of_subj

labels_all_3=[0]*n_of_subj
subj_data_all_3=[0]*n_of_subj

SubjName=[0]*n_of_subj
Mod1='Perc'
Mod2='Im'
Mod3='VS'
#Here, we have to stack together each subjects beta maps into a 
#single 4D nifti image; then, each subject's 4D goes into the list
#Labels - because I included them into the file name, here you are free to change
for subj in range(0, n_of_subj): 
    SubjName[subj]=SubjID[subj][-4:]
    print SubjName[subj]
    datapath1='/home/elena/ATTEND/cross_modal/results/whole_brain/whole_brain_' +SubjName[subj] + '_Perc_fts/'
    os.chdir(datapath1)
    labels1=[]
    for file1 in os.listdir(datapath1):
     #   print re.findall('\d+', file1)
        try:
            labels1.append(int(re.findall('\d+', file1)[2]))
        except:
            IndexError
            pass
  #  print len(labels1)
    labels_all_1[subj]=labels1
    
    subj_data_all_1[subj]=concat_imgs(nib.load(os.listdir(datapath1)[k]) for k in range(0, len(os.listdir(datapath1))))
    
  #  print subj_data_all_1[subj].shape
    subj_mod1=main_masker.fit_transform(subj_data_all_1[subj])
    filename1=SubjName[subj]+'_'+Mod1
    np.save(os.path.join('/home/elena/ATTEND/cross_modal/results/whole_brain/', filename1), subj_mod1)
    
    
    datapath2='/home/elena/ATTEND/cross_modal/results/whole_brain/whole_brain_'+SubjName[subj] + '_Im_fts/'
    os.chdir(datapath2)
    labels2=[]
    for file2 in os.listdir(datapath2):
        labels2.append(int(re.findall('\d+', file2)[2]))
  #  print len(labels2)
    labels_all_2[subj]=labels2

    subj_data_all_2[subj]=concat_imgs(nib.load(os.listdir(datapath2)[k]) for k in range(0, len(os.listdir(datapath2))))
  #  print subj_data_all_2[subj].shape
    subj_mod2=main_masker.fit_transform(subj_data_all_2[subj])
    filename2=SubjName[subj]+'_'+Mod2
    np.save(os.path.join('/home/elena/ATTEND/cross_modal/results/whole_brain/', filename2), subj_mod2)
    
    
    datapath3='/home/elena/ATTEND/cross_modal/results/whole_brain/whole_brain_'+SubjName[subj] + '_VS_fts/'
    os.chdir(datapath3)
    labels3=[]
    for file3 in os.listdir(datapath3):
        labels3.append(int(re.findall('\d+', file3)[2]))
  #  print len(labels2)
    labels_all_3[subj]=labels3

    subj_data_all_3[subj]=concat_imgs(nib.load(os.listdir(datapath3)[k]) for k in range(0, len(os.listdir(datapath3))))
  #  print subj_data_all_2[subj].shape
    subj_mod3=main_masker.fit_transform(subj_data_all_3[subj])
    filename3=SubjName[subj]+'_'+Mod3
    np.save(os.path.join('/home/elena/ATTEND/cross_modal/results/whole_brain/', filename3), subj_mod3)

filename_lab1='labels'+Mod1
filename_lab2='labels'+Mod2
filename_lab3='labels'+Mod3
np.save(os.path.join('/home/elena/ATTEND/cross_modal/results/whole_brain/', filename_lab1), labels_all_1)
np.save(os.path.join('/home/elena/ATTEND/cross_modal/results/whole_brain/', filename_lab2), labels_all_2)
np.save(os.path.join('/home/elena/ATTEND/cross_modal/results/whole_brain/', filename_lab3), labels_all_3)

permdatapath='/home/elena/ATTEND/cross_modal/sl_res/test_4'
for perm in range(0, n_perm):
        labels1_perm=scrambled(labels_all_1[1])
                 #   print len(labels1_perm)
        labels2_perm=scrambled(labels_all_2[1])
        labels3_perm=scrambled(labels_all_3[1])
        conditions=np.unique(labels1_perm)
#        print conditions
#        print [labels1_perm==conditions[0]]
        for subj in range(0, n_of_subj): 
                if os.path.isdir(permdatapath+'/'+SubjName[subj])==False:
                    os.mkdir(permdatapath+'/'+SubjName[subj])
                os.chdir(cwd)
                
                subj_cond_dir1_1=permdatapath+'/'+SubjName[subj]+'/'+'Perc_cond0'
            #    print subj_cond_dir1_1
                if os.path.isdir(subj_cond_dir1_1)==False:
                    os.mkdir(subj_cond_dir1_1)
                
                subj_cond_dir1_2=permdatapath+'/'+SubjName[subj]+'/'+'Perc_cond1'
                if os.path.isdir(subj_cond_dir1_2)==False:
                    os.mkdir(subj_cond_dir1_2)
                
                subj_cond_dir2_1=permdatapath+'/'+SubjName[subj]+'/'+'Im_cond0'
                if os.path.isdir(subj_cond_dir2_1)==False:
                    os.mkdir(subj_cond_dir2_1)
                
                subj_cond_dir2_2=permdatapath+'/'+SubjName[subj]+'/'+'Im_cond1'
                if os.path.isdir(subj_cond_dir2_2)==False:
                    os.mkdir(subj_cond_dir2_2)
                    
                subj_cond_dir3_1=permdatapath+'/'+SubjName[subj]+'/'+'VS_cond0'
            #    print subj_cond_dir1_1
                if os.path.isdir(subj_cond_dir3_1)==False:
                    os.mkdir(subj_cond_dir3_1)
                
                subj_cond_dir3_2=permdatapath+'/'+SubjName[subj]+'/'+'VS_cond1'
                if os.path.isdir(subj_cond_dir3_2)==False:
                    os.mkdir(subj_cond_dir3_2)
                
                temp_array1=main_masker.fit_transform(subj_data_all_1[subj])
                temp_array2=main_masker.fit_transform(subj_data_all_2[subj])
                temp_array3=main_masker.fit_transform(subj_data_all_3[subj])
                
                temp_array1_0=np.array((temp_array1[labels1_perm==conditions[0]]).mean(axis=0), dtype='f4')
                temp_array1_1=np.array((temp_array1[labels1_perm==conditions[1]]).mean(axis=0), dtype='f4')
                                
                temp_array2_0=np.array((temp_array2[labels2_perm==conditions[0]]).mean(axis=0), dtype='f4')
                temp_array2_1=np.array((temp_array2[labels2_perm==conditions[1]]).mean(axis=0), dtype='f4')
                
                temp_array3_0=np.array((temp_array3[labels3_perm==conditions[0]]).mean(axis=0), dtype='f4')
                temp_array3_1=np.array((temp_array3[labels3_perm==conditions[1]]).mean(axis=0), dtype='f4')
             #   print np.array(temp_array1[labels1_perm==conditions[0]]).shape
              #  print temp_array1_0.shape
              #  print temp_array1_1.shape
                nii1=main_masker.inverse_transform(temp_array1_0)
                nii2=main_masker.inverse_transform(temp_array1_1)
                nii3=main_masker.inverse_transform(temp_array2_0)
                nii4=main_masker.inverse_transform(temp_array2_1)
                nii5=main_masker.inverse_transform(temp_array3_0)
                nii6=main_masker.inverse_transform(temp_array3_1)
                
                os.chdir(results_dir)        
                nib.save(nii1, os.path.join(subj_cond_dir1_1, 'beta_'+SubjName[subj]+'_'+str(perm)+'_Perc'+'_cond0.nii.gz'))
                nib.save(nii2, os.path.join(subj_cond_dir1_2,'beta_'+SubjName[subj]+'_'+str(perm)+'_Perc'+'_cond1.nii.gz'))
                nib.save(nii3, os.path.join(subj_cond_dir2_1,'beta_'+SubjName[subj]+'_'+str(perm)+'_Im'+'_cond0.nii.gz'))
                nib.save(nii4, os.path.join(subj_cond_dir2_2,'beta_'+SubjName[subj]+'_'+str(perm)+'_Im'+'_cond1.nii.gz'))
                nib.save(nii5, os.path.join(subj_cond_dir3_1, 'beta_'+SubjName[subj]+'_'+str(perm)+'_VS'+'_cond0.nii.gz'))
                nib.save(nii6, os.path.join(subj_cond_dir3_2,'beta_'+SubjName[subj]+'_'+str(perm)+'_VS'+'_cond1.nii.gz'))

subj_data_perm_1_1=[0]*n_of_subj
subj_data_perm_1_2=[0]*n_of_subj
subj_data_perm_2_1=[0]*n_of_subj
subj_data_perm_2_2=[0]*n_of_subj
subj_data_perm_3_1=[0]*n_of_subj
subj_data_perm_3_2=[0]*n_of_subj

os.chdir(permdatapath)

for mysplit in all_perms:
    print mysplit[0]
    for s in range(0, n_of_subj): 
        SubjName=SubjID[s][-4:]
        
        subj_cond_dir1_1=permdatapath+'/'+SubjName+'/'+'Perc_cond0'
        subj_cond_dir1_2=permdatapath+'/'+SubjName+'/'+'Perc_cond1'
        subj_cond_dir2_1=permdatapath+'/'+SubjName+'/'+'Im_cond0'
        subj_cond_dir2_2=permdatapath+'/'+SubjName+'/'+'Im_cond1'
        subj_cond_dir3_1=permdatapath+'/'+SubjName+'/'+'VS_cond0'
        subj_cond_dir3_2=permdatapath+'/'+SubjName+'/'+'VS_cond1'
        
        print SubjName
        os.chdir(subj_cond_dir1_1)
        subj_data_perm_1_1[s]=main_masker.fit_transform(concat_imgs(nib.load(os.listdir(subj_cond_dir1_1)[k]) for k in mysplit))
     #   print subj_data_perm_1_1.shape
              
        os.chdir(subj_cond_dir2_1)
        subj_data_perm_2_1[s]=main_masker.fit_transform(concat_imgs(nib.load(os.listdir(subj_cond_dir2_1)[k]) for k in mysplit))
    #    print subj_data_perm_2_1.shape
        
        os.chdir(subj_cond_dir1_2)
        subj_data_perm_1_2[s]=main_masker.fit_transform(concat_imgs(nib.load(os.listdir(subj_cond_dir1_2)[k]) for k in mysplit))
    #    print subj_data_perm_1_2.shape
                
        os.chdir(subj_cond_dir2_2)
        subj_data_perm_2_2[s]=main_masker.fit_transform(concat_imgs(nib.load(os.listdir(subj_cond_dir2_2)[k]) for k in mysplit))
      
        os.chdir(subj_cond_dir3_1)
        subj_data_perm_3_1[s]=main_masker.fit_transform(concat_imgs(nib.load(os.listdir(subj_cond_dir1_1)[k]) for k in mysplit))
     #   print subj_data_perm_1_1.shape

        os.chdir(subj_cond_dir3_2)
        subj_data_perm_3_2[s]=main_masker.fit_transform(concat_imgs(nib.load(os.listdir(subj_cond_dir1_2)[k]) for k in mysplit))
    #    print subj_data_perm_1_2.shape
 
      
     #   print subj_data_perm_2_2.shape
         
    filename_data_1_1=('100_perm_'+str(mysplit[0])+'_1_1')
    np.save(os.path.join(permdatapath,filename_data_1_1), subj_data_perm_1_1)
    
    filename_data_2_1=('100_perm_'+str(mysplit[0])+'_2_1')
    np.save(os.path.join(permdatapath,filename_data_2_1), subj_data_perm_2_1)
    
    filename_data_1_2=('100_perm_'+str(mysplit[0])+'_1_2')
    np.save(os.path.join(permdatapath,filename_data_1_2), subj_data_perm_1_2)
    
    filename_data_2_2=('100_perm_'+str(mysplit[0])+'_2_2')
    np.save(os.path.join(permdatapath,filename_data_2_2), subj_data_perm_2_2)
    
    filename_data_3_1=('100_perm_'+str(mysplit[0])+'_3_1')
    np.save(os.path.join(permdatapath,filename_data_3_1), subj_data_perm_3_1)
   
    
    filename_data_3_2=('100_perm_'+str(mysplit[0])+'_3_2')
    np.save(os.path.join(permdatapath,filename_data_3_2), subj_data_perm_3_2)