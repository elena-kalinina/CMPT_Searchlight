# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:17:27 2016

@author: elena
"""
import time
from timeit import default_timer as timer
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
from pearson_vectorized import pearsonr_vectorized

def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest    
    
def my_cmpt_searchlight(n_of_subj, subj_data_all_1, subj_data_all_2, labels_all_1, labels_all_2, mask_index, i, gm_mask):
        counter=0
   # for i in range(0, len(mask_index[0])):
        print "index", i
        coord=[(mask_index[0][i], mask_index[1][i], mask_index[2][i])]
     #   print coord
        
        #This variable is needed to break out of the loop
        #if the sphere is too small (smaller than 50 )
        alert=0
      #  if gm_mask.get_data()[mask_index[0][i], mask_index[1][i], mask_index[2][i]]==1:
      #      print gm_mask.get_data()[mask_index[0][i], mask_index[1][i], mask_index[2][i]]
            
        stat=0
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
            #    print maps1.shape  
                if maps1.shape[1]<50:
                    print "sphere too small, will skip"
                    alert=1
                    break
                else:
                    maps1=np.squeeze(maps1)
                    
                    #here we append together the data from the subject's spheres
                    #because we will run permutations on them  
                    subj_maps_all_1[subj]=maps1#maps1.reshape((maps1.shape[0], maps1.shape[1])) #=maps1.squeeze(axis=2)
                 #   print 'data1 created'
           
                    maps2=np.squeeze(my_sphere_masker.fit_transform(subj_data_all_2[subj]))
              
               #     print maps2.shape
                    subj_maps_all_2[subj]=maps2
               #     print 'data2 created'
            
                    for cond in np.unique(labels_all_1[subj]):
                  #      print cond
               
                   #     print np.array(maps1[labels_all_1[subj]==cond]).mean(axis=0).shape
                    
                        img_cond_modality_1.append(np.array(maps1[labels_all_1[subj]==cond]).mean(axis=0))
                        img_cond_modality_2.append(np.array(maps2[labels_all_2[subj]==cond]).mean(axis=0))
                    stat+=test_stat(np.asarray(img_cond_modality_1), np.asarray(img_cond_modality_2))
             #       print 'test stat computed'
                   # stat_map[:, i]+=t0
                    #stat_map[i]+=t0
                    
        return alert, stat, subj_maps_all_1,subj_maps_all_2
        
def my_cmpt_searchlight_new(n_of_subj, subj_data_all_1, subj_data_all_2, labels_all_1, labels_all_2, mask_index, i, gm_mask):
        counter=0
   # for i in range(0, len(mask_index[0])):
        print "index", i
        coord=[(mask_index[0][i], mask_index[1][i], mask_index[2][i])]
     #   print coord
        
        #This variable is needed to break out of the loop
        #if the sphere is too small (smaller than 50 )
        alert=0
      #  if gm_mask.get_data()[mask_index[0][i], mask_index[1][i], mask_index[2][i]]==1:
      #      print gm_mask.get_data()[mask_index[0][i], mask_index[1][i], mask_index[2][i]]
            
        stat=0
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
            #    print maps1.shape  
                if maps1.shape[1]<30:
                    print "sphere too small, will skip"
                    alert=1
                    break
                else:
                    maps1=np.squeeze(maps1)
                    
                              
                    maps2=np.squeeze(my_sphere_masker.fit_transform(subj_data_all_2[subj]))
              
            
                    for cond in np.unique(labels_all_1[subj]):
                  #      print cond
               
                   #     print np.array(maps1[labels_all_1[subj]==cond]).mean(axis=0).shape
                    
                        img_cond_modality_1.append(np.array(maps1[labels_all_1[subj]==cond]).mean(axis=0))
                        img_cond_modality_2.append(np.array(maps2[labels_all_2[subj]==cond]).mean(axis=0))
                    stat+=test_stat(np.asarray(img_cond_modality_1), np.asarray(img_cond_modality_2))
             #       print 'test stat computed'
                   # stat_map[:, i]+=t0
                    #stat_map[i]+=t0
                    
        return alert, stat, my_sphere_masker
                

def sl_cmpt_permutations(all_labels1, all_labels2, all_maps1, all_maps2, n_of_subj, stat_true):
        img_cond_modality_1=[]
        img_cond_modality_2=[]
        count_perm=0
        stat_perm=0
        print "Permutations started"
        
                 #   print p
               #     if p%1000==0:
                 #       print ('permutation' +str(p))
                        
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
                           
                        if stat_perm<stat_true:
                            count_perm=1
                           # perm_count[perm]=1#stat_map[i]:   #[:, i]:
                          #  sign_map[:, i]+=1
                         #  sign_map[i]+=1
                    
       
#Here I am saving only significance map but the map 
#with the original statistic can be saved as well
        return count_perm
        
        
def sl_cmpt_permutations_new(perm_data1, perm_data2, masker, nperm, stat_true):
   # permdatapath='/home/elena/ATTEND/cross_modal/sl_res/test/'
    stat_perm=0
    count_perm=0
    
    
    for s in subjects:
        fname11=permdatapath+'beta_'+SubjName[subj]+'_'+str(nperm)+'_Perc'+'_cond0.nii'
        fname12=permdatapath+'beta_'+SubjName[subj]+'_'+str(nperm)+'_Perc'+'_cond1.nii'
        fname21=permdatapath+'beta_'+SubjName[subj]+'_'+str(nperm)+'_Im'+'_cond0.nii'
        fname22=permdatapath+'beta_'+SubjName[subj]+'_'+str(nperm)+'_Im'+'_cond1.nii'
        img_cond_modality_1=[0]*2
        img_cond_modality_2=[0]*2
       
        
        
        img_cond_modality_1[0]=np.squeeze(masker.fit_transform(nib.load(fname11)))
        img_cond_modality_1[1]=np.squeeze(masker.fit_transform(nib.load(fname12)))
        img_cond_modality_2[0]=np.squeeze(masker.fit_transform(nib.load(fname21)))
        img_cond_modality_2[1]=np.squeeze(masker.fit_transform(nib.load(fname22)))
        
        stat_perm+=test_stat(np.asarray(img_cond_modality_1), np.asarray(img_cond_modality_2))
                           
    if stat_perm<stat_true:
            count_perm=1
                           # perm_count[perm]=1#stat_map[i]:   #[:, i]:
                          #  sign_map[:, i]+=1
                         #  sign_map[i]+=1
                    
    print stat_perm
    print count_perm
#Here I am saving only significance map but the map 
#with the original statistic can be saved as well
    return count_perm
        

# if __name__ == '__main__':
start = time.time()
cwd="/home/elena/ATTEND/cross_modal/code/"
results_dir="/home/elena/ATTEND/cross_modal/sl_res/test"
if os.path.isdir(results_dir)==False:
                os.mkdir(results_dir)
SubjID=["19881016MCBL", "19890126ANPS", "19901103GBTE", "19900422ADDL", "19850630IAAD", "19851030DNGL", "19750827RNPL", "19830905RBMS", "19861104GGBR"]
n_of_subj=len(SubjID)

counter=0
n_perm=10
perm_count=np.zeros([n_perm])
threshold=9500
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
stat_map=np.zeros_like(mm).T

#This one will hold significances. Both need to have the same shape
#as the transformed gm mask to be transformed back to .nii
sign_map=np.zeros_like(mm).T
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
subj_data_perm_1_1=[0]*n_of_subj
subj_data_perm_2_1=[0]*n_of_subj
subj_data_perm_1_2=[0]*n_of_subj
subj_data_perm_2_2=[0]*n_of_subj
SubjName=[0]*n_of_subj
permdatapath='/home/elena/ATTEND/cross_modal/sl_res/test'
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
        labels1.append(int(re.findall('\d+', file1)[2]))
    print len(labels1)
    labels_all_1[subj]=labels1
   
    subj_data_all_1[subj]=concat_imgs(nib.load(os.listdir(datapath1)[k]) for k in range(0, len(os.listdir(datapath1))))
    print subj_data_all_1[subj].shape
    
    datapath2='/home/elena/ATTEND/cross_modal/results/whole_brain/whole_brain_'+SubjName[subj] + '_Im_fts/'
    os.chdir(datapath2)
    labels2=[]
    for file2 in os.listdir(datapath2):
        labels2.append(int(re.findall('\d+', file2)[2]))
    print len(labels2)
    labels_all_2[subj]=labels2

    subj_data_all_2[subj]=concat_imgs(nib.load(os.listdir(datapath2)[k]) for k in range(0, len(os.listdir(datapath2))))
    print subj_data_all_2[subj].shape
    os.chdir(permdatapath)
    subj_data_perm_1_1[subj]=concat_imgs(nib.load(os.listdir(permdatapath)[k]) for k in range(0, len(os.listdir(permdatapath))) if fnmatch.fnmatch(os.listdir(permdatapath)[k], '*'+SubjName[subj]+'*') and fnmatch.fnmatch(os.listdir(permdatapath)[k], '*Perc*') & fnmatch.fnmatch(os.listdir(permdatapath)[k], '*cond0*'))
    subj_data_perm_2_1[subj]=concat_imgs(nib.load(os.listdir(permdatapath)[k]) for k in range(0, len(os.listdir(permdatapath))) if fnmatch.fnmatch(os.listdir(permdatapath)[k], '*'+SubjName[subj]+'*') and fnmatch.fnmatch(os.listdir(permdatapath)[k], '*Im*') & fnmatch.fnmatch(os.listdir(permdatapath)[k], '*cond0*'))
    subj_data_perm_1_2[subj]=concat_imgs(nib.load(os.listdir(permdatapath)[k]) for k in range(0, len(os.listdir(permdatapath))) if fnmatch.fnmatch(os.listdir(permdatapath)[k], '*'+SubjName[subj]+'*') and fnmatch.fnmatch(os.listdir(permdatapath)[k], '*Perc*') and fnmatch.fnmatch(os.listdir(permdatapath)[k], '*cond1*'))
    subj_data_perm_2_2[subj]=concat_imgs(nib.load(os.listdir(permdatapath)[k]) for k in range(0, len(os.listdir(permdatapath))) if fnmatch.fnmatch(os.listdir(permdatapath)[k], '*'+SubjName[subj]+'*') and fnmatch.fnmatch(os.listdir(permdatapath)[k], '*Im*') and fnmatch.fnmatch(os.listdir(permdatapath)[k], '*cond1*'))
    print subj_data_perm_1_1[subj].shape
    print subj_data_perm_2_1[subj].shape
    print subj_data_perm_1_2[subj].shape
    print subj_data_perm_2_2[subj].shape
#########################################
#for perm in range(0, n_perm):
#        labels1_perm=scrambled(labels_all_1[1])
#                 #   print len(labels1_perm)
#        labels2_perm=scrambled(labels_all_2[1])
#        conditions=np.unique(labels1_perm)
##        print conditions
##        print [labels1_perm==conditions[0]]
#        for subj in range(0, n_of_subj): 
#                os.chdir(cwd)
#                
#                my_new_masker=input_data.NiftiMasker(mask_img=gm_mask)
#                temp_array1=my_new_masker.fit_transform(subj_data_all_1[subj])
#                temp_array2=my_new_masker.fit_transform(subj_data_all_2[subj])
#                
#                temp_array1_0=np.array(temp_array1[labels1_perm==conditions[0]]).mean(axis=0)
#                temp_array1_1=np.array(temp_array1[labels1_perm==conditions[1]]).mean(axis=0)
#                
#                
#                temp_array2_0=np.array(temp_array2[labels1_perm==conditions[0]]).mean(axis=0)
#                temp_array2_1=np.array(temp_array2[labels1_perm==conditions[1]]).mean(axis=0)
#                print np.array(temp_array1[labels1_perm==conditions[0]]).shape
#                print temp_array1_0.shape
#                print temp_array1_1.shape
#                nii1=my_new_masker.inverse_transform(temp_array1_0)
#                nii2=my_new_masker.inverse_transform(temp_array1_1)
#                nii3=my_new_masker.inverse_transform(temp_array2_0)
#                nii4=my_new_masker.inverse_transform(temp_array2_1)
#                os.chdir(results_dir)        
#                nib.save(nii1, 'beta_'+SubjName[subj]+'_'+str(perm)+'_Perc'+'_cond0.nii')
#                nib.save(nii2, 'beta_'+SubjName[subj]+'_'+str(perm)+'_Perc'+'_cond1.nii')
#                nib.save(nii3, 'beta_'+SubjName[subj]+'_'+str(perm)+'_Im'+'_cond0.nii')
#                nib.save(nii4, 'beta_'+SubjName[subj]+'_'+str(perm)+'_Im'+'_cond1.nii')
########################################




print len(mask_index[0])
n_jobs=-1


for vox in range(0, 20): # len(mask_index[0])):
    os.chdir(cwd)
    alert, stat_map[vox], sphere_masker = my_cmpt_searchlight_new(n_of_subj, subj_data_all_1, subj_data_all_2, labels_all_1, labels_all_2, mask_index, vox, gm_mask)
    print stat_map[vox]
    
    if alert==0:
        stat_perm=[0]*n_of_subj
     #   perms=[0]*n_of_subj
      #  for perm in range(0, n_perm):
        for s in range(0, n_of_subj):
                img_cond_modality_1=[0]*2
                img_cond_modality_2=[0]*2
                      
                
                img_cond_modality_1[0]=np.squeeze(sphere_masker.fit_transform(subj_data_perm_1_1[s]))
                img_cond_modality_1[1]=np.squeeze(sphere_masker.fit_transform(subj_data_perm_1_2[s]))
                img_cond_modality_2[0]=np.squeeze(sphere_masker.fit_transform(subj_data_perm_2_1[s]))
                img_cond_modality_2[1]=np.squeeze(sphere_masker.fit_transform(subj_data_perm_2_2[s]))
                print img_cond_modality_1[0].shape
                print img_cond_modality_2[1].shape
#                stat_perm=(np.corrcoef(img_cond_modality_1[0], img_cond_modality_2[0])+np.corrcoef(img_cond_modality_1[1], img_cond_modality_2[1]) - \
#                    np.corrcoef(img_cond_modality_1[0], img_cond_modality_2[1])+np.corrcoef(img_cond_modality_1[1], img_cond_modality_2[0]))
#                print stat_perm
              #  print np.diag(np.corrcoef(img_cond_modality_1[1], img_cond_modality_2[1])[10:20, 0:10])
                stat_perm[s]=(np.diag(np.corrcoef(img_cond_modality_1[0], img_cond_modality_2[0])[1000:2000, 0:1000])+np.diag(np.corrcoef(img_cond_modality_1[1], img_cond_modality_2[1])[1000:2000, 0:1000]) - \
                    np.diag(np.corrcoef(img_cond_modality_1[0], img_cond_modality_2[1])[1000:2000, 0:1000])+np.diag(np.corrcoef(img_cond_modality_1[1], img_cond_modality_2[0])[1000:2000, 0:1000]))
             #   print stat_perm[s]
                
    #    for perm in range(0, 10):
    #        result[perm]=sl_cmpt_permutations_new(SubjName,sphere_masker, perm, stat_map[vox])
       #sign_map[vox]=sl_cmpt_permutations(labels_all_1, labels_all_2, all_maps_1, all_maps_2, n_of_subj, stat_map[vox])
   #    result = Parallel(n_jobs=n_jobs)(delayed(sl_cmpt_permutations_new)(SubjName,sphere_masker, perm, stat_map[vox]) for perm in range(0, n_perm)) #len(mask_index[0]))) #len(mask_index[0])))
  #  print np.sum(stat_perm, axis=0)


    sign_map[vox]=np.sum(np.sum(stat_perm, axis=0)< stat_map[vox])
    print sign_map[vox]
end = time.time()
print (end - start)

if __name__ == "__main__":
    # execute only if run as a script
    main()