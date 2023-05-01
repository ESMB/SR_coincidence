#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:46:35 2021

@author: Mathew
"""

import numpy as np
import pandas as pd
import imreg_dft as ird
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from skimage import filters,measure
import scipy
import scipy.ndimage
import scipy.stats
from PIL import Image
import napari
from PIL import Image
from skimage.io import imread
from skimage import io
from skimage.filters import threshold_otsu
# File information here
Pixel_size=117
image_width=int(684)
image_height=int(684)
scale=8
precision_thresh=30

# Thresholds that need changing are here
eps_threshold=1
minimum_locs_threshold=100

perform_mapping = 1

beads_image_path=r'/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/RS_STAPull_SR_Tetraspeck_posXY4_channels_t0_posZ0.tif'

pathlist=[]
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/48hr/1/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/48hr/2/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/48hr/3/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/48hr/4/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/48hr/5/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/48hr/6/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/48hr/7/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/48hr/8/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/72hr/1/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/72hr/2/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/72hr/3/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/72hr/4/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/72hr/5/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/72hr/6/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/72hr/7/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/72hr/8/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/72hr/9/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/24hr/1/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/24hr/2/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/24hr/3/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/24hr/4/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/24hr/5/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/24hr/6/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/24hr/7/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/24hr/8/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/24hr/9/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/96hr/1/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/96hr/1/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/96hr/2/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/96hr/3/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/96hr/4/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/96hr/5/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/96hr/6/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/96hr/7/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/96hr/8/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/DNA PAINT STAPull/96hr/9/")
filename1='Cy3b.csv'   # This is the name of the SR file containing the localisations from first channel
filename2='655.csv'    # This is the name of the SR file containing the localisations from second channel



def cluster(coords):
     db = DBSCAN(eps=eps_threshold, min_samples=minimum_locs_threshold).fit(coords)
     labels = db.labels_
     n_clusters_ = len(set(labels)) - (1 if-1 in labels else 0)  # This is to calculate the number of clusters.
     print('Estimated number of clusters: %d' % n_clusters_)
     return labels


def gkern(l,sigx,sigy):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    # ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx)/np.square(sigx) + np.square(yy)/np.square(sigy)) )
    # print(np.sum(kernel))
    # test=kernel/np.max(kernel)
    # print(test.max())
    return kernel/np.sum(kernel)

# Generate the super resolution image with points equating to the cluster number
def generate_SR(coords,clusters):
    SR_plot=np.zeros((image_width*scale,image_height*scale),dtype=float)
    j=0
    for i in clusters:
        if i>-1:
            
            xcoord=coords[j,1]
            ycoord=coords[j,0]
            scale_xcoord=round(xcoord*scale)
            scale_ycoord=round(ycoord*scale)
            SR_plot[scale_xcoord,scale_ycoord]+=1
            
        j+=1
    return SR_plot

# Generate SR image with width = precision
def generate_SR_prec(coords,precsx,precsy):
    box_size=20
    SR_prec_plot_def=np.zeros((image_width*scale,image_height*scale),dtype=float)
    dims=np.shape(SR_prec_plot_def)
    print(dims)
    j=0
    for i in coords:

      
        precisionx=precsx[j]/Pixel_size*scale
        precisiony=precsy[j]/Pixel_size*scale
        xcoord=coords[j,0]
        ycoord=coords[j,1]
        scale_xcoord=round(xcoord*scale)
        scale_ycoord=round(ycoord*scale)
        
        
        
        sigmax=precisionx
        sigmay=precisiony
        
        
        # tempgauss=SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
        
        tempgauss=gkern(2*box_size,sigmax,sigmay)
        
        # SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
        
        
        
        ybox_min=scale_ycoord-box_size
        ybox_max=scale_ycoord+box_size
        xbox_min=scale_xcoord-box_size
        xbox_max=scale_xcoord+box_size 
        
        
        if(np.shape(SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempgauss)):
            SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+tempgauss
        
        
           
        j+=1
    
    return SR_prec_plot_def

def generate_SR_prec_cluster(coords,precsx,precsy,clusters):
    box_size=50
    SR_prec_plot_def=np.zeros((image_width*scale+100,image_height*scale+100),dtype=float)
    SR_fwhm_plot_def=np.zeros((image_width*scale+100,image_height*scale+100),dtype=float)

    j=0
    for clu in clusters:
        if clu>-1:
       
            precisionx=precsx[j]/Pixel_size*scale
            precisiony=precsy[j]/Pixel_size*scale
            xcoord=coords[j,0]
            ycoord=coords[j,1]
            scale_xcoord=round(xcoord*scale)+50
            scale_ycoord=round(ycoord*scale)+50
            
            sigmax=precisionx
            sigmay=precisiony
            
            
            # tempgauss=SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
            tempgauss=gkern(2*box_size,sigmax,sigmay)
            ybox_min=scale_ycoord-box_size
            ybox_max=scale_ycoord+box_size
            xbox_min=scale_xcoord-box_size
            xbox_max=scale_xcoord+box_size 
        
        
            if(np.shape(SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempgauss)):
                SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+tempgauss
                
            tempfwhm_max=tempgauss.max()
            tempfwhm=tempgauss>(0.5*tempfwhm_max)
            
            tempfwhm_num=tempfwhm*(clu+1)
           
            
            if(np.shape(SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempfwhm)):
               plot_temp=np.zeros((2*box_size,2*box_size),dtype=float)
               plot_add=np.zeros((2*box_size,2*box_size),dtype=float)
               plot_temp=SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]
               plot_add_to=plot_temp==0
               
               plot_add1=plot_temp+tempfwhm_num
               
               plot_add=plot_add1*plot_add_to
               
               SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+plot_add
                
                
                # (SR_fwhm_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]+tempfwhm_num).where(SR_fwhm_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]==0)
                # SR_tot_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]=SR_tot_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]+tempfwhm
            
            # SR_tot_plot_def[SR_tot_plot_def==0]=1
            labelled=SR_fwhm_plot_def
            
            SR_prec_plot=SR_prec_plot_def[50:image_width*scale+50,50:image_height*scale+50]
            labelled=labelled[50:image_width*scale+50,50:image_height*scale+50]
            
            
        j+=1
    
    return SR_prec_plot,labelled,SR_fwhm_plot_def


def generate_SR_prec_cluster_coinc(coords,precsx,precsy,clusters,coinc):
    box_size=50
    SR_prec_plot_def=np.zeros((image_width*scale+100,image_height*scale+100),dtype=float)
    SR_fwhm_plot_def=np.zeros((image_width*scale+100,image_height*scale+100),dtype=float)

    j=0
    for clu in clusters:
        if clu in coinc:
       
            precisionx=precsx[j]/Pixel_size*scale
            precisiony=precsy[j]/Pixel_size*scale
            xcoord=coords[j,0]
            ycoord=coords[j,1]
            scale_xcoord=round(xcoord*scale)+50
            scale_ycoord=round(ycoord*scale)+50
            
            sigmax=precisionx
            sigmay=precisiony
            
            
            # tempgauss=SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
            tempgauss=gkern(2*box_size,sigmax,sigmay)
            ybox_min=scale_ycoord-box_size
            ybox_max=scale_ycoord+box_size
            xbox_min=scale_xcoord-box_size
            xbox_max=scale_xcoord+box_size 
        
        
            if(np.shape(SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempgauss)):
                SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+tempgauss
                
            tempfwhm_max=tempgauss.max()
            tempfwhm=tempgauss>(0.5*tempfwhm_max)
            
            tempfwhm_num=tempfwhm*(clu+1)
           
            
            if(np.shape(SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempfwhm)):
               plot_temp=np.zeros((2*box_size,2*box_size),dtype=float)
               plot_add=np.zeros((2*box_size,2*box_size),dtype=float)
               plot_temp=SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]
               plot_add_to=plot_temp==0
               
               plot_add1=plot_temp+tempfwhm_num
               
               plot_add=plot_add1*plot_add_to
               
               SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+plot_add
                
                
                # (SR_fwhm_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]+tempfwhm_num).where(SR_fwhm_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]==0)
                # SR_tot_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]=SR_tot_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]+tempfwhm
            
            # SR_tot_plot_def[SR_tot_plot_def==0]=1
            labelled=SR_fwhm_plot_def
            
            SR_prec_plot=SR_prec_plot_def[50:image_width*scale+50,50:image_height*scale+50]
            labelled=labelled[50:image_width*scale+50,50:image_height*scale+50]
            
            
        j+=1
    
    return SR_prec_plot,labelled,SR_fwhm_plot_def



def analyse_labelled_image(labelled_image):
    measure_image=measure.regionprops_table(labelled_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
 
    return number_of_features,labelled_image


if perform_mapping>0:
    beads = io.imread(beads_image_path)



    # Extract the channelTwo and channelOne parts of the image (only looks at the first frame, which is fine for beads)         
    GreenSlice = (beads[21,0:683,0:427]/255).astype(np.uint8)
    RedSlice = (beads[0,0:683,427:854]/255).astype(np.uint8)
    
    # Green = (beads[21,0:683,0:427])
    # Red = (beads[0,0:683,427:854])
    
    # GreenIm=Image.fromarray(Green)
    # GreenLarge=GreenIm.resize((684*8, 428*8))
    # GreenSlice_1=np.asarray(GreenLarge)
    # GreenSlice=(GreenSlice_1/255).astype(np.uint8)
    
    # RedIm=Image.fromarray(Red)
    # RedLarge=RedIm.resize((684*8, 428*8))
    # RedSlice_1=np.asarray(RedLarge)
    # RedSlice=(RedSlice_1/255).astype(np.uint8)

    # Perform the image registration
    result = ird.similarity(RedSlice, GreenSlice, numiter=5)
    tvec=result["tvec"].round(4)
    print("Translationis{},successrate{:.4g}" .format(tuple(tvec),result["success"]))
    
    channelOneSlice=GreenSlice
    channelTwoSlice=RedSlice
    
    # To look at the overlay- make binary
    thr_ch1 = threshold_otsu(channelTwoSlice)
    thr_ch2 = threshold_otsu(channelOneSlice)

    binary_ch1 = channelTwoSlice > thr_ch1
    binary_ch2 = channelOneSlice > thr_ch2
    
    # Make an RGB image for overlay
    
    imRGB = np.zeros((channelOneSlice.shape[0],channelOneSlice.shape[1],3))
    imRGB[:,:,0] = binary_ch1
    imRGB[:,:,1] = binary_ch2
    
    fig, ax = plt.subplots(1,3, figsize=(14, 4))
    
    ax[0].imshow(channelOneSlice,cmap='Greens_r')
    ax[0].set_title('channelOne')
    ax[1].imshow(channelTwoSlice,cmap='Reds_r');
    ax[1].set_title('channelTwo')
    ax[2].imshow(imRGB)
    ax[2].set_title('Overlay')
    
    
    # Now show the transformed image
    
    binary_ch2_transformed = result['timg'] > thr_ch2
    imRGB_t = np.zeros((channelOneSlice.shape[0],channelOneSlice.shape[1],3))
    imRGB_t[:,:,0] = binary_ch1
    imRGB_t[:,:,1] = binary_ch2_transformed
    
    
    fig, ax = plt.subplots(1,3, figsize=(14, 4))
    
    ax[0].imshow(result['timg'],cmap='Greens_r')
    ax[0].set_title('channelOne Transformed')
    ax[1].imshow(channelTwoSlice,cmap='Reds_r');
    ax[1].set_title('channelTwo')
    ax[2].imshow(imRGB_t)
    ax[2].set_title('Transformed Overlay')
    
    im = Image.fromarray(GreenSlice)
    im.save(beads_image_path+'Green.tif')
    
    im2 = Image.fromarray(RedSlice)
    im2.save(beads_image_path+'Red.tif')

    newresult=ird.transform_img_dict(GreenSlice, result, bgval=None, order=1, invert=False).astype('uint16')

    im2 = Image.fromarray(newresult)
    im2.save(beads_image_path+'Green_transformed.tif')

for path in pathlist:
    #  Load the data from channel 1
    data_df_1 = pd.read_csv(path+filename1)
    index_names = data_df_1[data_df_1['uncertainty [nm]']>precision_thresh].index
    data_df_1.drop(index_names, inplace = True)
    
    
    
    # Get the correct rows out
    coords_1 = np.array(list(zip(tvec[0]+data_df_1['x [nm]']/Pixel_size,-tvec[1]+data_df_1['y [nm]']/Pixel_size)))
    precs_1= np.array(data_df_1['uncertainty [nm]'])
    xcoords_1=np.array(tvec[0]+(data_df_1['x [nm]']/Pixel_size))
    ycoords_1=np.array(tvec[1]+(data_df_1['y [nm]']/Pixel_size))
    
    
    #  Load the data from channel 2
    data_df_2 = pd.read_csv(path+filename2)
    index_names = data_df_2[data_df_2['uncertainty [nm]']>precision_thresh].index
    data_df_2.drop(index_names, inplace = True)
    
    # Get the correct rows out
    coords_2 = np.array(list(zip(data_df_2['x [nm]']/Pixel_size,data_df_2['y [nm]']/Pixel_size)))
    precs_2= np.array(data_df_2['uncertainty [nm]'])
    xcoords_2=np.array(data_df_2['x [nm]']/Pixel_size)
    ycoords_2=np.array(data_df_2['y [nm]']/Pixel_size)
    
    
    
    
    #  Go through all of the first channel
       
    
    clusters_1=cluster(coords_1)
    
    SR_1=generate_SR(coords_1,clusters_1)
    
    
    SR_prec_1,labelled_1,SR_fwhm_plot_def_1=generate_SR_prec_cluster(coords_1,precs_1,precs_1,clusters_1)
    
    
    imsr = Image.fromarray(SR_prec_1)
    imsr.save(path+'SR_channel_1.tif')
    
    
    ims = Image.fromarray(SR_1)
    ims.save(path+'SR_points_channel_1.tif')
    
    
    
    #  Go through all of the second channel
       
    
    clusters_2=cluster(coords_2)
    
    SR_2=generate_SR(coords_2,clusters_2)
    
    
    SR_prec_2,labelled_2,SR_fwhm_plot_def_2=generate_SR_prec_cluster(coords_2,precs_2,precs_2,clusters_2)
    
    
    imsr = Image.fromarray(SR_prec_2)
    imsr.save(path+'SR_channel_2.tif')
    
    
    ims = Image.fromarray(SR_2)
    ims.save(path+'SR_points_channel_2.tif')
    
    
       
    # Check for coincidence:
    coinc=(labelled_2*labelled_1)>0
    
    #  Generate just labelled list
    coinc_labelled_1=coinc*labelled_1
    coinc_labelled_2=coinc*labelled_2
    
    # Need to go through the labeled image and find coincident numbers
    
    coinc_1_list, coinc_1_list_size = np.unique(coinc_labelled_1, return_counts=True)
    coinc_2_list, coinc_2_list_size = np.unique(coinc_labelled_2, return_counts=True)
    
    # Now need to generate SR images from coincident clusters only
    
    
    SR_prec_1_coinc,labelled_1_coinc,SR_fwhm_plot_def_1_coinc=generate_SR_prec_cluster_coinc(coords_1,precs_1,precs_1,clusters_1,coinc_1_list)
    
    imsr = Image.fromarray(SR_prec_1_coinc)
    imsr.save(path+'SR_channel_1_coinc.tif')
    
    
    SR_prec_2_coinc,labelled_2_coinc,SR_fwhm_plot_def_2_coinc=generate_SR_prec_cluster_coinc(coords_2,precs_2,precs_2,clusters_2,coinc_2_list)
    
    imsr = Image.fromarray(SR_prec_1_coinc)
    imsr.save(path+'SR_channel_1_coinc.tif')
    
    imsr = Image.fromarray(SR_prec_2_coinc)
    imsr.save(path+'SR_channel_2_coinc.tif')
    
    # number,labelled=label_image(binary)
    
    labelled_1_to_analyse=labelled_1.astype('int')
    measurements_1=analyse_labelled_image(labelled_1_to_analyse)
    labelled_2_to_analyse=labelled_2.astype('int')
    measurements_2=analyse_labelled_image(labelled_2_to_analyse)
    
    # areas=measurements_1['area']*((Pixel_size/8000)**2)
    # plt.hist(areas, bins = 20,range=[0,0.5], rwidth=0.9,color='#ff0000')
    # plt.xlabel('Area (\u03bcm$^2$)',size=20)
    # plt.ylabel('Number of Features',size=20)
    # plt.title('PSD-95 area',size=20)
    # plt.show()
    plt.show()
    
    length=measurements_1['major_axis_length']*((Pixel_size/8))
    plt.hist(length, bins = 20,range=[0,1000], rwidth=0.9,color='#ff0000')
    plt.xlabel('Length (nm)',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.title('Image 1 Lengths',size=20)
    plt.savefig(path+'Coincident_1_lengths.pdf')
    plt.show()
    length=measurements_2['major_axis_length']*((Pixel_size/8))
    plt.hist(length, bins = 20,range=[0,1000], rwidth=0.9,color='#ff0000')
    plt.xlabel('Length (nm)',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.title('Image 2 Lengths',size=20)
    plt.savefig(path+'Coincident_2_lengths.pdf')
    plt.show()
    
    
    measurements_1.to_csv(path + '/' + 'Channel_1_Coinc.csv', sep = '\t')    
    measurements_2.to_csv(path + '/' + 'Channel_2_Coinc.csv', sep = '\t')  
    
    Number_1_clusters=coinc_labelled_1.max()
    Number_2_clusters=coinc_labelled_2.max()
    
    Number_1_coinc=len(coinc_1_list)
    Number_2_coinc=len(coinc_2_list)
    
    text_file = open(path+"Results.txt", "w")
    n = text_file.write('Total clusters in channel 1: %d \n'%Number_1_clusters)
    n = text_file.write('Total clusters in channel 2: %d \n'%Number_2_clusters)
    n = text_file.write('Coincident clusters in channel 1: %d \n'%Number_1_coinc)
    n = text_file.write('Coincident clusters in channel 2: %d \n'%Number_2_coinc)
    text_file.close()