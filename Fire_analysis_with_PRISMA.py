#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import scipy.stats as st
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import cv2
import glob
import gdal
import math
import time       
import warnings
from joblib import Parallel, delayed
from itertools import compress
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from random import seed
from random import random, uniform
from osgeo import osr
from shapely.geometry import Polygon, Point
from datetime import date
from tqdm import tqdm

seed(1) # seed random number generator

#%% Functions

def extract_satur_pixels(band,satur_hypercube,wl_SWIR):
    temp = [np.where(satur_hypercube[1] == band)[0]][0]
    print(f'Ci sono {len(temp)} pixel saturati nella banda {band} di SWIR con lunghezza d\'onda {wl_SWIR[band]} nm')
    return {'num_satur_pixels':len(temp),
            'band':band,
            'wavelength':wl_SWIR[band],
            'pixels_array':np.array(list(zip(satur_hypercube[0][temp], 
                                             satur_hypercube[2][temp])))}


def dms(deg):
    f,d = math.modf(deg)
    s,m = math.modf(abs(f) * 60)
    return (d,m,s * 60)


def extract_product_level(folder,level):
    '''Extracting the PRISMA image located in the required folder'''
    files = glob.glob(folder+'/**/*.he5', recursive=True)
    search_output = list(map(lambda x:  level in x, files))
    return next(compress(files, search_output))

def search_band(bands,cw,target_wavelength):
    '''This function extracts the band from the datacube which is closer to the requested wavelength'''
    if len(cw) != bands.shape[1]:
        raise ValueError('Dimensions do not match!')
    
    temp = np.abs(cw - target_wavelength)
    target_band_index = np.where(temp == temp.min())[0]
    print(f'Index of target wavelength {target_wavelength} nm is {target_band_index}')
    print(f'  The optimal PRISMA wavelength is {cw[target_band_index]} nm.')
    target_band = np.squeeze(bands[:,target_band_index,:])
    return target_band

def select_bkg_point(fire_points,padding,N):
    '''This function return the indices of the background points for each fire pixel.
    Input:
    fire_points=list of the fire points
    padding=max distance to search for bkg points
    N=number of bkg points
    Output= dictionary reporting the list of bkg points for each fire pixel'''
    results = {}
    for i, x in enumerate(fire_points):
        results_x = []
        for _ in range(N):
            countur = x + [int((1-2*random())*padding),int((1-2*random())*padding)]
            cond1 = (countur[0] > 999) or (countur[0] < 0) # x within image range
            cond2 = (countur[1] > 999) or (countur[1] < 0) # y within image range
            cond3 = countur in fire_points  # x not in fire points
            while (cond1 | cond2 | cond3):
                countur = x + [int((1-2*random())*padding),int((1-2*random())*padding)]
                cond1 = (countur[0] > 999) or (countur[0] < 0) # x within image range
                cond2 = (countur[1] > 999) or (countur[1] < 0) # y within image range
                cond3 = countur in fire_points  # x not in fire points
            
            results_x.append(list(countur))
        results[i] = {'row':x[0], 'col': x[1], 'points': results_x}
    return results

def Planck_function(x, Temp, emissivity = 1.0):
    # from https://ncc.nesdis.noaa.gov/data/planck.html
    #wavelength = np.arange(100,10001,1).astype('float')*1e-3 #micrometri
    c1 = 1e-3*1.191042*1e8 # 1e-3 per mettere in nanometri!
    c2 = 1.4387752*1e4
    y_planck_T = emissivity*(c1/(x**5))*(1/(np.exp(c2/(Temp*x))-1))
    return y_planck_T
    
class fitClass:

    def __init__(self):
        self.bkg              = []
        self.emissivity       = []
        self.N_sources        = []
        self.absorption_bands = []
        
    def linear_mixture_function(self, x, *args):
        sum_fire_sources = 0
        bkg              = self.bkg
        emissivity       = self.emissivity
        N_sources        = self.N_sources
        absorption_bands = self.absorption_bands

        Norm_coeffs = np.sum(args[N_sources:])
        
        for i in range(N_sources):
            Temp = args[i]
            p_fire = args[i + N_sources]/Norm_coeffs

            x = x*1e-3 #micrometri
            y_planck_T = Planck_function(x, Temp, emissivity)
            y_planck_T[absorption_bands] = 0
            
            # _ = plt.figure()
            # plt.plot(x,y_planck_T)
            # plt.title('planck')
            # plt.show()
            
            sum_fire_sources += p_fire*y_planck_T

        sum_backgrounds = 0    

        p_bkg = np.reshape(args[N_sources*2:],(bkg.shape[0],1))/Norm_coeffs
        sum_backgrounds = np.sum(bkg*p_bkg,0)
        fitted_curve = sum_fire_sources + sum_backgrounds

        return fitted_curve

    def mixt_grad(self, x, *args):
        
        Temp = args[0]
        
        emissivity = self.emissivity
        Norm_coeffs = np.sum(args[1:])
        p_fire = args[1]/Norm_coeffs
        
        x = x*1e-3 #micrometri
        c1 = 1e-3*1.191042*1e8 # 1e-3 per mettere in nanometri!
        c2 = 1.4387752*1e4
        y_planck_T = emissivity*(c1/(x**5))*(1/(np.exp(c2/(Temp*x))-1))
         
        y_planck_T_der = p_fire*2*emissivity*(c1/(x**5))*(1/(np.exp(c2/(Temp*x))-1))**2*c2/(x*Temp**2)*np.exp(c2/(Temp*x))
                
        bkg = self.bkg
        
        grad = np.vstack([y_planck_T_der,y_planck_T/Norm_coeffs,bkg/Norm_coeffs]).T
        return grad
        
def estimate_temperature(ydata, xdata, bkg, emissivity, N_sources, 
                         plot, verbose, T_min, T_max, opt_method):
    
    
    # Initial guess 
    x0 = [uniform(T_min, T_max)]*N_sources + [0.5]*N_sources + [1/(bkg.shape[0]*2)]*bkg.shape[0]# + [0,0]*N_sources 
    x0 = np.array(x0)   

    bounds = [(T_min,T_max)]*N_sources + [(0.05,1)]*N_sources + [(0,1)]*bkg.shape[0]#+ [(0,1e-16),(0,1e-16)]*N_sources 
    bounds = np.array(bounds).T

    # Optimization
    inst = fitClass()
    inst.bkg                = bkg
    inst.emissivity         = emissivity
    inst.N_sources          = N_sources
    inst.absorption_bands   = ydata < 0.003
    
    try:
        # try the optimization process
        popt = optimization.curve_fit(inst.linear_mixture_function, xdata, ydata, 
                                      x0, bounds=bounds, gtol = 1e-8, ftol = 1e-8,
                                      xtol = 1e-8, verbose = verbose, method = opt_method,
                                      max_nfev=None) #max_nfev=5000, jac = inst.mixt_grad, loss = 'cauchy'
    except:
        # if the optimization process is not successful, then set a void output
        warnings.warn("The optimization process was not successful...")
        popt = []
        
    if (plot == True) and (len(popt) > 1):
        _ = plt.figure()
        a = inst.linear_mixture_function(xdata, *popt[0])
        plt.style.use('ggplot')
        plt.plot(xdata, a, color = 'red',label = 'Linear mixture model')
        plt.plot(xdata, ydata, color = 'blue', label = 'Fire Signal')
        #plt.plot(x_ref[:-1], bkg[1], color='yellow')
        plt.legend(loc='upper right',fontsize=15)
        plt.xlabel('Wavelength ($nm$)',fontsize=15)
        plt.ylabel('Spectral radiance ($W/(m^2\cdot sr \cdot nm)$)',fontsize=15)
    elif len(popt) == 0:
        print('No plot generated as no feasible solution has been found.')
        
    if len(popt) > 0:
        Norm_coeffs = np.sum(popt[0][N_sources:])
        output = [list(popt[0][:N_sources]),list(popt[0][N_sources:]/Norm_coeffs)]
    else:
        output = []

    return output

def main_temp_estimation(DATA_CUBE_2B, pixel_number, xdata, fire_pixel,  
                         fire_pixel_2D, bkg_points_HFDI, config, use_saved_bkg,
                         bkg0):

    results = []
    
    if config['only_SWIR'] == True:
        index_SWIR = xdata > 1400
        if config['only_short_SWIR'] == True:
            index_SWIR = (xdata > 1400) & (xdata < 1900)
            #print('Used SWIR 1400-1900 nm')
        elif config['only_feasible_SWIR'] == True:
            if np.max(fire_pixel_2D[index_SWIR]) > 0.8:
                # use all the SWIR up to 2300 nm, after which there is too noise
                index_satur = np.where(fire_pixel_2D[index_SWIR] > 0.8)[0][0]
                upper_bound = max([1900, xdata[index_SWIR][index_satur]])
                index_SWIR = (xdata > 1400) & (xdata < upper_bound)
                #print(f'Used SWIR 1400-{upper_bound} nm')
            else:
                # use only the SWIR up to 1900 nm because after that there is saturation
                index_SWIR = (xdata > 1400) & (xdata < 1900)
                #print('Used SWIR 1400-1900 nm')

    elif config['only_short_SWIR'] == True:
        index_SWIR = (xdata > 1400) & (xdata < 1900)
        #print('Used SWIR 1400-1900 nm')
    else:
        index_SWIR = xdata > 0
        #print('Used all SWIR channels')

    # Data for active fire pixels
    xdata = xdata[index_SWIR]
    fire_pixel = fire_pixel[index_SWIR]
    # _ = plt.figure()
    # print(xdata.shape,fire_pixel.shape)
    # plt.plot(xdata,fire_pixel)
    # plt.title('fire')
    # plt.show()
    
    
    for i in range(config['N_trials']):
        #print('\r' + f'\tIteration {i+1}/{config["N_trials"]}', end= '       ')

        
        
        # _ = plt.figure()
        # for i in range(10):
        #     bkg_i = bkg[i,:]
        #     plt.plot(xdata,bkg_i)
        #     plt.title('bkg')
        #     plt.show()
        counter = 0
        result_i = []
        while (len(result_i) == 0) & (counter < 5):
            
            if use_saved_bkg == False:
                idx = np.random.randint(len(bkg_points_HFDI), size = config['N_bkg_points'])
                bkg = np.squeeze(DATA_CUBE_2B[bkg_points_HFDI[idx,0],:, bkg_points_HFDI[idx,1]])
            else:
                idx = np.random.randint(bkg0.shape[0], size = config['N_bkg_points'])
                bkg = bkg0[idx,:]
            
            bkg = bkg[:,index_SWIR]
            # call for the temperature estimation function
            result_i = estimate_temperature(fire_pixel, xdata, bkg, 
                                            config['emissivity'], config['N_sources'],
                                            config['plot'], config['verbose'],
                                            config['T_min'], config['T_max'],
                                            config['method'])
            counter += 1
                                        
        if len(result_i) > 0:
            # append only valid temperature estimations
            results.append(result_i)

    Temps = []
    p_coeff = []
    
    # number of valid temperature estimations
    valid_estimates = len(results)
    
    if valid_estimates > 0:
        # at least one valid temperature estimation is provided
        for i in range(valid_estimates):
            Temps.append(results[i][0][0])
            p_coeff.append(results[i][1][0])
        
        
        if valid_estimates > 1:
            conf95_temp = st.t.interval(0.95, len(Temps)-1, loc=np.mean(Temps), scale=st.sem(Temps))
            #print(f'\n\tT -> Mean: {np.mean(Temps)} K, std: {np.std(Temps)} K (95% C.I. [{conf95_temp[0]}, {conf95_temp[1]}] K)') 
        else:
            #print(f'\n\tT -> Mean: {np.mean(Temps)} K')
            conf95_temp = [np.nan,np.nan]
            
        if valid_estimates > 1:
            conf95_p_coeff = st.t.interval(0.95, len(p_coeff)-1, loc=np.mean(p_coeff), scale=st.sem(p_coeff))
            #print(f'\tp -> Mean: {np.mean(p_coeff)}, std: {np.std(p_coeff)} (95% C.I. [{conf95_p_coeff[0]}, {conf95_p_coeff[1]}])') 
        else:
            #print(f'\tp -> Mean: {np.mean(p_coeff)}')
            conf95_p_coeff = [np.nan,np.nan]
    else:
        # no valid temeprature estimations are given
        p_coeff = 0
        conf95_p_coeff = [np.nan,np.nan]
        Temps = np.nan
        conf95_temp = [np.nan,np.nan]
        
    #print('\n**************************************\n')
    result = {'Pixel': pixel_number,
            'emissivity': config['emissivity'],
            'mean_P_coeff': np.mean(p_coeff),
            'std_P_coeff': np.std(p_coeff),
            'CI_P_coeff': [conf95_p_coeff[0], conf95_p_coeff[1]],
            'mean_temp': np.mean(Temps), 
            'std_temp': np.std(Temps), 
            'CI_temp': [conf95_temp[0], conf95_temp[1]]}
    
    return result

def geoTIFF_creation(numRows, numCols, numChannels, channels, lat, 
                         lon, dtype, name):
    """
    numRows: number of rows in the image
    numCols: number of cols in the image
    numChannels: number of channel in the image
    channels: image channels to be put in the geoTIFF
    lat: lat grid
    lon: lon grid
    dtype: datat type for the geoTIFF output
    name: output name (with tif extension)
        
    """
    if dtype == 'uint32':
        dst_ds = gdal.GetDriverByName('GTiff').Create(name, numRows, 
                                                      numCols, numChannels, 
                                                      gdal.GDT_UInt32)
    elif dtype == 'float32':
        dst_ds = gdal.GetDriverByName('GTiff').Create(name, numRows, 
                                                      numCols, numChannels, 
                                                      gdal.GDT_Float32)
    
    # Evaluate the rotation of the polygon by projecting in a projected CRS
    pol_deg_2B = Polygon([[lon[0,0], lat[0,0]], 
                         [lon[0,-1], lat[0,-1]], 
                         [lon[-1,-1], lat[-1,-1]], 
                         [lon[-1,0], lat[-1,0]], 
                         [lon[0,0], lat[0,0]]])
    
    # x,y = pol_deg_2B.exterior.xy
    # plt.plot(x,y,color='r')
    # x1,y1 = pol_deg.exterior.xy
    # plt.plot(x1,y1,color='b')
    
    d = {'unit': ['degree'], 'geometry': [pol_deg_2B]}
    gdf_deg = gpd.GeoDataFrame(d, crs="EPSG:4326")
    gdf_meter = gdf_deg.geometry.to_crs('epsg:3857')
    x,y = gdf_meter.geometry[0].exterior.xy
    row_rot = np.abs((y[0] - y[1])/(x[0] - x[1]))
    col_rot = np.abs((x[0] - x[-2])/(y[0] - y[-2]))
    rot = 0.5*(row_rot + col_rot) # mean rotation angle (best estimation possible
    # of the real rotation angle)
    # plt.plot(x,y)
    # plt.scatter(x[0],y[0])
    # plt.scatter(x[1],y[1],color='r')
    # plt.scatter(x[2],y[2],color='g')
    # plt.scatter(x[3],y[3],color='y')
    
    # Evaluating the geotransform information
    nx = numCols
    ny = numRows
    
    # 
    #                   -----
    #                 --     -----  side_a
    #       side_b  --            -----
    #             --                    -----
    # 		    --                          --
    #         --                          --
    #       --                          --
    #         -----	                  --
    # 		       -----            --
    # 			        -----     --
    # 				         -----
    side_a = (np.max(lon) - lon[0,0])/np.cos(rot)
    side_b = (lat[lon==np.min(lon)][0] - lat[0,0])/np.cos(rot)
    xres = side_a/nx
    yres = side_b/ny
    geotransform_2B = (lon[0,0], xres*np.cos(rot), -xres*np.sin(rot), 
                       lat[0,0], yres*np.sin(rot),  yres*np.cos(rot))
    
    # plt.scatter(lon_2B[0,0],lat_2B[0,0])
    # plt.scatter(lon_2B[0,-1],lat_2B[0,-1],color='r')
    # plt.scatter(lon_2B[-1,0],lat_2B[-1,0],color='g')
    
    # Saving the GeoTiff image
    dst_ds.SetGeoTransform(geotransform_2B)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(4326)                # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    for i in range(numChannels):
        if dtype == 'uint32':
            dst_ds.GetRasterBand(i+1).WriteArray(channels[i].astype('uint32'))   # write r-band to the raster
        elif dtype == 'float32':
            dst_ds.GetRasterBand(i+1).WriteArray(channels[i].astype('float32'))   # write r-band to the raster
        
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None

    # print(lat_2B[-1,0],lon_2B[-1,0])
    # print(np.min(lat_2B),np.max(lon_2B))

def eval_2D_points(ind,coord, coords_2D):
        
    eps_lat = (coords_2D[0,:,:] - coord[0])**2
    eps_lon = (coords_2D[1,:,:] - coord[1])**2
    sum_eps = eps_lat + eps_lon
    pos_min = np.where(sum_eps == np.min(sum_eps))
    Feasible_Fire_points_2D = [ind,pos_min[0][0],pos_min[1][0]]
    return Feasible_Fire_points_2D
       
def extract_points_in_AOI(polygon,lon_2B,lat_2B):
    """ This function extracts the indices of the points within the areas of
    Interest.
    
    output: list of tuples
    """
    points = [(x,y) for y in range(0,1000) for x in range(0,1000) if \
              polygon.contains(Point(lon_2B[x,y],lat_2B[x,y]))]
    return points
         

def main(): 
    #%% CONSTANTS OF THE SOFTWARE
        
    save_results            = False
    PLOT_IMAGES             = False
    enlarge_AOI_L2D         = False
    enlarge_AOI_L2B         = False # To be put to True if you want pure rectangles in the 2B image
    prescribed_AOI          = True
    run_parallel            = True
    use_L2D_level           = False
    use_L1_level            = True
    fire_points_from_HFDI   = False
    num_img                 = 'Im3'
    ROI_img                 = 'Oregon' # Oregon o Australia
    HFDI_thr                = -0.2 # 0.2 è definito da analisi con Stefania
    only_plot_prev_results  = False
    use_saved_bkg           = True
    
    
    if only_plot_prev_results == True:
        date_results = '16-03-2022'
        prev_res = 'results_temperature' + os.path.sep + f'Results_{ROI_img}_{num_img}_{date_results}.csv'
        if os.path.exists(prev_res) == False:
           raise ValueError('No previous results found for that image or day.') 
           
    config = {}
    config['only_SWIR']             = True
    config['only_short_SWIR']       = True
    config['only_feasible_SWIR']    = False
    config['plot']                  = False
    config['N_sources']             = 1
    config['N_bkg_points']          = 20
    config['N_trials']              = 15 # number of different estimates to provide the final temperature
    config['emissivity']            = 1
    config['verbose']               = 0
    config['T_min']                 = 100 # temperature più basse possono portare a errori!
    config['T_max']                 = 2000
    config['method']                = 'trf' # 'trf', 'dogbox'
    
    if prescribed_AOI == True:
        ROI_from_ENVI = []
        for j in range(1,5,1):
            ROI_from_ENVI.append(gpd.read_file('ROI-PRISMA' + os.path.sep + f'ROI-PRISMA-area{j}.shp'))
         
        geoms = [x.geometry[0] for x in ROI_from_ENVI]
        aoi = gpd.GeoSeries(geoms)
    
    
    
    #%% *********************************************************************** %%#
    #                     Locating level 2B and level 2D images
    # *************************************************************************** #
    
    folder = 'Oregon' + os.path.sep + 'PRISMA' + os.path.sep + num_img
    Level_2D = extract_product_level(folder,'L2D')
    Level_2B = extract_product_level(folder,'L2B')
    Level_1 = extract_product_level(folder,'L1')
    
    #%% *********************************************************************** %%#
    #                            Working with level 2D
    # *************************************************************************** #
    
    if use_L2D_level == True:
        
        code_img = Level_2D
        
        raster_path = os.getcwd()  + os.path.sep + code_img
        
        ## Open HDF file
        hdflayer = gdal.Open(raster_path, gdal.GA_ReadOnly)
        
        # Get the file metadata
        metadata = hdflayer.GetMetadata()
        #print(metadata)
        
        # List of the datasets within the file
        datasets_list = hdflayer.GetSubDatasets()
        
        print('List of the Datasets in the PRISMA L2D file:')
        for num,elem in enumerate(datasets_list):
            print(num, ' - ', elem[0].split('/')[-1])
        
        # Extract the SWIR bands 
        SWIR_2D = datasets_list[0][0]
              
        # Extract the NVIR bands 
        VNIR_2D = datasets_list[2][0]
              
        # Extract the hypercube LAT
        LAT_2D = datasets_list[4][0]
              
        # Extract the hypercube LON
        LON_2D = datasets_list[5][0]
              
                
        CNM_SWIR_2D = metadata['List_Cw_Swir_Flags'].split(' ')
        CNM_SWIR_2D = np.array([int(x) for x in CNM_SWIR_2D[:-1]])
        
        CNM_VNIR_2D = metadata['List_Cw_Vnir_Flags'].split(' ')
        CNM_VNIR_2D = np.array([int(x) for x in CNM_VNIR_2D[:-1]])
        
        bands_VNIR_2D = gdal.Open(VNIR_2D, gdal.GA_ReadOnly).ReadAsArray()[:,CNM_VNIR_2D==1,:] # deleting the all-zero elements
        bands_SWIR_2D = gdal.Open(SWIR_2D, gdal.GA_ReadOnly).ReadAsArray()[:,CNM_SWIR_2D==1,:] # deleting the all-zero elements
        
        VNIR_array_2D = float(metadata['L2ScaleVnirMin']) + (float(metadata['L2ScaleVnirMax']) - float(metadata['L2ScaleVnirMin']))*(bands_VNIR_2D/((2**16)-1))  # deleting the all-zero elements
        SWIR_array_2D = float(metadata['L2ScaleSwirMin']) + (float(metadata['L2ScaleSwirMax']) - float(metadata['L2ScaleSwirMin']))*(bands_SWIR_2D/((2**16)-1)) # deleting the all-zero elements
        
        lat_2D = gdal.Open(LAT_2D, gdal.GA_ReadOnly).ReadAsArray()
        lon_2D = gdal.Open(LON_2D, gdal.GA_ReadOnly).ReadAsArray()
        print('mean lat,lon in decimal degree:', [np.mean(lat_2D),np.mean(lon_2D)])
        print('mean lat in dms:', dms(np.mean(lat_2D)))
        print('mean lon in dms:', dms(np.mean(lon_2D)))
        
        
        print('Start time:', metadata['Product_StartTime'], '. Stop time:', metadata['Product_StopTime'])
        
        
        SWIR_CW_2D = metadata['List_Cw_Swir'].split(' ')
        SWIR_CW_2D = np.array([float(x) for x in SWIR_CW_2D[:-1]])[CNM_SWIR_2D==1]
        
        VNIR_CW_2D = metadata['List_Cw_Vnir'].split(' ')
        VNIR_CW_2D = np.array([float(x) for x in VNIR_CW_2D[:-1]])[CNM_VNIR_2D==1]
        wavelenghts_2D = np.append(SWIR_CW_2D, VNIR_CW_2D)
        sort_array_2D = np.argsort(wavelenghts_2D)
        wavelenghts_2D = wavelenghts_2D[sort_array_2D]
        
        # Array containing both SWIR and VNIR
        DATA_CUBE = np.concatenate((SWIR_array_2D, VNIR_array_2D),axis=1)
        DATA_CUBE = DATA_CUBE[:,sort_array_2D,:]
        
        
        
        
        if len(VNIR_CW_2D) != VNIR_array_2D.shape[1]:
            raise ValueError('Dimensions do not match!')
        else:
            IMG_red = search_band(VNIR_array_2D,VNIR_CW_2D,632) #660
            IMG_green = search_band(VNIR_array_2D,VNIR_CW_2D,530) #660
            IMG_blue = search_band(VNIR_array_2D,VNIR_CW_2D,463) #660
            IMG_660 = search_band(VNIR_array_2D,VNIR_CW_2D,660) #660
            IMG_770 = search_band(VNIR_array_2D,VNIR_CW_2D,770)
            IMG_780 = search_band(VNIR_array_2D,VNIR_CW_2D,780)
        
            
        if len(SWIR_CW_2D) != SWIR_array_2D.shape[1]:
            raise ValueError('Dimensions do not match!')
        else:
            IMG_1100 = search_band(SWIR_array_2D,SWIR_CW_2D,1100) #1100
            IMG_1700 = search_band(SWIR_array_2D,SWIR_CW_2D,1700) #1700
            #IMG_1990 = search_band(SWIR_array_2D,SWIR_CW_2D,1990)
            #IMG_2010 = search_band(SWIR_array_2D,SWIR_CW_2D,2010)
            #IMG_2040 = search_band(SWIR_array_2D,SWIR_CW_2D,2040)
            IMG_2060 = search_band(SWIR_array_2D,SWIR_CW_2D,2060)
            IMG_2430 = search_band(SWIR_array_2D,SWIR_CW_2D,2430)
            #IMG_1088 = search_band(SWIR_array_2D,SWIR_CW_2D,1088)   
            #IMG_2200 = search_band(SWIR_array_2D,SWIR_CW_2D,2200) 
        
        
        
        #%% Generating the false color RGB composite image
        
        red = IMG_1700
        green = IMG_1100
        blue = IMG_660
        IM = np.array([red,green,blue])#.reshape(1181,1203,3)
        w,h = IM.shape[1],IM.shape[2]
        print('Dimensions of the image:', w,h)
        IM = IM.reshape((3,w,h))
        print('Dimensions of the pre-processed RGB image:',IM.shape)
        
        t=(w,h,3)
        FalseColorComp = np.zeros(t)
        for i in range(w):
            for j in range(h):
                FalseColorComp[i,j] = [red[i,j],green[i,j],blue[i,j]]
        print('Dimensions of the post-processed RGB image:', FalseColorComp.shape)
        
        
        #%% Generating the true color RGB image
        
        red = IMG_red # 36=632.13165 nm, 35=641.33325 nm
        green = IMG_green # 48=530.66705 nm, 45=554.5646 nm
        blue = IMG_blue # 463.731 nm
        IM = np.array([red,green,blue])#.reshape(1181,1203,3)
        w,h = IM.shape[1],IM.shape[2]
        print('Dimensions of the image:', w,h)
        IM = IM.reshape((3,w,h))
        print('Dimensions of the pre-processed RGB image:', IM.shape)
        
        t=(w,h,3)
        RGBComp=np.zeros(t)
        for i in range(w):
            for j in range(h):
                RGBComp[i,j]=[red[i,j],green[i,j],blue[i,j]]
        print('Dimensions of the post-processed RGB image:', RGBComp.shape)
        
        #%% Searching for saturated pixels in SWIR
        
        th = 0.48
        
        # Searching saturated pixels only in the part os SWIR we are interested in
        index_SWIR = SWIR_CW_2D > 2200
        saturated_hypercube = np.where(SWIR_array_2D[:,index_SWIR,:] >= th)
        saturated_bands = np.unique(saturated_hypercube[1])
        
        saturated_pixels = []
        for num,ind in enumerate(saturated_bands):
            saturated_pixels.append(extract_satur_pixels(ind,saturated_hypercube,SWIR_CW_2D))
        
        saturated_pixels = pd.DataFrame(saturated_pixels)
        index_satur = saturated_pixels.sort_values('num_satur_pixels',ascending=False).index[0]
        
        
        if PLOT_IMAGES == True:
            sns.set(rc={'figure.figsize':(10, 3)})
            plt.scatter(saturated_pixels['wavelength'],saturated_pixels['num_satur_pixels'])
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('N. saturated pixels')
            # plt.savefig( os.getcwd() + os.path.sep + 'Num_saturated_pixels.png',dpi=300,bbox_inches='tight')
            plt.show()
        
        
        Feasible_Fire_points_2D = saturated_pixels.iloc[index_satur].pixels_array
        
        if enlarge_AOI_L2D == True:
            img = np.zeros([SWIR_array_2D.shape[0],SWIR_array_2D.shape[2]],dtype=np.uint8)
            for i in range(len(Feasible_Fire_points_2D)):
                x,y = Feasible_Fire_points_2D[i]
                img[x,y] = 1
            # finding the clusters of feasible fire points
            num_labels, labels_im = cv2.connectedComponents(img)
            
            # surrounding each cluster with a rectangle with n pixel margin
            margin = 1
            labels_im2 = labels_im.copy()
            for i in range(1,num_labels):
                aa = np.where(labels_im2==i)
                if len(aa[0]) > 0:
                    x_min,y_min,x_max,y_max = np.min(aa[0]),np.min(aa[1]),np.max(aa[0]),np.max(aa[1])
                    labels_im2[x_min-margin:x_max+margin+1, y_min-margin:y_max+margin+1] = 1
        
            Feasible_Fire_points_2D = np.where(labels_im2==1)
            Feasible_Fire_points_2D = np.array(Feasible_Fire_points_2D).T
            # def imshow_components(labels):
            #     # Map component labels to hue val
            #     label_hue = np.uint8(179*labels/np.max(labels))
            #     blank_ch = 255*np.ones_like(label_hue)
            #     labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            
            #     # cvt to BGR for display
            #     labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            
            #     # set bg label to black
            #     labeled_img[label_hue==0] = 0
            
            #     cv2.imshow('labeled.png', labeled_img)
            #     cv2.waitKey()
            
            # imshow_components(labels_im)
        
        fire_lons = lon_2D[Feasible_Fire_points_2D[:,0],Feasible_Fire_points_2D[:,1]]
        fire_lats = lat_2D[Feasible_Fire_points_2D[:,0],Feasible_Fire_points_2D[:,1]]
        
        fire_coords = np.array([fire_lats,fire_lons])
        print(fire_coords.shape)
        
        #%% Figure with the feasible fire points
        
        if PLOT_IMAGES == True:
            # name_fig = code_img.split("\\")[0]
            
            # initialize the figure
            _,ax = plt.subplots(figsize = (16,16))
            
            # Define lats and longs for the figure
            lons_fig = [np.min(lon_2D), np.max(lon_2D)]
            lats_fig = [np.min(lat_2D), np.max(lat_2D)]
            lat_0 = np.mean(lats_fig)
            lon_0 = np.mean(lons_fig)
            
            # initialize the map
            map_kwargs = dict(projection='merc', resolution='l',
                              llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                              llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                              lat_0=lat_0, lon_0=lon_0)
            m = Basemap(**map_kwargs)
            
            # drawing parallels and meridians
            m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            
            ## What you put in for the image doesn't matter because of the color mapping
            m.pcolormesh(lon_2D, lat_2D, SWIR_array_2D[:,3,:], latlon=True)
            
            # Super-imposing the scatter plot of the saturated pixels
            x, y = m(lon_2D[Feasible_Fire_points_2D[:,0],Feasible_Fire_points_2D[:,1]], lat_2D[Feasible_Fire_points_2D[:,0],Feasible_Fire_points_2D[:,1]])  # transform coordinates
            m.scatter(x,y, 10, marker='o', color='Red')
            
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            cb = plt.colorbar(cax=cax)
            cb.ax.tick_params(labelsize=25)
            
            # Labels on axes
            #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
            #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
            
            # Showing image
            # plt.savefig( os.getcwd() + os.path.sep + f'far_SWIR_{name_fig}.png',dpi=300,bbox_inches='tight')
            plt.show()
        
        
        #%% RGB figure
        
        if PLOT_IMAGES == True:
            # initialize the figure
            fig,ax = plt.subplots(figsize = (16,16))
            
            # Define lats and longs for the figure
            lons_fig = [np.min(lon_2D), np.max(lon_2D)]
            lats_fig = [np.min(lat_2D), np.max(lat_2D)]
            lat_0 = np.mean(lats_fig)
            lon_0 = np.mean(lons_fig)
            
            # initialize the map
            map_kwargs = dict(projection='merc', resolution='l',
                              llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                              llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                              lat_0=lat_0, lon_0=lon_0)
            m = Basemap(**map_kwargs)
            
            # drawing parallels and meridians
            m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                           color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            
            # drawing the PRISMA image
            mesh_rgb = RGBComp[:, :-1, :]/np.max(RGBComp)  # also RGBComp[:, 1:, :] works fine
            colorTuple = mesh_rgb.reshape((mesh_rgb.shape[0] * mesh_rgb.shape[1]), 3)
            alpha = 1.0
            colorTuple = np.insert(colorTuple,3,alpha,axis=1)
            
            ## What you put in for the image doesn't matter because of the color mapping
            m.pcolormesh(lon_2D, lat_2D, RGBComp[:,:,0], latlon=True, color=colorTuple)
            
            # Labels on axes
            #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
            #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
            # Showing image
            ax.set_aspect('equal')
            #plt.savefig( os.getcwd() + os.path.sep + f'fire_image_{name_fig}.png',dpi=300,bbox_inches='tight')
            #plt.savefig( os.getcwd() + os.path.sep + 'fire_image_Australia_no_grid.png',dpi=100,bbox_inches='tight')
            plt.show()
        
        
        
        #%% Flase color composite image
        
        if PLOT_IMAGES == True:
            # initialize the figure
            fig,ax = plt.subplots(figsize = (16,16))
            
            # Define lats and longs for the figure
            lons_fig = [np.min(lon_2D), np.max(lon_2D)]
            lats_fig = [np.min(lat_2D), np.max(lat_2D)]
            lat_0 = np.mean(lats_fig)
            lon_0 = np.mean(lons_fig)
            
            # initialize the map
            map_kwargs = dict(projection='merc', resolution='l',
                              llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                              llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                              lat_0=lat_0, lon_0=lon_0)
            m = Basemap(**map_kwargs)
            
            # drawing parallels and meridians
            m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            
            
            # drawing the PRISMA image
            mesh_rgb = FalseColorComp[:, 1:, :]/np.max(FalseColorComp)  # also RGBComp[:, 1:, :] works fine
            colorTuple = mesh_rgb.reshape((mesh_rgb.shape[0] * mesh_rgb.shape[1]), 3)
            alpha = 1.0
            colorTuple = np.insert(colorTuple,3,alpha,axis=1)
            
            
            ## What you put in for the image doesn't matter because of the color mapping
            m.pcolormesh(lon_2D, lat_2D, FalseColorComp[:,:,0], latlon=True, color=colorTuple)
            #m.pcolormesh(lon_2D, lat_2D, FalseColorComp[:,:,0], latlon=True, cmap='Reds')
            
            plt.colorbar()
            
            
            # Labels on axes
            #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
            #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
            
            
            
            # Showing image
            # plt.savefig( os.getcwd() + os.path.sep + f'false_color_image_{name_fig}.png',dpi=300)
            plt.show()
        
        
        #%% False color composite images - zoom
        
        if PLOT_IMAGES == True:
            
            # FIGURE 1
            
            # initialize the figure
            fig,ax = plt.subplots(figsize = (16,16))
            
            # Define lats and longs for the figure
            lons_fig = [-121.238, -121.058]
            lats_fig = [42.724,42.85]
            lat_0 = np.mean(lats_fig)
            lon_0 = np.mean(lons_fig)
            
            # initialize the map
            map_kwargs = dict(projection='merc', resolution='l',
                              llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                              llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                              lat_0=lat_0, lon_0=lon_0)
            m = Basemap(**map_kwargs)
            
            # drawing parallels and meridians
            m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.05*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                            color=[0.6,0.6,0.6], fontsize=15, rotation=30, fmt='%10.7g')
            m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.05*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                            color=[0.6,0.6,0.6], fontsize=15, rotation=30, fmt='%10.7g')
            
            
            # drawing the PRISMA image
            mesh_rgb = FalseColorComp[:, :-1, :]/np.max(FalseColorComp)  # also A[:, 1:, :] works fine
            colorTuple = mesh_rgb.reshape((mesh_rgb.shape[0] * mesh_rgb.shape[1]), 3)
            alpha = 1.0
            colorTuple = np.insert(colorTuple,3,alpha,axis=1)
            
            ## What you put in for the image doesn't matter because of the color mapping
            m.pcolormesh(lon_2D, lat_2D, FalseColorComp[:,:,0], latlon=True, color=colorTuple)
            
            # Labels on axes
            #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
            #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
            
            # Showing image
            # plt.savefig( os.getcwd() + os.path.sep + f'false_color1_image_{name_fig}.png',dpi=300)
            plt.show()
            
            
            # FIGURE 2
            
            # initialize the figure
            fig,ax = plt.subplots(figsize = (16,16))
            
            # Define lats and longs for the figure
            lons_fig = [-121.04, -120.85]
            lats_fig = [42.724,42.79]
            lat_0 = np.mean(lats_fig)
            lon_0 = np.mean(lons_fig)
            
            # initialize the map
            map_kwargs = dict(projection='merc', resolution='l',
                              llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                              llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                              lat_0=lat_0, lon_0=lon_0)
            m = Basemap(**map_kwargs)
            
            # drawing parallels and meridians
            m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            
            
            # drawing the PRISMA image
            mesh_rgb = FalseColorComp[:, :-1, :]/np.max(FalseColorComp)  # also A[:, 1:, :] works fine
            colorTuple = mesh_rgb.reshape((mesh_rgb.shape[0] * mesh_rgb.shape[1]), 3)
            alpha = 1.0
            colorTuple = np.insert(colorTuple,3,alpha,axis=1)
            
            ## What you put in for the image doesn't matter because of the color mapping
            m.pcolormesh(lon_2D, lat_2D, FalseColorComp[:,:,0], latlon=True, color=colorTuple)
            
            # Labels on axes
            #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
            #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
            
            # Showing image
            # plt.savefig( os.getcwd() + os.path.sep + f'false_color1_image_{name_fig}.png',dpi=300)
            plt.show()
        
        
        #%% VNIR image
        
        if PLOT_IMAGES == True:
            # initialize the figure
            fig,ax = plt.subplots(figsize = (16,16))
            
            # Define lats and longs for the figure
            eps_lon = -0.001 # additive offset in degree
            eps_lat = -0.0012 # additive offset in degree
            lons_fig = [np.min(lon_2D), np.max(lon_2D)]
            lats_fig = [np.min(lat_2D), np.max(lat_2D)]
            lat_0 = np.mean(lats_fig)
            lon_0 = np.mean(lons_fig)
            
            # initialize the map
            map_kwargs = dict(projection='merc', resolution='l',
                              llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                              llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                              lat_0=lat_0, lon_0=lon_0)
            m = Basemap(**map_kwargs)
            
            # drawing parallels and meridians
            m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            
            VNIR_band = VNIR_array_2D[:,61,:].copy().astype('float')
            VNIR_band[VNIR_band<8000] = np.nan
            VNIR_band[VNIR_band>=8000] = 40000
            
            
            ## What you put in for the image doesn't matter because of the color mapping
            m.pcolormesh(lon_2D, lat_2D, VNIR_array_2D[:,60,:], latlon=True)
            
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            cb = plt.colorbar(cax=cax)
            cb.ax.tick_params(labelsize=25)
            # Labels on axes
            #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
            #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
            
            # Showing image
            # plt.savefig( os.getcwd() + os.path.sep + f'VNIR_image_{name_fig}.png',dpi=300,bbox_inches='tight')
            plt.show()
        
        
        #%% Indices plot
        
        if PLOT_IMAGES == True:
            # initialize the figure
            fig,ax = plt.subplots(figsize = (16,16))
            
            # Define lats and longs for the figure
            lons_fig = [np.min(lon_2D), np.max(lon_2D)]
            lats_fig = [np.min(lat_2D), np.max(lat_2D)]
            lat_0 = np.mean(lats_fig)
            lon_0 = np.mean(lons_fig)
            
            # initialize the map
            map_kwargs = dict(projection='merc', resolution='l',
                              llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                              llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                              lat_0=lat_0, lon_0=lon_0)
            m = Basemap(**map_kwargs)
            
            # drawing parallels and meridians
            m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            
            K_ratio = np.divide(IMG_770.astype('float'), IMG_780.astype('float'),where=IMG_780.astype('float')>0) 
            K_ratio[K_ratio>1.17] = np.nan
            K_ratio[K_ratio<0.9] = np.nan
            
            K_difference = IMG_770.astype('float') - IMG_780.astype('float') #/IMG_770.astype('float')
            K_difference[K_difference>6000] = 0
            
            HFDI = np.divide(IMG_2430.astype('float')-IMG_2060.astype('float'), IMG_2430.astype('float')+IMG_2060.astype('float'),where=IMG_2430.astype('float')+IMG_2060.astype('float')>0) 
            #HFDI[HFDI>-0.3] = np.nan
            #HFDI[HFDI<-0.7] = np.nan
            #HFDI[HFDI>-0.32] = np.nan  # con level 2B!
            HFDI[HFDI<-0.3] = np.nan  # con level 2B!
            
            # CIBR = np.divide(IMG_2010.astype('float'), 0.666*IMG_1990.astype('float')+0.334*IMG_2040.astype('float'),where=0.666*IMG_1990.astype('float')+0.334*IMG_2040.astype('float')>0) 
            # CIBR[CIBR>1.3] = np.nan
            # CIBR[CIBR<0.6] = np.nan
            
            # NBR = np.divide(IMG_1088.astype('float') - IMG_2200.astype('float'), IMG_1088.astype('float') + IMG_2200.astype('float'),where=IMG_1088.astype('float') + IMG_2200.astype('float')>0) 
            #https://www.indexdatabase.de/db/i-single.php?id=53 #about NBR
            #https://www.indexdatabase.de/db/s-single.php?id=28 #about AVIRIS
            #https://www.indexdatabase.de/db/s-single.php?id=36 #about HYPERION
                
            to_plt = HFDI
            
            #to_plt = IMG_2010
            m.pcolormesh(lon_2D, lat_2D, to_plt, latlon=True)
            
            ## What you put in for the image doesn't matter because of the color mapping
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            cb = plt.colorbar(cax=cax)
            cb.ax.tick_params(labelsize=25)
            # Labels on axes
            #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
            #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
            
            # Showing image
            # plt.savefig( os.getcwd() + os.path.sep + f'CIBR_image_{name_fig}.png',dpi=300)
            plt.show()
        
        #%% Free memory
        
        del SWIR_array_2D, VNIR_array_2D
    
    
    #%% *********************************************************************** %%#
    #                            Working with level 1
    # *************************************************************************** #
    # 
    # The level 1 image is needed to propperly evaluate the HFDI index according to 
    # the literature, where only TOA signal is used, not BOA.
    
    if use_L1_level == True:
        raster_path_1 =  os.getcwd()  + os.path.sep + Level_1 #f'D:\\Documenti_Lavoro\\PRISMA\\fire\\{Level_1}'
    
        print('Loading PRISMA L1 image...')
        f = h5py.File(raster_path_1,'r')
    
        f.visit(print)
        print(f.attrs.keys())
        
        
        bands_SWIR_1 = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Data Fields']['SWIR_Cube']/f.attrs['ScaleFactor_Swir'] - f.attrs['Offset_Swir']
        SWIR_array_1_micrometri = bands_SWIR_1/1000
    
        bands_VNIR_1 = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Data Fields']['VNIR_Cube']/f.attrs['ScaleFactor_Vnir'] - f.attrs['Offset_Vnir']
        VNIR_array_1_micrometri = bands_VNIR_1/1000
    
        wavelength_SWIR_L1 = f.attrs.get('List_Cw_Swir')
        wavelength_VNIR_L1 = f.attrs.get('List_Cw_Vnir')
        
    
        lat_L1 = f['HDFEOS/SWATHS/PRS_L1_HRC/Geolocation Fields/Latitude_SWIR'][:]
        lon_L1 = f['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_SWIR'][:]
    
        lat_VNIR_L1 = f['HDFEOS/SWATHS/PRS_L1_HRC/Geolocation Fields/Latitude_VNIR'][:]
        lon_VNIR_L1 = f['HDFEOS/SWATHS/PRS_L1_HRC/Geolocation Fields/Longitude_VNIR'][:]
        
        if sum(sum(lat_L1 - lat_VNIR_L1)):
            raise ValueError('Latitudini di SWIR e VNIR differenti!')
            
        if sum(sum(lon_L1 - lon_VNIR_L1)):    
            raise ValueError('Logitudini di SWIR e VNIR differenti!')
            
        IMG_L1_2060 = search_band(SWIR_array_1_micrometri,wavelength_SWIR_L1,2060)
        IMG_L1_2430 = search_band(SWIR_array_1_micrometri,wavelength_SWIR_L1,2430)  
    
        HFDI_L1 = np.divide(IMG_L1_2430.astype('float')-IMG_L1_2060.astype('float'), IMG_L1_2430.astype('float')+IMG_L1_2060.astype('float'),where=IMG_L1_2430.astype('float')+IMG_L1_2060.astype('float')>0) 
        
        HFDI_L1_plot = HFDI_L1.copy()
        HFDI_L1_plot[HFDI_L1_plot < HFDI_thr] = np.nan  # 
            
        
        if PLOT_IMAGES == True:
            
            # initialize the figure
            fig,ax = plt.subplots(figsize = (16,16))
            
            # Define lats and longs for the figure
            lons_fig = [np.min(lon_L1), np.max(lon_L1)]
            lats_fig = [np.min(lat_L1), np.max(lat_L1)]
            lat_0 = np.mean(lats_fig)
            lon_0 = np.mean(lons_fig)
            
            # initialize the map
            map_kwargs = dict(projection='merc', resolution='l',
                              llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                              llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                              lat_0=lat_0, lon_0=lon_0)
            m = Basemap(**map_kwargs)
            
            # drawing parallels and meridians
            m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            
            m.pcolormesh(lon_L1, lat_L1, HFDI_L1_plot, latlon=True)
            
            ## What you put in for the image doesn't matter because of the color mapping
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            cb = plt.colorbar(cax=cax)
            cb.ax.tick_params(labelsize=25)
            # Labels on axes
            #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
            #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
            
            # Showing image
            # plt.savefig( os.getcwd() + os.path.sep + f'CIBR_image_{name_fig}.png',dpi=300)
            plt.show()
        
        
        fire_lons = lon_L1[HFDI_L1 == HFDI_L1]
        fire_lats = lat_L1[HFDI_L1 == HFDI_L1]
        
        fire_coords = np.array([fire_lats,fire_lons])
        
        
        IMG_red_level1 = search_band(VNIR_array_1_micrometri,wavelength_VNIR_L1,632) #660
        IMG_green_level1 = search_band(VNIR_array_1_micrometri,wavelength_VNIR_L1,530) #660
        IMG_blue_level1 = search_band(VNIR_array_1_micrometri,wavelength_VNIR_L1,463) #660
            
        red = IMG_red_level1 # 36=632.13165 nm, 35=641.33325 nm
        green = IMG_green_level1 # 48=530.66705 nm, 45=554.5646 nm
        blue = IMG_blue_level1 # 463.731 nm
        IM = np.array([red,green,blue])#.reshape(1181,1203,3)
        w,h = IM.shape[1],IM.shape[2]
        print('Dimensions of the image:', w,h)
        IM = IM.reshape((3,w,h))
        print('Dimensions of the pre-processed RGB image:',IM.shape)
        
        t=(w,h,3)
        RGB_level1=np.zeros(t)
        for i in range(w):
            for j in range(h):
                RGB_level1[i,j]=[red[i,j],green[i,j],blue[i,j]]
        print('Dimensions of the post-processed RGB image:',RGB_level1.shape)    
        
        if PLOT_IMAGES == True:    
            # initialize the figure
            fig,ax = plt.subplots(figsize = (16,16))
            
            # Define lats and longs for the figure
            # lons_fig = [np.min(lon_2B), np.max(lon_2B)]
            # lats_fig = [np.min(lat_2B), np.max(lat_2B)]
            lons_fig = [np.min(lon_L1), np.max(lon_L1)]
            lats_fig = [np.min(lat_L1), np.max(lat_L1)]
            lat_0 = np.mean(lats_fig)
            lon_0 = np.mean(lons_fig)
            
            # initialize the map
            map_kwargs = dict(projection='merc', resolution='l',
                              llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                              llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                              lat_0=lat_0, lon_0=lon_0)
            m = Basemap(**map_kwargs)
            
            # drawing parallels and meridians
            m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                            color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
            
            
            # drawing the PRISMA image
            mesh_rgb = RGB_level1[:, :-1, :]/np.max(RGB_level1)  # also A[:, 1:, :] works fine
            colorTuple = mesh_rgb.reshape((mesh_rgb.shape[0] * mesh_rgb.shape[1]), 3)
            alpha = 1.0
            colorTuple = np.insert(colorTuple,3,alpha,axis=1)
            
            ## What you put in for the image doesn't matter because of the color mapping
            m.pcolormesh(lon_L1, lat_L1, RGB_level1[:,:,0], latlon=True, color=colorTuple)
            
            ## What you put in for the image doesn't matter because of the color mapping
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            
            # Labels on axes
            #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
            #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
            
            # Showing image
            #plt.savefig( os.getcwd() + os.path.sep + 'Oregon_with_fire_temperatures.png',dpi=300)
            plt.show()
        
    
        
    #%% *********************************************************************** %%#
    #                            Working with level 2B
    # *************************************************************************** #
    # 
    # The at ground spectral radiance product is needed in order to retrieve 
    # the temperature information 
    
    raster_path =  os.getcwd() + os.path.sep + Level_2B
    
    ## Open HDF file
    hdflayer = gdal.Open(raster_path, gdal.GA_ReadOnly)
    
    # Get the file metadata
    metadata_2B = hdflayer.GetMetadata()
    
    # List of the datasets within the file
    datasets_list = hdflayer.GetSubDatasets()
    
    print('List of the Datasets in the PRISMA L2B file:')
    for num,elem in enumerate(datasets_list):
        print(num, ' - ', elem[0].split('/')[-1])
    
    # Extract the SWIR bands 
    SWIR_2B = datasets_list[0][0]
          
    # Extract the NVIR bands 
    VNIR_2B = datasets_list[2][0]
          
    # Extract the hypercube LAT
    lat_2B = datasets_list[4][0]
          
    
    # Extract the hypercube LON
    lon_2B = datasets_list[5][0]
    
    CNM_SWIR = metadata_2B['List_Cw_Swir_Flags'].split(' ')
    CNM_SWIR = np.array([int(x) for x in CNM_SWIR[:-1]])
    
    CNM_VNIR = metadata_2B['List_Cw_Vnir_Flags'].split(' ')
    CNM_VNIR = np.array([int(x) for x in CNM_VNIR[:-1]])
    
    lat_2B = gdal.Open(lat_2B, gdal.GA_ReadOnly).ReadAsArray() # deleting the all-zero elements
    lon_2B = gdal.Open(lon_2B, gdal.GA_ReadOnly).ReadAsArray() # deleting the all-zero elements
    bands_SWIR_2B = gdal.Open(SWIR_2B, gdal.GA_ReadOnly).ReadAsArray()
    bands_VNIR_2B = gdal.Open(VNIR_2B, gdal.GA_ReadOnly).ReadAsArray()
    
    
    wavelength_VNIR = np.array([float(x) for x in metadata_2B['List_Cw_Vnir'].split(' ')[:-1]])[CNM_VNIR==1]
    wavelength_SWIR = np.array([float(x) for x in metadata_2B['List_Cw_Swir'].split(' ')[:-1]])
    CNM_SWIR_2 = (wavelength_SWIR > max(wavelength_VNIR)).astype('int')
    
    if min(CNM_SWIR - CNM_SWIR_2) == 0:
        print(f'{sum(CNM_SWIR - CNM_SWIR_2)} more SWIR bands have been removed as overlapping with the VNIR channels')
        CNM_SWIR = CNM_SWIR_2
    else:
        raise ValueError('Check the overlapping of SWOR and VNIR bands!')
        
    wavelength_SWIR = wavelength_SWIR[CNM_SWIR==1]    
        
    wavelenghts = np.append(wavelength_SWIR,wavelength_VNIR)
    sort_array = np.argsort(wavelenghts)
    wavelenghts = wavelenghts[sort_array]
    
    VNIR_array_2B_micrometri = float(metadata_2B['L2ScaleVnirMin']) + (float(metadata_2B['L2ScaleVnirMax']) - float(metadata_2B['L2ScaleVnirMin']))*(bands_VNIR_2B[:,CNM_VNIR==1,:]/((2**16)-1))  # deleting the all-zero elements
    SWIR_array_2B_micrometri = float(metadata_2B['L2ScaleSwirMin']) + (float(metadata_2B['L2ScaleSwirMax']) - float(metadata_2B['L2ScaleSwirMin']))*(bands_SWIR_2B[:,CNM_SWIR==1,:]/((2**16)-1)) # deleting the all-zero elements
    
    
    VNIR_array_2B = VNIR_array_2B_micrometri/1000 # to put the unit [W/(m2*sr*nm)]
    SWIR_array_2B = SWIR_array_2B_micrometri/1000 # to put the unit [W/(m2*sr*nm)]
    
    # Array containing both SWIR and VNIR
    DATA_CUBE_2B = np.concatenate((SWIR_array_2B, VNIR_array_2B),axis=1)
    DATA_CUBE_2B = DATA_CUBE_2B[:,sort_array,:]
    
    
    #%% Genereting RGB level2B composite
    
    
    if len(wavelength_VNIR) != VNIR_array_2B.shape[1]:
        raise ValueError('Dimensions do not match!')
    else:
        IMG_red_levelB = search_band(VNIR_array_2B,wavelength_VNIR,632) #660
        IMG_green_levelB = search_band(VNIR_array_2B,wavelength_VNIR,530) #660
        IMG_blue_levelB = search_band(VNIR_array_2B,wavelength_VNIR,463) #660
        
    print('Dimensions of the VNIR datacube:',VNIR_array_2B.shape)
    red = IMG_red_levelB # 36=632.13165 nm, 35=641.33325 nm
    green = IMG_green_levelB # 48=530.66705 nm, 45=554.5646 nm
    blue = IMG_blue_levelB # 463.731 nm
    IM = np.array([red,green,blue])#.reshape(1181,1203,3)
    w,h = IM.shape[1],IM.shape[2]
    print('Dimensions of the image:', w,h)
    IM = IM.reshape((3,w,h))
    print('Dimensions of the pre-processed RGB image:',IM.shape)
    
    t=(w,h,3)
    RGB_levelB=np.zeros(t)
    for i in range(w):
        for j in range(h):
            RGB_levelB[i,j]=[red[i,j],green[i,j],blue[i,j]]
    print('Dimensions of the post-processed RGB image:',RGB_levelB.shape)    
    
    #%% Retrieving the shapefile of interesting fires (from Stefania)
    
    fire_shp = gpd.read_file('shapefile_oregon_fires' + os.path.sep + 'ECOSTRESS17-7-2021-ROI-soglie-Lee.shp')
    fire_shp = gpd.GeoSeries(fire_shp.geometry)
    shape_extent = fire_shp.total_bounds
    print('Ecostress results, from Stefania:', shape_extent)
    
    #%% Free memory to optimize further computation
    IMGB_2060 = search_band(SWIR_array_2B_micrometri,wavelength_SWIR,2060)
    IMGB_2430 = search_band(SWIR_array_2B_micrometri,wavelength_SWIR,2430)  
     
    del SWIR_array_2B, SWIR_array_2B_micrometri, VNIR_array_2B, VNIR_array_2B_micrometri
    del bands_SWIR_2B, bands_VNIR_2B, 
    
    if use_L2D_level == True:
        del bands_SWIR_2D, bands_VNIR_2D
    
    
    #%% RGB image - Level 2B
    
    if PLOT_IMAGES == True:
        # initialize the figure
        fig,ax = plt.subplots(figsize = (16,16))
        
        # Define lats and longs for the figure
        lons_fig = [np.min(lon_2B), np.max(lon_2B)]
        lats_fig = [np.min(lat_2B), np.max(lat_2B)]
        #lons_fig = [shape_extent[0], shape_extent[2]]
        #lats_fig = [shape_extent[1], shape_extent[3]]
        
        lat_0 = np.mean(lats_fig)
        lon_0 = np.mean(lons_fig)
        
        # initialize the map
        map_kwargs = dict(projection='merc', resolution='l',
                          llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                          llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                          lat_0=lat_0, lon_0=lon_0)
        m = Basemap(**map_kwargs)
        
        # drawing parallels and meridians
        m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                        color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
        m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                        color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
        
        
        # drawing the PRISMA image
        mesh_rgb = RGB_levelB[:, :-1, :]/np.max(RGB_levelB)  # also A[:, 1:, :] works fine
        colorTuple = mesh_rgb.reshape((mesh_rgb.shape[0] * mesh_rgb.shape[1]), 3)
        alpha = 1.0
        colorTuple = np.insert(colorTuple,3,alpha,axis=1)
        
        ## What you put in for the image doesn't matter because of the color mapping
        m.pcolormesh(lon_2B, lat_2B, RGB_levelB[:,:,0], latlon=True, color=colorTuple)
        
        # m.readshapefile('shapefile_oregon_fires' + os.path.sep + 'ECOSTRESS17-7-2021-ROI-soglie-Lee', 'shapes', drawbounds = False)
        
        # for info, shape in zip(m.shapes_info, m.shapes):
        #     x, y = zip(*shape) 
        #     m.plot(x, y, marker=None,color='m')    
        
        
        
        
        ## What you put in for the image doesn't matter because of the color mapping
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        
        # Labels on axes
        #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
        #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
        
        # Showing image
        #plt.savefig( os.getcwd() + os.path.sep + 'Oregon_with_fire_temperatures.png',dpi=300)
        plt.show()
        
    
    
    #%% Creating the Geotiff image
    lat_2B_tiff = np.fliplr(lat_2B.T)
    lon_2B_tiff = np.fliplr(lon_2B.T)
        
    if save_results == True:
        RGB_cube = np.array([np.fliplr(red.T),np.fliplr(green.T),np.fliplr(blue.T)])*(2**16-1)
        geoTIFF_creation(1000, 1000, 3, RGB_cube, lat_2B_tiff, 
                                 lon_2B_tiff, 'uint32', 
                                 'results_temperature' + os.path.sep + f'L2B_{ROI_img}_{num_img}.tif')
    
    
    #%% HFDI image at level 2B
    
       
    if PLOT_IMAGES == True:
        
        # initialize the figure
        fig,ax = plt.subplots(figsize = (16,16))
        
        # Define lats and longs for the figure
        lons_fig = [np.min(lon_2B), np.max(lon_2B)]
        lats_fig = [np.min(lat_2B), np.max(lat_2B)]
        lat_0 = np.mean(lats_fig)
        lon_0 = np.mean(lons_fig)
        
        # initialize the map
        map_kwargs = dict(projection='merc', resolution='l',
                          llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                          llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                          lat_0=lat_0, lon_0=lon_0)
        m = Basemap(**map_kwargs)
        
        # drawing parallels and meridians
        m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                        color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
        m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                        color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
        
        
        HFDI_L2B = np.divide(IMGB_2430.astype('float')-IMGB_2060.astype('float'), IMGB_2430.astype('float')+IMGB_2060.astype('float'),where=IMGB_2430.astype('float')+IMGB_2060.astype('float')>0) 
        #HFDI[HFDI>-0.3] = np.nan
        #HFDI[HFDI<-0.7] = np.nan
        #HFDI[HFDI>-0.32] = np.nan  # con level 2B!
        #HFDI_L2B[HFDI_L2B<-0.56] = np.nan  # con level 2B!
        HFDI_L2B[HFDI_L2B<-0.119] = np.nan  # con level 2B!
        
        #to_plt = IMG_2010
        m.pcolormesh(lon_2B, lat_2B, HFDI_L2B, latlon=True)
        
        ## What you put in for the image doesn't matter because of the color mapping
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        cb = plt.colorbar(cax=cax)
        cb.ax.tick_params(labelsize=25)
        # Labels on axes
        #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
        #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
        
        # Showing image
        # plt.savefig( os.getcwd() + os.path.sep + f'CIBR_image_{name_fig}.png',dpi=300)
        plt.show()
        
    
    
    
    #%% I am searching for the saturated points in the L2B product. To do so, I search the nearest 
    #   points to the previous 2D-based points
    # 
    # The saturation at level 2B is not easy to be recognized, as it depends on the 
    # calibration process which varies from target to target. On the contrary, the 
    # saturation level for the 2D product is always equal to 1 (or 2^16 - 1 at bit level) 
    
    if prescribed_AOI == False:
        # I will be using the areas around the saturated pixels
        
        if fire_points_from_HFDI == False:
            
            Feasible_Fire_points_2B = []
            coords_2B = np.array([lat_2B,lon_2B])
            
            for coord in fire_coords.T:
                eps_lat = np.abs(coords_2B[0,:,:] - coord[0])
                eps_lon = np.abs(coords_2B[1,:,:] - coord[1])
                sum_eps = eps_lat + eps_lon
                pos_min = np.where(sum_eps == np.min(sum_eps))
                Feasible_Fire_points_2B.append([pos_min[0][0],pos_min[1][0]])
                
            Feasible_Fire_points_2B = np.array(Feasible_Fire_points_2B)
            # The previous algorithm can create some duplicates!
            Feasible_Fire_points_2B = np.unique(Feasible_Fire_points_2B, axis=0)
            
        else:
            
            xx,yy = np.where(HFDI_L1_plot == HFDI_L1_plot)
            Feasible_Fire_points_2B = np.column_stack((xx,yy))
            
        fire_points = np.squeeze(DATA_CUBE_2B[Feasible_Fire_points_2B[:,0],:, Feasible_Fire_points_2B[:,1]])
        
        if enlarge_AOI_L2B == True:
            img = np.zeros([1000, 1000], dtype=np.uint8)
            for i in range(len(Feasible_Fire_points_2B)):
                x,y = Feasible_Fire_points_2B[i]
                img[x,y] = 1
            # finding the clusters of feasible fire points
            num_labels, labels_im = cv2.connectedComponents(img)
            
            # surrounding each cluster with a rectangle with n pixel margin
            margin = 3
            labels_im2 = labels_im.copy()
            for i in range(1,num_labels):
                aa = np.where(labels_im2==i)
                if len(aa[0]) > 0:
                    x_min,y_min,x_max,y_max = np.min(aa[0]),np.min(aa[1]),np.max(aa[0]),np.max(aa[1])
                    labels_im2[x_min-margin:x_max+margin+1, y_min-margin:y_max+margin+1] = 1
        
            Feasible_Fire_points_2B = np.where(labels_im2==1)
            Feasible_Fire_points_2B = np.array(Feasible_Fire_points_2B).T
            # def imshow_components(labels):
            #     # Map component labels to hue val
            #     label_hue = np.uint8(179*labels/np.max(labels))
            #     blank_ch = 255*np.ones_like(label_hue)
            #     labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            
            #     # cvt to BGR for display
            #     labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            
            #     # set bg label to black
            #     labeled_img[label_hue==0] = 0
            
            #     cv2.imshow('labeled.png', labeled_img)
            #     cv2.waitKey()
            
            # imshow_components(labels_im)
        
            # reconstructing the 2D feasible fire points for comparison
            fire_lons_2B = lon_2B[Feasible_Fire_points_2B[:,0],Feasible_Fire_points_2B[:,1]]
            fire_lats_2B = lat_2B[Feasible_Fire_points_2B[:,0],Feasible_Fire_points_2B[:,1]]
            fire_coords_2B = np.array([fire_lats_2B,fire_lons_2B])
            fire_points = np.squeeze(DATA_CUBE_2B[Feasible_Fire_points_2B[:,0],:, Feasible_Fire_points_2B[:,1]])
        
            print('Retrieving L2D points corresponding to L2B feasible fire points.\n')
            if config['only_feasible_SWIR'] == True:
                Feasible_Fire_points_2D = []
                coords_2D = np.array([lat_2D,lon_2D])
                for coord in tqdm(fire_coords_2B.T):
                    eps_lat = np.abs(coords_2D[0,:,:] - coord[0])
                    eps_lon = np.abs(coords_2D[1,:,:] - coord[1])
                    sum_eps = eps_lat + eps_lon
                    pos_min = np.where(sum_eps == np.min(sum_eps))
                    Feasible_Fire_points_2D.append([pos_min[0][0],pos_min[1][0]])
                    
                Feasible_Fire_points_2D = np.array(Feasible_Fire_points_2D)
                fire_points_2D = np.squeeze(DATA_CUBE[Feasible_Fire_points_2D[:,0],:, Feasible_Fire_points_2D[:,1]])
            else:
                fire_points_2D = np.empty(fire_points.shape)
                Feasible_Fire_points_2D = np.empty(Feasible_Fire_points_2B.shape)
        else:
            if config['only_feasible_SWIR'] == True:
                fire_points_2D = np.squeeze(DATA_CUBE[Feasible_Fire_points_2D[:,0],:, Feasible_Fire_points_2D[:,1]])
            else:
                fire_points_2D = np.empty(fire_points.shape)
                Feasible_Fire_points_2D = np.empty(Feasible_Fire_points_2B.shape)    
    else:
        
        Feasible_Fire_points_2B = []
        
        pp0 = extract_points_in_AOI(aoi[0],lon_2B,lat_2B)
        pp1 = extract_points_in_AOI(aoi[1],lon_2B,lat_2B)
        pp2 = extract_points_in_AOI(aoi[2],lon_2B,lat_2B)
        pp3 = extract_points_in_AOI(aoi[3],lon_2B,lat_2B)
        
        Feasible_Fire_points_2B = pp0 + pp1 + pp2 + pp3
        
        # x,y = aoi[0].exterior.xy
        # plt.plot(x,y)
        # x,y = aoi[1].exterior.xy
        # plt.plot(x,y)
        # x,y = aoi[2].exterior.xy
        # plt.plot(x,y)
        # x,y = 500,0
        # x,y = Point(lon_2B[x,y],lat_2B[x,y]).xy
        # plt.scatter(x,y)
        
        Feasible_Fire_points_2B = np.array(Feasible_Fire_points_2B, dtype=np.int64)
        
        fire_lons_2B = lon_2B[Feasible_Fire_points_2B[:,0],Feasible_Fire_points_2B[:,1]]
        fire_lats_2B = lat_2B[Feasible_Fire_points_2B[:,0],Feasible_Fire_points_2B[:,1]]
        fire_coords_2B = np.array([fire_lats_2B,fire_lons_2B]).T
        fire_points = np.squeeze(DATA_CUBE_2B[Feasible_Fire_points_2B[:,0],:, Feasible_Fire_points_2B[:,1]])
        
        
        if config['only_feasible_SWIR'] == True:
            print('Retrieving L2D points corresponding to L2B feasible fire points.\n')
            coords_2D = np.array([lat_2D,lon_2D])
            results = Parallel(n_jobs=12)(
                delayed(eval_2D_points)(i, fire_coords_2B[i,:], coords_2D)   
                for i in tqdm(range(len(fire_coords_2B)))
                )
            
            Feasible_Fire_points_2D = np.array(results) 
            Feasible_Fire_points_2D = Feasible_Fire_points_2D[np.argsort(Feasible_Fire_points_2D[:, 0])]
            Feasible_Fire_points_2D = Feasible_Fire_points_2D[:,1:]
            fire_points_2D = np.squeeze(DATA_CUBE[Feasible_Fire_points_2D[:,0],:, Feasible_Fire_points_2D[:,1]])
    
        else:
            fire_points_2D = np.empty(fire_points.shape)
            Feasible_Fire_points_2D = np.empty(Feasible_Fire_points_2B.shape)
        
    #%% Defining background points starting from HFDI L1 mask
    
    HFDI_L1_mask = HFDI_L1.copy()
    HFDI_L1_mask[HFDI_L1_mask > HFDI_thr] = np.nan   
    
    HFDI_indices = np.where(HFDI_L1_mask != HFDI_L1_mask)
    
    # considering a margin of N_margin pixels from the original mask
    N_margin = 5
    for i in range(-N_margin,N_margin+1,1):
        for j in range(-N_margin,N_margin+1,1):
            x_index = HFDI_indices[0] + i
            y_index = HFDI_indices[1] + j
            
            x_index[x_index > HFDI_L1_mask.shape[0] - 1] = HFDI_L1_mask.shape[0] - 1
            y_index[y_index > HFDI_L1_mask.shape[1] - 1] = HFDI_L1_mask.shape[1] - 1
            
            x_index[x_index < 0] = 0
            y_index[y_index < 0] = 0
            
            HFDI_L1_mask[x_index,y_index] = np.nan
        
    bkg_points_HFDI = np.array([[i,j] for i in range(1000) for j in range(1000) if HFDI_L1_mask[i,j]==HFDI_L1_mask[i,j]])
    
    if use_saved_bkg == False:
        bkg_spectra = np.squeeze(DATA_CUBE_2B[bkg_points_HFDI[:,0],:, bkg_points_HFDI[:,1]])
        with open('results_temperature' + os.path.sep + f'bkg_spectra_{ROI_img}_{num_img}.npy', 'wb') as f:
            np.save(f, bkg_spectra)   
    else:
        with open('results_temperature' + os.path.sep + f'bkg_spectra_{ROI_img}_Im2.npy', 'rb') as f:
            a = np.load(f)
        with open('results_temperature' + os.path.sep + f'bkg_spectra_{ROI_img}_Im3.npy', 'rb') as f:
            b = np.load(f)
            
        bkg_spectra = np.append(a, b, axis=0)  
        del a, b
        np.random.shuffle(bkg_spectra)
        DATA_CUBE_2B = []
        
    #%% Background points for the fire pixels
    # 
    # Plotting the mean fire spectral curve with confidence interval (1 std)
    # 
    
    
    if PLOT_IMAGES == True:
        means_to_plt = np.mean(fire_points, axis=0)
        means_to_plt = fire_points[0,:]
        stds_to_plt = np.std(fire_points, axis=0)
        
        plt.style.use('ggplot')
        plt.plot(wavelenghts,means_to_plt)
        plt.fill_between(wavelenghts,means_to_plt-stds_to_plt, means_to_plt+stds_to_plt, alpha=.3)
        plt.ylabel('At-surface Radiance') #($W m-2 sr-1 nm-1$)
        plt.xlabel('Wavelength (nm)')
        # plt.savefig( os.getcwd() + os.path.sep + 'fire_pixels.png',dpi=300)
        plt.show()
        
        
        # initialize the figure
        fig,ax = plt.subplots(figsize = (16,16))
        
        # Define lats and longs for the figure
        lons_fig = [np.min(lon_2B), np.max(lon_2B)]
        lats_fig = [np.min(lat_2B), np.max(lat_2B)]
        lat_0 = np.mean(lats_fig)
        lon_0 = np.mean(lons_fig)
        
        # initialize the map
        map_kwargs = dict(projection='merc', resolution='l',
                          llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                          llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                          lat_0=lat_0, lon_0=lon_0)
        m = Basemap(**map_kwargs)
        
        # drawing parallels and meridians
        m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                        color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
        m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                        color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
        
        
        HFDI_L2B = np.divide(IMGB_2430.astype('float')-IMGB_2060.astype('float'), IMGB_2430.astype('float')+IMGB_2060.astype('float'),where=IMGB_2430.astype('float')+IMGB_2060.astype('float')>0) 
        
        # to plot feasible bkg pixels
        #HFDI_L2B[HFDI_L2B<-0.56] = np.nan  # con level 2B!
        #HFDI_L2B[HFDI_L2B_mask != HFDI_L2B_mask] = np.nan
        # to save the HFDI image
        HFDI_L2B[HFDI_L2B<-0.68] = -0.68  # con level 2B!
        HFDI_L2B[HFDI_L2B> 0.9] = np.nan  # con level 2B!
       
        #to_plt = IMG_2010
        m.pcolormesh(lon_2B, lat_2B, HFDI_L2B, latlon=True)
        
        lat_2B_tiff = np.fliplr(lat_2B.T)
        lon_2B_tiff = np.fliplr(lon_2B.T)
        HFDI_L2B_tiff = np.reshape(np.fliplr(HFDI_L2B.T),(1,1000,1000))
        geoTIFF_creation(1000, 1000, 1, HFDI_L2B_tiff, lat_2B_tiff, lon_2B_tiff,
                         'float32', 'results_temperature' + os.path.sep + f'HFDI_L2B_{num_img}.tif')
        
        
        idx = np.random.randint(len(bkg_points_HFDI), size=10)
        x, y = m(lon_2B[bkg_points_HFDI[idx,0],bkg_points_HFDI[idx,1]], lat_2B[bkg_points_HFDI[idx,0],bkg_points_HFDI[idx,1]])  # transform coordinates
        m.scatter(x,y, 10, marker='o', color='Blue')
        
        # Super-imposing the scatter plot of the saturated pixels
        x, y = m(lon_2B[Feasible_Fire_points_2B[:,0],Feasible_Fire_points_2B[:,1]], lat_2B[Feasible_Fire_points_2B[:,0],Feasible_Fire_points_2B[:,1]])  # transform coordinates
        m.scatter(x,y, 10, marker='o', color='Red')
        
        m.readshapefile('shapefile_oregon_fires' + os.path.sep + 'ECOSTRESS17-7-2021-ROI-soglie-Lee', 'shapes', drawbounds = False)
        
        for _, shape in zip(m.shapes_info, m.shapes):
            x, y = zip(*shape) 
            m.plot(x, y, marker=None,color='m')    
            
        pA = m(lon_2B_tiff[500,-1], lat_2B_tiff[500,-1])
        pB = m(lon_2B_tiff[500,750], lat_2B_tiff[500,750])
        pC = m(lon_2B_tiff[0,750], lat_2B_tiff[0,750])
        pD = m(lon_2B_tiff[0,-1], lat_2B_tiff[0,-1])
        AOI = Polygon([pA, pB, pC, pD, pA])
        x,y = AOI.exterior.coords.xy
        m.plot(x, y, marker=None,color='y') 
        ## What you put in for the image doesn't matter because of the color mapping
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        cb = plt.colorbar(cax=cax)
        cb.ax.tick_params(labelsize=25)
        # Labels on axes
        #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
        #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
        
        # Showing image
        # plt.savefig( os.getcwd() + os.path.sep + f'CIBR_image_{name_fig}.png',dpi=300)
        plt.show()
    
    
    #%% Example of Plank Black body curves
    
    if PLOT_IMAGES == True:
        
        T_list = [300,350]   # [K] 2000,4000,5000,6000,7000
        
        x, y = [],[]
        
        for T in T_list:
            print(f'Computing at {T} K...')
            plank_black_body = Planck_function(np.linspace(400,10000), T)
            x.append(np.linspace(400,10000))
            y.append(plank_black_body)
            
        for i in range(len(x)):
            plt.plot(x[i],y[i])
            
        plt.show()
    
    
    #%% Temperature estimation - Dennison approach
    
    plot_2B_2D_comparison = False
    
    if plot_2B_2D_comparison == True:
        # Plotting the comparison between the signal at level 2D and the signal at level 2B for each pixel
        for index in range(len(Feasible_Fire_points_2D[:30])):
            index = 74399  #4214,5590
            print('************************************************************')
            print(f'Pixel {index}')
            # fire_pixel_2D = np.squeeze(DATA_CUBE[Feasible_Fire_points_2D[index,0],:, Feasible_Fire_points_2D[index,1]])
            # xx = wavelenghts
            # index_SWIR = xx > 1400
            # yy = fire_pixel_2D
            # fig = plt.figure()
        
            # plt.plot(xx[index_SWIR],yy[index_SWIR])
            # plt.xlabel('Wavelength ($nm$)',fontsize=15)
            # plt.ylabel('Reflectance',fontsize=15)
        
            # plt.show()
        
            fire_pixel_2B = np.squeeze(DATA_CUBE_2B[Feasible_Fire_points_2B[index,0],:, Feasible_Fire_points_2B[index,1]])
            fire_pixel_2B = np.squeeze(DATA_CUBE_2B[142,:,224])
            xx = wavelenghts
            index_SWIR = xx > 1400
            yy = fire_pixel_2B
            fig = plt.figure()
        
            plt.plot(xx[index_SWIR],yy[index_SWIR])
            plt.xlabel('Wavelength ($nm$)',fontsize=15)
            plt.ylabel('Radiance',fontsize=15)
            plt.title(f'Pixel n. {index}')
            plt.show()



            fire_pixel_2B = np.squeeze(SWIR_array_1_micrometri[863,:,29])
            plt.plot(fire_pixel_2B[::-1])
            plt.xlabel('Wavelength ($nm$)',fontsize=15)
            plt.ylabel('Radiance',fontsize=15)
            plt.title(f'Pixel n. {index}')
            plt.show()
            
    # for index in range(len(Feasible_Fire_points_2D)):
    #     fire_pixel_2B = np.squeeze(DATA_CUBE_2B[Feasible_Fire_points_2B[index,0],:, Feasible_Fire_points_2B[index,1]])
    #     if np.max(fire_pixel_2B) > 0.145:
    #         print(index)
            
    # for i in range(1000):
    #     for j in range(1000):
    #         fire_pixel_2B = np.squeeze(bands_SWIR_1[i,:, j])
    #         if np.max(fire_pixel_2B) > 250:
    #             print(i,j)
            
    #%% Running serial computation
    
    # free memory
    if use_L2D_level == True:
        del DATA_CUBE
    
    if only_plot_prev_results == False:
        
        if run_parallel == False:
            print('Starting serial computation for temperature estimation...')
            all_results = []
            for pixel_number in tqdm(range(len(Feasible_Fire_points_2B))):
                print(f'PIXEL {pixel_number + 1} of {len(Feasible_Fire_points_2B)}')
                fire_pixel = fire_points[pixel_number,:]  #6  10  13
                fire_pixel_2D = fire_points_2D[pixel_number,:]  #6  10  13
                result = main_temp_estimation(DATA_CUBE_2B, pixel_number, wavelenghts, 
                                              fire_pixel, fire_pixel_2D, bkg_points_HFDI, 
                                              config, use_saved_bkg, bkg_spectra)
                all_results.append(result)
            
            print('Serial computation for temperature estimation accomplished.')
            config['plot'] = False
            config['verbose'] = 0
            config['N_trials'] = 30 # number of different estimates to provide the final temperature
            pixel_number = 4214   #4214,5590
            fire_pixel = fire_points[pixel_number,:]  #6  10  13
            fire_pixel_2D = fire_points_2D[pixel_number,:]  #6  10  13
            t = time.time()
            result = main_temp_estimation(DATA_CUBE_2B, pixel_number, wavelenghts, 
                                          fire_pixel, fire_pixel_2D, bkg_points_HFDI, 
                                          config, use_saved_bkg, bkg_spectra)
            print(time.time() - t)
            
        
        
        #%% Stima temperatura usando approccio parallelo
        else:
            
            print('Starting parallel computation for temperature estimation...\n')
            from joblib import Parallel, delayed
            t = time.time()
            results = Parallel(n_jobs=12)(
                delayed(main_temp_estimation)(
                    DATA_CUBE_2B, pix_num, wavelenghts, fire_points[pix_num,:], 
                    fire_points_2D[pix_num,:], bkg_points_HFDI, config, 
                    use_saved_bkg, bkg_spectra) 
                for pix_num in tqdm(range(len(Feasible_Fire_points_2B[:500])))
                )  
                                         
            print(time.time() - t)
            print('Parallel computation for temperature estimation accomplished.')
    
        results_pd = pd.DataFrame.from_dict(results)
        results_pd['ind_x'] = Feasible_Fire_points_2B[:,0]
        results_pd['ind_y'] = Feasible_Fire_points_2B[:,1]
        
        # Today as dd-mm-YY
        today = date.today()
        date_today = today.strftime("%d-%m-%Y")
        if save_results == True:
            results_pd.to_csv('results_temperature' + os.path.sep + f'Results_{ROI_img}_{num_img}_{date_today}.csv', index = False)
    
    else:
        
        results_pd = pd.read_csv('results_temperature' + os.path.sep + f'Results_{ROI_img}_{num_img}_{date_results}.csv')
        
    # esempio di pixel semi-caldo e freddo, funziona senza AOI prescribed e con True 
    # su entrambi i flag enlarge
    # aa = results_pd[results_pd.mean_temp > 1900][['ind_x','ind_y']]  #5590
    # bb = results_pd[results_pd.mean_temp < 300][['ind_x','ind_y']]   #4214
    
    
    # Histogram for the distribution of the temperature estimations
    fig,ax = plt.subplots(figsize = (16,16))
    plt.hist(results_pd.mean_temp,100)
    
    
    
    # Creo il layer con la mappa delle temperature!--
    img = np.empty([1000,1000])  #,dtype=np.uint8
    img[:] = np.nan
    print('Creating image with temperature estimation')
    for i in tqdm(range(len(results_pd))):
        x,y = results_pd.iloc[i][['ind_x','ind_y']].values
        img[x,y] = results_pd.iloc[i]['mean_temp'] 
        
    if save_results == True:
        lat_2B_tiff = np.fliplr(lat_2B.T)
        lon_2B_tiff = np.fliplr(lon_2B.T)
        temp_map = np.reshape(np.fliplr(img.T),(1,1000,1000))
        geoTIFF_creation(1000, 1000, 1, temp_map, lat_2B_tiff, lon_2B_tiff,
                         'float32', 'results_temperature' + os.path.sep + f'T_map_{ROI_img}_{num_img}.tif')
    
    # Creo il layer con la mappa delle temperature pesate con T ambiente!--
    img_MWIR_LWIR = np.empty([1000,1000])  #,dtype=np.uint8
    T_ref = 300
    img_MWIR_LWIR[:] = np.nan
    print('Creating image with average with room temperature')
    for i in tqdm(range(len(results_pd))):
        x,y = results_pd.iloc[i][['ind_x','ind_y']].values
        img_MWIR_LWIR[x,y] = (results_pd.iloc[i]['mean_P_coeff']*results_pd.iloc[i]['mean_temp']**4 + \
            (1 - results_pd.iloc[i]['mean_P_coeff'])*T_ref**4)**(1/4)
    
    if save_results == True:
        lat_2B_tiff = np.fliplr(lat_2B.T)
        lon_2B_tiff = np.fliplr(lon_2B.T)
        temp_map_MWIR_LWIR = np.reshape(np.fliplr(img_MWIR_LWIR.T),(1,1000,1000))
        geoTIFF_creation(1000, 1000, 1, temp_map_MWIR_LWIR, lat_2B_tiff, lon_2B_tiff,
                         'float32', 'results_temperature' + os.path.sep + f'T_mean_map_{ROI_img}_{num_img}.tif')
    
    thrs = np.array([0,400,700,1200,2000])
    levels = np.array([0,1,2,3])
    # Creo il layer con la mappa delle temperature!--
    img_thr = np.empty([1000,1000])  #,dtype=np.uint8
    img_thr[:] = np.nan
    print('Creating image with thresholded temperature estimation')
    for i in tqdm(range(len(Feasible_Fire_points_2B))):
        x,y = Feasible_Fire_points_2B[i]
        est_temp = results_pd.iloc[i]['mean_temp'] 
        if est_temp == est_temp:
            loc = np.where(thrs < est_temp)[0][-1]
            img_thr[x,y] = levels[loc] 
        else:
            img_thr[x,y] = np.nan
    
    if save_results == True:
        lat_2B_tiff = np.fliplr(lat_2B.T)
        lon_2B_tiff = np.fliplr(lon_2B.T)
        temp_map_thr = np.reshape(np.fliplr(img_thr.T),(1,1000,1000))
        geoTIFF_creation(1000, 1000, 1, temp_map_thr, lat_2B_tiff, lon_2B_tiff,
                         'uint32', 'results_temperature' + os.path.sep + f'T_thr_{ROI_img}_{num_img}.tif')
    
    #%% Figure with the temperature estimation map
     # initialize the figure
    fig,ax = plt.subplots(figsize = (16,16))
    
    # Define lats and longs for the figure
    lons_fig = [np.min(lon_2B), np.max(lon_2B)]
    lats_fig = [np.min(lat_2B), np.max(lat_2B)]
    lat_0 = np.mean(lats_fig)
    lon_0 = np.mean(lons_fig)
    
    # initialize the map
    map_kwargs = dict(projection='merc', resolution='l',
                      llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                      llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                      lat_0=lat_0, lon_0=lon_0)
    m = Basemap(**map_kwargs)
    
    # drawing parallels and meridians
    m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                    color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
    m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                    color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
    
    m.pcolormesh(lon_2B, lat_2B, img, latlon=True)
    # xx,yy = results_pd.iloc[4214][['ind_x','ind_y']]  #4214,5590
    
    # x, y = m(lon_2B[xx,yy], lat_2B[xx,yy])  # transform coordinates
    # m.scatter(x,y, 20, marker='o', color='Red')
    
    ## What you put in for the image doesn't matter because of the color mapping
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cb = plt.colorbar(cax=cax)
    cb.ax.tick_params(labelsize=25)
    # Labels on axes
    #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
    #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
    
    # Showing image
    if save_results == True:
        fileName = 'results_temperature' + os.path.sep + f'T_map_{ROI_img}_{num_img}.png'
        if os.path.isfile(fileName):
            os.remove(fileName)
        plt.savefig(fileName,dpi=300)
    plt.show()
    plt.close()
    
    #%% Figure with the thresholded temperature estimation map
     # initialize the figure
    fig,ax = plt.subplots(figsize = (16,16))
    
    # Define lats and longs for the figure
    lons_fig = [np.min(lon_2B), np.max(lon_2B)]
    lats_fig = [np.min(lat_2B), np.max(lat_2B)]
    lat_0 = np.mean(lats_fig)
    lon_0 = np.mean(lons_fig)
    
    # initialize the map
    map_kwargs = dict(projection='merc', resolution='l',
                      llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                      llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                      lat_0=lat_0, lon_0=lon_0)
    m = Basemap(**map_kwargs)
    
    # drawing parallels and meridians
    m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                    color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
    m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                    color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
    
    cmap = plt.cm.get_cmap('viridis', 4) 
    m.pcolormesh(lon_2B, lat_2B, img_thr, cmap = cmap, latlon=True)
    
    # Creating 4 Patch instances
    plt.legend([mpatches.Patch(color=cmap(b)) for b in levels],
           ['{} - {} K'.format(thrs[i], thrs[i+1]) for i in range(len(levels))],
           loc='lower center', ncol=2, fontsize=25)
    ## What you put in for the image doesn't matter because of the color mapping
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    
    
    # Labels on axes
    #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
    #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
    
    # Showing image
    if save_results == True:
        fileName = 'results_temperature' + os.path.sep + f'T_thr_{ROI_img}_{num_img}.png'
        if os.path.isfile(fileName):
            os.remove(fileName)
        plt.savefig(fileName,dpi=300)
    plt.show()
    plt.close()
    
    #%% Figure with the room temperature average
     # initialize the figure
    fig,ax = plt.subplots(figsize = (16,16))
    
    # Define lats and longs for the figure
    lons_fig = [np.min(lon_2B), np.max(lon_2B)]
    lats_fig = [np.min(lat_2B), np.max(lat_2B)]
    lat_0 = np.mean(lats_fig)
    lon_0 = np.mean(lons_fig)
    
    # initialize the map
    map_kwargs = dict(projection='merc', resolution='l',
                      llcrnrlat=np.min(lats_fig), urcrnrlat=np.max(lats_fig),
                      llcrnrlon=np.min(lons_fig), urcrnrlon=np.max(lons_fig),
                      lat_0=lat_0, lon_0=lon_0)
    m = Basemap(**map_kwargs)
    
    # drawing parallels and meridians
    m.drawparallels(np.arange(np.min(lats_fig),np.max(lats_fig),0.2*(np.max(lats_fig)-np.min(lats_fig))),labels=[1,0,0,0],
                    color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
    m.drawmeridians(np.arange(np.min(lons_fig),np.max(lons_fig),0.2*(np.max(lons_fig)-np.min(lons_fig))),labels=[0,0,0,1],
                    color=[0.6,0.6,0.6], fontsize=30, rotation=30, fmt='%8.5g')
    
    m.pcolormesh(lon_2B, lat_2B, img_MWIR_LWIR, latlon=True)
                 
    
    ## What you put in for the image doesn't matter because of the color mapping
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cb = plt.colorbar(cax=cax)
    cb.ax.tick_params(labelsize=25)
    
    
    # Labels on axes
    #plt.xlabel('Longitude (degree)', labelpad=80, fontsize=16)
    #plt.ylabel('Latitude (degree)', labelpad=100, fontsize=16)
    
    # Showing image
    if save_results == True:
        fileName = 'results_temperature' + os.path.sep + f'T_mean_map_{ROI_img}_{num_img}.png'
        if os.path.isfile(fileName):
            os.remove(fileName)
        plt.savefig(fileName,dpi=300)
    plt.show()
    plt.close()
    
    
    # from osgeo import gdal
    
    # ds = gdal.Open('ECO2LSTE.001_SDS_LST_doy2021204022331_aid0001.tif', gdal.GA_ReadOnly)
    # rb = ds.GetRasterBand(1)
    # img_array = rb.ReadAsArray()
    # print(ds.GetProjection())
    # print(ds.GetGeoTransform())
    
    # proj = osr.SpatialReference(wkt=ds.GetProjection())
    # print(proj.GetAttrValue('AUTHORITY',1))



if __name__ == "__main__":
    main()

