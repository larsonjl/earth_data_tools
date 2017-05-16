#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:55:12 2016

Function to get altimeter data for a given bounding box. Outputs an individual
netCDF4 file for each cycle for every altimeter of interest.

Main executable is get_altimeter_data(bbox_extent, sat_list, output_dir)

bbox_extent = [lat_max, lat_min, lon_max, lon_min]
sat_list    = list of satellites to use i.e. ['sa', 'c2', 'e1', 'e2']
output_dir  = str, path to output data to

NOTES
- If you want to change the rads variables which are output, add it to the dict
in 'var_out' in save_bbox_cycle_data
- Currently set up to run parallel on 6 processors


@author: jala8901
"""
import numpy as np
from netCDF4 import Dataset
import glob
from collections import defaultdict
import multiprocessing
from subprocess import call


def find_groundtracks(satellite, bbox_extent):
    '''
    Function to take in satellite name and bounding box and returns a list
    of the passes that fall within that box

    Inputs:
    satellite = str, use 'sa' for saral/altika, 'c2' for cryosat-2, 'j1' for
    jason-1
    bbox_extent  = list, [lat_max, lat_min, lon_max, lon_min].  Lon from 0-360.

    Returns:
    track_arr = list of tracks with zero padded string of length 4 for ease in
    pulling files later
    '''
    # rads dir
    dir_rads = '/project/rads/data/'
    # which cycle to use as a reference orbit for each sat (arbitrarily chosen)
    ref_orb = {'sa': '/a/c005/', 'c2': '/a/c005/', 'j1': '/a/c005/',
               'j2': '/a/c005/', 'e1': '/a/c005',  'e2': '/a/c005',
               'tx': '/a/c130', '3a': '/a/c005'}
    track_arr = []
    nc_file_list = glob.glob(dir_rads+satellite+ref_orb[satellite]+'/*.nc')
    for files in nc_file_list:
        dfile = Dataset(files)
        lats = dfile.variables['lat'][:]
        lons = dfile.variables['lon'][:]
        lons[lons < 0] = lons[lons < 0] + 360
        latbox = (lats < bbox_extent[0]) * (lats > bbox_extent[1])
        lonbox = (lons < bbox_extent[2]) * (lons > bbox_extent[3])
        if sum(latbox * lonbox) > 0:
            track_arr.append(str(dfile.pass_number).zfill(4))
    return track_arr


def save_bbox_cycle_data(cycle_path, out_path, groundtracks, bbox_extent):
    '''
    Start at cycle level of rads directory and create a .nc
    file of all variables of interest falling within a given bounding box.
    Passes of interest are included in groundtracks array which is determined
    by find_groundtracks(satellite, bbox_extent)

    Inputs
    -----
    cycle_path   = str, path to folder containing cycle of interest
    out_path     = str, path where to write .nc file with data
    groundtracks = list of strings containing passes within bbox
    bbox_extent  = list, [lat_max, lat_min, lon_max, lon_min], lon from 0-360

    Outputs
    ------
    None
    '''
    var_out = {'ssha': np.array([]), 'time': np.array([]), 'lat': np.array([]),  
                'lon': np.array([]), 'sst': np.array([]), 'dist_coast': np.array([]),
                }

    nc_file_list = glob.glob(cycle_path+'/*.nc')
    for ncfiles in nc_file_list:
        if ncfiles[-11:-7] in groundtracks:
            dfile = Dataset(ncfiles)
            lats  = dfile.variables['lat'][:]
            lons  = dfile.variables['lon'][:]
            lons[lons<0] = lons[lons<0] + 360
            indx  = (lats < bbox_extent[0]) * (lats > bbox_extent[1]) * \
                    (lons < bbox_extent[2]) * (lons > bbox_extent[3])
            
            for measurements in var_out:
                if var_out!='mission_name':
                    var_out[measurements] = np.append(var_out[measurements], \
                    dfile.variables[measurements][indx])
                else:
                    var_out[measurements] = np.append(var_out[measurements],\
                    dfile.measurements * np.ones(len(dfile.variables['time'][indx])))
                    
                
    if 'lats' in locals():
        # Now save to new .nc file
        call(["rm", out_path+"/%s%d.nc"%(dfile.filename[0:2], dfile.cycle_number)])
        outfile = Dataset(out_path+"/%s%d.nc"%(dfile.filename[0:2], dfile.cycle_number), "w", format="NETCDF4")
        outfile.createDimension("timelen", len(var_out['time']))
        for time_series in var_out.keys():
            if time_series!='time':
                fill_val = dfile.variables[time_series]._FillValue
            else:
                fill_val = -999.99
            xx  = outfile.createVariable("%s"%time_series, "f4" ,("timelen",), fill_value = fill_val)
            xx.units      = dfile.variables[time_series].units
            xx.long_name  = dfile.variables[time_series].long_name
            xx[:] = var_out[time_series]
        outfile.satellite = dfile.mission_name
        outfile.close()
    
        print("Finished: %s %s"%(dfile.filename[0:2], cycle_path[-5::]))
    else:
        print("No data: %s"%(cycle_path))
    
def get_altimeter_data(bbox_extent, sat_list, output_dir):
    #  Which Satellites and phases to include
    phase_list = {'c2':['/a/'], 'sa':['/a/'], 'e1':['/a/'], 'e2':['/a/'],\
                  'tx':['/a/'], 'j1':['/a/'], 'j2':['/a/'], '3a':['/a/']}
    
    # Get groundtracks for each satellite that fall within bounding box for efficiency
    dir_rads     = '/project/rads/data/'
    for satellites in sat_list:
        groundtracks = defaultdict()
        groundtracks[satellites] = find_groundtracks(satellites, bbox_extent)
        
        for phases in phase_list[satellites]:
            dir_rads_sat = dir_rads + satellites + phases
            cycle_list   = glob.glob(dir_rads_sat+'c*')
            
            # parallelization
            pool = multiprocessing.Pool(processes=6)
            [pool.apply(save_bbox_cycle_data, (x+'/', output_dir, groundtracks[satellites], bbox_extent)) for x in cycle_list]
 
'''
# Example of execution
bbox = [73.35, 72.31, (360 - 54), (360 - 59)]
sat_list = ['e1', 'e2', 'sa', '3a']
saveDir = '/tmp/upernavik/'

get_altimeter_data(bbox, sat_list, saveDir)
'''
