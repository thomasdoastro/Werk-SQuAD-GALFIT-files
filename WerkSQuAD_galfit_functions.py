# These are the functions used to make input files

import numpy as np
import pandas as pd
import numpy.linalg as lin

from astropy.io import fits
from astropy.table import Table

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

#################### Function 1 ####################

def make_full_galaxy_table(path):
    # Takes the path to the file cgmsquared_full_galaxy_table.fits file
    # Returns the dataframe of the full galaxy table
    
    # Reads in the full galaxy table fits file to extract data
    fit1 = fits.open(path, memmap=True)
    table = Table(fit1[1].data)
    fit1.close()

    # Gets the column names
    column_array = []
    for column in table.columns:
        column = str(column)
        column_array.append(column)
    
    # Turns Astropy Table into a Pandas DataFrame, then sorts based on QSO field ('OBJECT') and redshift ('z')
    df = table[column_array].to_pandas()
    df = df.sort_values(["OBJECT", "z"], ascending=True).reset_index(drop=True)
    
    # Returns a Pandas DataFrame with all of the objects in cgmsquared_full_galaxy_table.fits ordered by QSO sightline and then by redshift
    return df

#################### Function 2 ####################

def select_near_qso(full_galaxy_df, qso_name, max_ang_dist, max_rho_impact, max_z, include_SF=True, include_E=False, include_SFE=False):
    # full_galaxy_df should be the output of make_full_galaxy_table
    # qso_name is a string
    # max_ang_dist is in degrees
    # Default is to only include SF galaxies and exclude E and SF+E galaxies
    
    # Selects all entries in full_galaxy_df which belong to the desired QSO field
    df_qso = full_galaxy_df[full_galaxy_df['OBJECT'] == qso_name]
    
    # Of the galaxies in the desired QSO field, sorts them based on gal_type (SF, E, SF+E)
    df_qso_sf = df_qso[df_qso['gal_type']=='SF']
    df_qso_sfe = df_qso[df_qso['gal_type']=='SF+E']
    df_qso_e = df_qso[df_qso['gal_type']=='E']

    # Creates a Pandas DataFrame, adds on the E and SF+E galaxies if desired
    df_qso = df_qso_sf
    if include_E==True:
        df_qso = df_qso.append(df_qso_e, ignore_index=False)
    if include_SFE==True:
        df_qso = df_qso.append(df_qso_sfe, ignore_index=False)

    # Sorts the resulting Pandas DataFrame
    df_qso = df_qso.sort_values(["OBJECT", "z"], ascending=True)

    # Extracts RA and DEC for the QSO and each galaxy of the desired type
    qso_ra = df_qso['RA_QSO']
    qso_dec = df_qso['DEC_QSO']

    gal_ra = df_qso['RA']
    gal_dec = df_qso['DEC']

    # Finds the angular distance between the QSO sightline and each galaxy, then assigns these values to each galaxy in the Pandas DataFrame
    ra_diff = np.abs(qso_ra - gal_ra)
    dec_diff = np.abs(qso_dec - gal_dec)
    ang_diff = np.sqrt((ra_diff**2) + (dec_diff**2))
    df_qso['angular_distance'] = ang_diff
    
    # Filters out galaxies with ang_diff > max_ang_dist
    df_nearqso = df_qso.query('angular_distance < ' + str(max_ang_dist))
    
    # Filters out galaxies with rho_impact > max_rho_impact
    df_nearqso = df_nearqso.query('rho_impact < ' + str(max_rho_impact))
    
    # Filters out galaxies with z > max_z
    df_nearqso = df_nearqso.query('z < ' + str(max_z))
    
    # Returns a Pandas DataFrame with galaxies of the desired type within max_ang_dist of the QSO sightline
    return df_nearqso

#################### Function 3 ####################

def get_pix_coords(path, df_nearqso, x_offset, y_offset):
    # Takes path to unflattened image file
    # df_nearqso is the output from select_near_qso
    # region takes a list in the format [xmin, xmax, ymin, ymax], coords for galaxies in this box will be given
    
    # Define the WCS object
    hdulist = fits.open(path, memmap=True)
    hdu = hdulist[1]
    wcs = WCS(hdu.header)
    hdulist.close()

    # Initialize empty arrays for the x and y pixel coordinates
    pix_coordx = []
    pix_coordy = []

    # For each galaxy in df_nearqso, find the x and y pixel coordinates and append them to the above pix_coord lists
    for index, row in df_nearqso.iterrows():
        coord = SkyCoord(row['RA'], row['DEC'], unit='deg')
        object_pix = wcs.world_to_pixel(coord)
        pix_coordx.append(float(object_pix[0]) + x_offset)
        pix_coordy.append(float(object_pix[1]) + y_offset)
    
    # Create columns in the df_nearqso Pandas DataFrame which contain the x and y pixel coordinates
    df_nearqso['pix_xcoord'] = pix_coordx
    df_nearqso['pix_ycoord'] = pix_coordy

    # Selects only the x and y pixel coordinate columns of df_nearqso
    coords = df_nearqso.loc[:,['pix_xcoord', 'pix_ycoord']]
    
    # Returns a Pandas DataFrame with the pixel coordinates of each desired galaxy
    return coords

#################### Function 4 ####################

def make_models(coords, int_mag=21.8, half_light_rad=5, ser_ind=0.6, ax_rat=0.7, pa=60, vary_loc=True, vary_int_mag=True, vary_half_light_rad=True, vary_ser_ind=True, vary_ax_rat=True, vary_pa=True):
    # Initialize an empty list which will contain the dictionaries defining each model
    models = []
    
    # Sets number to either 0 (do not vary) or 1 (do vary) for each parameter (default for all is True/1)
    loc_i = 1
    if vary_loc==False:
        loc_i = 0
    
    int_mag_i = 1
    if vary_int_mag==False:
        int_mag_i = 0
    
    half_light_rad_i = 1
    if vary_half_light_rad==False:
        half_light_rad_i = 0
        
    ser_ind_i = 1
    if vary_ser_ind==False:
        ser_ind_i = 0
        
    ax_rat_i = 1
    if vary_ax_rat==False:
        ax_rat_i = 0
    
    pa_i = 1
    if vary_pa==False:
        pa_i = 0
    
    # For each set of coordinates in coord, make a model using the default values inputted above
    for index, row in coords.iterrows():
        x = row['pix_xcoord']
        y = row['pix_ycoord']
        model = {
            0: 'sersic',
            1: str(x) + ' ' + str(y) + ' ' + str(loc_i) + ' ' + str(loc_i),
            3: str(int_mag) + ' ' + str(int_mag_i),
            4: str(half_light_rad) + ' ' + str(half_light_rad_i),
            5: str(ser_ind) + ' ' + str(ser_ind_i),
            9: str(ax_rat) + ' ' + str(ax_rat_i),
            10: str(pa) + ' ' + str(pa_i),
            'Z': 0
        }
        models.append(model)
    
    # Returns a list of dictionaries where each dictionary is a model to fit
    return models

#################### Function 5 ####################

def make_sky_model(path):
    # Takes the path to the flattened image file
    
    # Opens up the flattened image file to extract data used to find the mean value of the pixels
    hdulist = fits.open(path, memmap=True)
    hdu = hdulist[0]
    data = hdu.data
    background = np.mean(data)
    hdulist.close()

    # Defines a model for the sky
    sky = {
        0: 'sky',
        1: str(background) + ' 1',
        2: '0 0',
        3: '0 0',
        'Z': 0
    }

    # # Returns a dictionary used for the sky model
    return sky

#################### Function 6 ####################

# The code below was obtained and modified from Grillard's GalfitPyWrap GitHub page (https://github.com/Grillard/GalfitPyWrap)

# MIT License

# Copyright (c) 2017 Grillard

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def CreateFile(Iimg, coords, models, psf, final_fits, input_file, ZP, scale, size=100, mask='none', sky='Default', **kwargs):
    '''
        Creates a file to be run with galfit
        options can be given through kwargs
        models is a list of dicts where the keys are the model parameters.
        Note that region includes the initial pixel, ie, a box from 200 to 300 will have 101 pixels, in python this will be a[199:300]
        Example sersic model:
        {
         0  : 'sersic',      #  object type
         1  : '250 490 1 1', #  position x, y
         3  : '12. 1',       #  Integrated magnitude
         4  : '9 1',         #  R_e (half-light radius)   [pix]
         5  : '1.5 1',       #  Sersic index n (de Vaucouleurs n=4)
        'c0': '0 1',         #  Boxyness
         9  : '1 1',         #  axis ratio (b/a)
        10  : '0 1',         #  position angle (PA) [deg: Up=0, Left=90]
        'Z' :  0}            #  output option (0 = resid., 1 = Don't subtract)
        PLEASE be aware that if filenames are too long galfit will dump core! (psffile specifically!)
    '''
    
    # Split input_file name into two to prevent io error (do this before the for loop to prevent overwrite of name)
        
    in_name = input_file.split('.')
    out_name = final_fits.split('.')
    
    # Make an input file for each galaxy in coords table / models list
    
    for i in np.arange(len(coords)):
        gal_num = coords.index[i]
    
        # For each galaxy, get the pixel coordinates of the region box with 100 pixels for buffer (round to integer)
    
        xmin = int(coords['pix_xcoord'][gal_num] - size)
        xmax = int(coords['pix_xcoord'][gal_num] + size)
        ymin = int(coords['pix_ycoord'][gal_num] - size)
        ymax = int(coords['pix_ycoord'][gal_num] + size)
    
        # Get shape of image
    
        hdulist = fits.open(Iimg, memmap=True)
        data = hdulist[0].data
        hdulist.close()
        image_xmax = data.shape[1]
        image_ymax = data.shape[0]
    
    
        # If region for a galaxy extends past the edge of image, fix it
        if xmin < 0:
            xmin = 1
        if ymin < 0:
            ymin = 1
        if xmax > image_xmax:
            xmax = image_xmax
        if ymax > image_ymax:
            ymax = image_xmax
    
        region = [xmin, xmax, ymin, ymax]
    
        if len(models) == 0:
            print('Need at least one model!')
            return 1
        defdict = {
            'Iimg': Iimg,  # Input data image (FITS file)
            'Oimg': final_fits,  # Output data image block
            'Simg': 'none',  # Sigma Image
            'Pimg': psf,  # PSF Image
            'PSFf': '1',  # PSF fine sampling factor
            'badmask': mask,  # Bad pixel mask (FITS image or ASCII coord list)'
            'constr': 'none',  # File with parameter constraints (ASCII file) '
            'region': '{0} {1} {2} {3}'.format(region[0], region[1], region[2], region[3]), # Image region to fit (xmin xmax ymin ymax)'
            'convbox': '100 100',  # Size of the convolution box (x y)'
            'ZP': ZP,  # Magnitude photometric zeropoint '
            'scale': scale,  # Plate scale (dx dy)    [arcsec per pixel]'
            'dispt': 'regular',  # Display type (regular, curses, both)'
            'opt': '0',  # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps'
        }
        defdict.update(kwargs)
        
        # Add in the image parameters
        
        input_file = open(in_name[0] + '_' + str(i) + '_' + str(gal_num) + '.' + in_name[1], 'w')
        input_file.write('===== \n')
        input_file.write('# IMAGE and GALFIT CONTROL PARAMETERS \n')
        input_file.write('A) {0} \n'.format(defdict['Iimg']))
        input_file.write('B) ' + out_name[0] + '_' + str(i) + '_' + str(gal_num) + '.' + out_name[1] + ' \n')
        input_file.write('C) {0} \n'.format(defdict['Simg']))
        input_file.write('D) {0} \n'.format(defdict['Pimg']))
        input_file.write('E) {0} \n'.format(defdict['PSFf']))
        input_file.write('F) {0} \n'.format(defdict['badmask']))
        input_file.write('G) {0} \n'.format(defdict['constr']))
        input_file.write('H) {0} \n'.format(defdict['region']))
        input_file.write('I) {0} \n'.format(defdict['convbox']))
        input_file.write('J) {0} \n'.format(defdict['ZP']))
        input_file.write('K) {0} \n'.format(defdict['scale']))
        input_file.write('O) {0} \n'.format(defdict['dispt']))
        input_file.write('P) {0} \n'.format(defdict['opt']))
        input_file.write('\n# INITIAL FITTING PARAMETERS \n')

        # Add in the models for galaxy and sky
        
        emodels = list(models)
        if 'sky' not in [a[0] for a in emodels]:
            if sky=='Default':
                sky = {0: 'sky', 1: '1 1', 2: '0 0',
                   3: '0 0', 'Z': 0, 'Comment': 'StandardSky'}
            if sky!='None':
                emodels.append(sky)
        model = emodels[i]
        for j in np.arange(len(list(model.keys()))):
            key = list(model.keys())[j]
            s='{0}) {1} \n'.format(key, model[key])
            if key in ['Comment','mskidx','origin']:s='#'+s
            input_file.write(s)
        
        input_file.write('\n')
        
        # Added in this part so sky is added to each input correctly (sky will be the last entry in emodels)
        
        model = emodels[-1]
        for j in np.arange(len(list(model.keys()))):
            key = list(model.keys())[j]
            s='{0}) {1} \n'.format(key, model[key])
            if key in ['Comment','mskidx','origin']:s='#'+s
            input_file.write(s)
            
        input_file.close()
    return 0

#################### Function 7 ####################

def run_all(coords, input_file):
    long_command = ''

    for i in np.arange(len(coords)):
        gal_num = coords.index[i]
        long_command = long_command + 'galfit ' + input_file.split('.')[0] + '_' + str(i) + '_' + str(gal_num) + '.txt ; '

    return long_command

#################### Function 8 ####################

def extract_best_fit_param(path, df_nearqso, coords):
    size = 100 / 2
    
    # Takes the path to the fit.log file
    # This file must have no duplicates, be sure to delete all runs for the same galaxy except for the one you want to keep
    
    # Read in fit.log file
    with open(path, 'r') as file:
        data = file.read()

    # Remove a bunch of the unnecessary parts and split strings into lists
    data = data.split('-----------------------------------------------------------------------------')
    data = [x for x in data if x !='']
    data = [x for x in data if x !='\n']
    
    # Create a bunch of empty lists to populate

    gal_number = []
    xcoord = []
    xcoord_err = []
    ycoord = []
    ycoord_err = []
    int_mag = []
    int_mag_err = []
    half_light_rad = []
    half_light_rad_err = []
    ser_ind = []
    ser_ind_err = []
    ax_ratio = []
    ax_ratio_err = []
    pa = []
    pa_err = []
    rchi2 = []

    # For each galaxy fit, grab the parameter values and their respective errors. Then append them into the empty lists above
    # If there are any bad values (with *), this code removes the *'s and still gives the value (need to inspect/correct manually)

    for i in np.arange(len(data)):
        
        # Select the data we want
        
        data_i = data[i].split('\n')
        data_i = [x for x in data_i if x !='']
        
        # Removes all components that are not the main (first) one
        
        comp_num = (len(data_i) - 8) / 2
        if comp_num > 1:
            for i in np.arange(comp_num - 1):
                data_i.pop(6)
                data_i.pop(6)
                
        # Extracts each parameter and puts it into their respective list
    
        gal_number_i = data_i[1].split('_')[-1].replace('.txt', '')
        gal_number.append(int(gal_number_i))
    
        xcoord_i = [x for x in data_i[4].split(' ') if x != ''][3].replace(',', '')
        if '*' in xcoord_i:
            xcoord_i = xcoord_i.split('*')[1]
        xcoord.append(float(xcoord_i))
    
        xcoord_err_i = [x for x in data_i[5].split('  ') if x != ''][1].replace(',', '')
        if '*' in xcoord_err_i:
            xcoord_err_i = xcoord_err_i.split('*')[1]
        xcoord_err.append(float(xcoord_err_i))
    
        ycoord_i = [x for x in data_i[4].split(' ') if x != ''][4].replace(')', '')
        if '*' in ycoord_i:
            ycoord_i = ycoord_i.split('*')[1]
        ycoord.append(float(ycoord_i))
    
        ycoord_err_i = [x for x in data_i[5].split('  ') if x != ''][2].replace(')', '').replace(' ', '')
        if '*' in ycoord_err_i:
            ycoord_err_i = ycoord_err_i.split('*')[1]
        ycoord_err.append(float(ycoord_err_i))
    
        int_mag_i = [x for x in data_i[4].split('  ') if x != ''][3].replace(' ', '').replace(')', '')
        if '*' in int_mag_i:
            int_mag_i = int_mag_i.split('*')[1]
        int_mag.append(float(int_mag_i))
    
        int_mag_err_i = [x for x in data_i[5].split('  ') if x != ''][3]
        if '*' in int_mag_err_i:
            int_mag_err_i = int_mag_err_i.split('*')[1]
        int_mag_err.append(float(int_mag_err_i))
    
        half_light_rad_i = [x for x in data_i[4].split('  ') if x != ''][4]
        if '*' in half_light_rad_i:
            half_light_rad_i = half_light_rad_i.split('*')[1]
        half_light_rad.append(float(half_light_rad_i))
    
        half_light_rad_err_i = [x for x in data_i[5].split('  ') if x != ''][4]
        if '*' in half_light_rad_err_i:
            half_light_rad_err_i = half_light_rad_err_i.split('*')[1]
        half_light_rad_err.append(float(half_light_rad_err_i))
    
        ser_ind_i = [x for x in data_i[4].split('  ') if x != ''][5]
        if '*' in ser_ind_i:
            ser_ind_i = ser_ind_i.split('*')[1]
        ser_ind.append(float(ser_ind_i))
    
        ser_ind_err_i = [x for x in data_i[5].split('  ') if x != ''][5]
        if '*' in ser_ind_err_i:
            ser_ind_err_i = ser_ind_err_i.split('*')[1]
        ser_ind_err.append(float(ser_ind_err_i))
    
        ax_ratio_i = [x for x in data_i[4].split('  ') if x != ''][6]
        if '*' in ax_ratio_i:
            ax_ratio_i = ax_ratio_i.split('*')[1]
        ax_ratio.append(float(ax_ratio_i))
    
        ax_ratio_err_i = [x for x in data_i[5].split('  ') if x != ''][6]
        if '*' in ax_ratio_err_i:
            ax_ratio_err_i = ax_ratio_err_i.split('*')[1]
        ax_ratio_err.append(float(ax_ratio_err_i))
    
        pa_i = [x for x in data_i[4].split('  ') if x != ''][7].replace(' ', '')
        if '*' in pa_i:
            pa_i = pa_i.split('*')[1]
        pa.append(float(pa_i))
    
        pa_err_i = [x for x in data_i[5].split('  ') if x != ''][7].replace(' ', '')
        if '*' in pa_err_i:
            pa_err_i = pa_err_i.split('*')[1]
        pa_err.append(float(pa_err_i))
    
        rchi2_i = data_i[-1].replace(' Chi^2/nu = ', '').replace(' ', '')
        rchi2.append(float(rchi2_i))
        
    # Get redshifts for each galaxy
    redshifts = df_nearqso['z'].values.tolist()

    # Create a data table with the best fit parameters and their errors for each galaxy
    best_fit_table = pd.DataFrame(
        {'gal_num': gal_number,
         'xcoord': xcoord,
         'xcoord_error': xcoord_err,
         'ycoord': ycoord,
         'ycoord_error': ycoord_err,
         'int_mag': int_mag,
         'int_mag_error': int_mag_err,
         'half_light_rad': half_light_rad,
         'half_light_rad_error': half_light_rad_err,
         'sersic_index': ser_ind,
         'sersic_index_error': ser_ind_err,
         'axis_ratio': ax_ratio,
         'axis_ratio_error': ax_ratio_err,
         'position_angle': pa,
         'position_angle_error': pa_err,
         'reduced_chi_squared': rchi2,
        }
    )

    # Sorts the table and re-indexes based on the galaxy number
    best_fit_table = best_fit_table.sort_values(['gal_num']).reset_index(drop=True)
    
    # Adds redshifts into dataframe as second column
    best_fit_table.insert(1, 'redshift', redshifts)
    
    # Returns a Pandas DataFrame with the fit parameters from fit.log if length of df equals length of coords
    # Make sure there are no duplicates in case not all galaxies are modeled and there are duplicates!
    if len(best_fit_table) == len(coords):
        return best_fit_table
    # If the fit.log contains too little models (some models could not run at all) 
    elif len(best_fit_table) < len(coords):
        not_acc_for = []

        coord_index = coords.index.values.tolist()
        best_fit_index = best_fit_table['gal_num'].tolist()

        for i in np.arange(len(coord_index)):
            test = coord_index[i]
            if test in best_fit_index:
                pass
            else:
                not_acc_for.append(i)
    
        for i in not_acc_for:
            gal_num_i = coords[i:i+1].index.values.tolist()[0]
            xcoord_i = coords[i:i+1]['pix_xcoord'].values[0]
            ycoord_i = coords[i:i+1]['pix_ycoord'].values[0]
        
            o = 'oob'
            add_row = {
                'gal_num': gal_num_i,
                'xcoord': xcoord_i,
                'xcoord_error': o,
                'ycoord': ycoord_i,
                'ycoord_error': o,
                'int_mag': 'oob',
                'int_mag_error': o,
                'half_light_rad': o,
                'half_light_rad_error': o,
                'sersic_index': o,
                'sersic_index_error': o,
                'axis_ratio': o,
                'axis_ratio_error': o,
                'position_angle': o,
                'position_angle_error': o,
                'reduced_chi_squared': o
            }
            best_fit_table = best_fit_table.append(add_row, ignore_index=True)

        best_fit_table.sort_values(['gal_num'], inplace=True)
        best_fit_table.reset_index(drop=True, inplace=True)
    
        return best_fit_table
    
    # If the fit.log contains too much models (there are duplicates)
    else:
        return 'There are more models in fit.log than rows in the coords table. Check for duplicates!'

#################### Function 9 ####################

# Create an input file containing all of the above galaxies' (plus the sky's) models
# Based on the code from galfitwrap.py from Grillard's galfitpywrap GitHub page (https://github.com/Grillard/GalfitPyWrap)

# MIT License

# Copyright (c) 2017 Grillard

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def CreateSingleFile(Iimg, models, psf, final_fits, input_file, ZP, scale, sky_value, **kwargs):
    # Iimg is input fits file with the data
    # models is the df from the previous step with the best fit parameters for all of the galaxies
    # psf is the name of the input psf file, set to 'none' if no psf is used
    # final_fits is the name of the output fits file
    # input_file is the output of this function, to be inputted into galfit along with the input fits file
    # ZP is the magnitude photometric zeropoint
    # scale is the conversion between one image pixel and arcseconds
    # sky_value is the value of the image background
    
    # Get shape of image
    hdulist = fits.open(Iimg, memmap=True)
    data = hdulist[0].data
    hdulist.close()
    image_xmax = data.shape[1]
    image_ymax = data.shape[0]
    
    # Set image size parameter to be the same as the input image
    region = [1, image_xmax, 1, image_ymax]
    
    # Check if the models df is empty
    if len(models) == 0:
            print('Need at least one model!')
            return 1
    
    defdict = {
            'Iimg': Iimg,  # Input data image (FITS file)
            'Oimg': final_fits,  # Output data image block
            'Simg': 'none',  # Sigma Image
            'Pimg': psf,  # PSF Image
            'PSFf': '1',  # PSF fine sampling factor
            'badmask': 'none',  # Bad pixel mask (FITS image or ASCII coord list)'
            'constr': 'none',  # File with parameter constraints (ASCII file) '
            'region': '{0} {1} {2} {3}'.format(region[0], region[1], region[2], region[3]), # Image region to fit (xmin xmax ymin ymax)'
            'convbox': '100 100',  # Size of the convolution box (x y)'
            'ZP': ZP,  # Magnitude photometric zeropoint '
            'scale': scale,  # Plate scale (dx dy)    [arcsec per pixel]'
            'dispt': 'regular',  # Display type (regular, curses, both)'
            'opt': '2',  # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps'
        }
    defdict.update(kwargs)
        
    # Add in the image parameters
    input_file = open(input_file, 'w')
    input_file.write('===== \n')
    input_file.write('# IMAGE and GALFIT CONTROL PARAMETERS \n')
    input_file.write('A) {0} \n'.format(defdict['Iimg']))
    input_file.write('B) {0} \n'.format(defdict['Oimg']))
    input_file.write('C) {0} \n'.format(defdict['Simg']))
    input_file.write('D) {0} \n'.format(defdict['Pimg']))
    input_file.write('E) {0} \n'.format(defdict['PSFf']))
    input_file.write('F) {0} \n'.format(defdict['badmask']))
    input_file.write('G) {0} \n'.format(defdict['constr']))
    input_file.write('H) {0} \n'.format(defdict['region']))
    input_file.write('I) {0} \n'.format(defdict['convbox']))
    input_file.write('J) {0} \n'.format(defdict['ZP']))
    input_file.write('K) {0} \n'.format(defdict['scale']))
    input_file.write('O) {0} \n'.format(defdict['dispt']))
    input_file.write('P) {0} \n'.format(defdict['opt']))
    input_file.write('\n# MODEL PARAMETERS \n')
    
    # Add in the models for each galaxy
    for index, row in models.iterrows():
        input_file.write('0) sersic \n')
        input_file.write('1) {0} {1} 0 0 \n'.format(str(row['xcoord']), str(row['ycoord'])))
        input_file.write('3) {0} 0 \n'.format(str(row['int_mag'])))
        input_file.write('4) {0} 0 \n'.format(str(row['half_light_rad'])))
        input_file.write('5) {0} 0 \n'.format(str(row['sersic_index'])))
        input_file.write('9) {0} 0 \n'.format(str(row['axis_ratio'])))
        input_file.write('10) {0} 0 \n'.format(str(row['position_angle'])))
        input_file.write('Z) 0 \n \n')
    
    # Add in the sky model
    input_file.write('0) sky \n')
    input_file.write('1) {0} \n'.format(str(sky_value)))
    input_file.write('2) 0 0 \n')
    input_file.write('3) 0 0 \n')
    input_file.write('Z) 0 \n')

#################### Function 10 ####################

def find_phi(unflattened_data, near_qso_df, best_fit_table):
    # Obtain wcs object from the unflattened fits file header
    hdulist = fits.open(unflattened_data, memmap=True)
    hdu = hdulist[1]
    wcs = WCS(hdu.header)
    hdulist.close()
    
    # Finds the RA and DEC of the QSO from near_qso_df
    qso_ra = near_qso_df['RA_QSO'][near_qso_df.index[0]]
    qso_dec = near_qso_df['DEC_QSO'][near_qso_df.index[0]]
    
    # Converts RA/DEC of the QSO to image x/y
    coord = SkyCoord(qso_ra, qso_dec, unit='deg')
    object_pix = wcs.world_to_pixel(coord)
    qso_pix_coordx = object_pix[0]
    qso_pix_coordy = object_pix[1]
    
    # Finds the changes in pixel x and y coordinates from each galaxy to the QSO
    delta_x = qso_pix_coordx - best_fit_table['xcoord']
    delta_y = qso_pix_coordy - best_fit_table['ycoord']
    
    # Calculates the azimuthal angle phi for each galaxy
    phi = []
    for i in np.arange(len(delta_x)):
        if best_fit_table['position_angle'][i] != 'oob':
            # Create vector a which is the vector pointing from galaxy to QSO
            a = [delta_x[i], delta_y[i]]
    
            # Create vectors b1 and b2 which are vectors pointing along galaxy's semi-minor axis
            # b1 and b2 are anti-parallel
            position_angle = best_fit_table['position_angle'][i] * (np.pi / 180)
            b1 = [np.cos(position_angle), np.sin(position_angle)]
            b2 = [-np.cos(position_angle), -np.sin(position_angle)]
    
            # Uses the relationship between the cross product of two vectors and cosine to find the angle between the two vectors
            phi_prime1 = np.arccos(np.dot(a, b1) / (lin.norm(a) * lin.norm(b1))) * (180 / np.pi)
            phi_prime2 = np.arccos(np.dot(a, b2) / (lin.norm(a) * lin.norm(b2))) * (180 / np.pi)
        
            # Takes the lesser of the two angles since azimuthal angle can go from 0 to +90 degrees
            if phi_prime1 <= phi_prime2:
                phi.append(phi_prime1)
            elif phi_prime1 > phi_prime2:
                phi.append(phi_prime2)
        else:
            phi.append('oob')
    
    # Returns the best_fit_table with phi column included
    best_fit_table['azimuthal_angle'] = phi
    return best_fit_table

#################### Function 11 ####################

def find_inclination(best_fit_table):
    inclins = []

    for index, row in best_fit_table.iterrows():
        ax_rat_i = row['axis_ratio']
        if ax_rat_i != 'oob':
            inclin_i = np.arccos(ax_rat_i) * (180 / np.pi)
            inclins.append(inclin_i)
        else:
            inclins.append('oob')
        
    best_fit_table['inclination_angle'] = inclins
    return best_fit_table

#################### Function 12 ####################
def apply_flags(best_fit_table, bad_fits_in, spiral_struct_in, overlap_obj_in, pixel_thresh,  edited_in, double_model_in):
#def apply_flags(best_fit_table, bad_fits_in, internal_struct_in, pixel_thresh, dark_center_res_in, overlap_in):
    
    # Applies "is_good_fit" flag
    good_fit = [True] * len(best_fit_table)
    for i in bad_fits_in:
        good_fit[i] = False
        
    # Applies "shows_spiral_struct" flag
    internal_struct = [False] * len(best_fit_table)
    for i in internal_struct_in:
        internal_struct[i] = True
        
    # Applies "overlaps" flag
    has_overlap = [False] * len(best_fit_table)
    for i in overlap_in:
        has_overlap[i] = True
    
    # Applies "is_big_galaxy" flag
    large_galaxy = [True] * len(best_fit_table)
    all_hlr = best_fit_table['half_light_rad']
    for i in np.arange(len(best_fit_table)):
        current_hlr = all_hlr[i]
        if current_hlr != 'oob':
            pass
        else:
            current_hlr = 0
        
        if current_hlr >= pixel_thresh:
            pass
        elif current_hlr < pixel_thresh:
            large_galaxy[i] = False
            
    # Applies "has_residual_dark_center" flag
    # - REMOVING- it will be assumed that 'double_model' galaxies automatically have dark center in resid.
    #has_dark_center_res = [False] * len(best_fit_table)
    #for i in dark_center_res_in:
    #    has_dark_center_res[i] = True
    
    ## Add for edited galaxies/ those varying from the initial parameters
    edited = [False] * len(best_fit_table)
    for i in edited_in:
        edited[i] = True
        
     ## Applies "double model" flag for when concentric sersic/ ally method is used
    double_model = [False] * len(best_fit_table)
    for i in double_model_in:
        double_model[i] = True
    
    # Adds flags as columns in best_fit_table DataFrame
    best_fit_table['is_good_fit'] = good_fit
    best_fit_table['shows_internal_struct'] = internal_struct
    best_fit_table['overlaps_w_object'] = has_overlap
    best_fit_table['is_big_galaxy'] = large_galaxy
    best_fit_table['edited'] = edited
    best_fit_table['double_model'] = double_model
    # best_fit_table['has_dark_center_res'] = has_dark_center_res
    
    return best_fit_table
