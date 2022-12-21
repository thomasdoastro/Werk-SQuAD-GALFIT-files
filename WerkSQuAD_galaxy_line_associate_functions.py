# This is the function used for associating transition lines to host galaxies
# Host galaxy: Within 500 km/s of the line and has the smallest rho_rvir

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table

# This function is for running qso fields one at a time
def associate_line_galaxy(galfit_csv_path, full_galaxy_path, line_csv_path, qso_name, rec_vel_threshold):
    
    # Reads in the galaxies for a particular qso sightline
    # Each galaxy has redshift, azimuthal angle, and inclination angle
    # Need to get rho_rvir from the full galaxy table using gal_num as index
    df_galfit = pd.read_csv(galfit_csv_path)
    
    # Reads in full galaxy table
    # We will get galaxy rho_rvir from this table

    fit1 = fits.open(full_galaxy_path, memmap=True)
    table = Table(fit1[1].data)
    fit1.close()

    # Gets the column names
    column_array = []
    for column in table.columns:
        column = str(column)
        column_array.append(column)

    # Turns Astropy Table into a Pandas DataFrame, then sorts based on QSO field ('OBJECT') and redshift ('z')
    df_full_galaxy = table[column_array].to_pandas()
    df_full_galaxy = df_full_galaxy.sort_values(["OBJECT", "z"], ascending=True).reset_index(drop=True)
    
    # Read in spectral lines
    # Contains all components for different transitions (multiplets)
    # Each line has col density and zsys (use zsys to compare with galaxies)
    df_lines = pd.read_csv(line_csv_path)
    
    # Gets the lines for a particular qso field
    df_lines_galfit = df_lines[df_lines['QSO'] == qso_name]
    df_lines_galfit = df_lines_galfit.reset_index()
    
    # The for loop for running through list of line redshifts and galaxy redshifts to see if they are within 500 km/s of each other
    c = 3*10**5 # km/s
    one_true_host = []

    for index, irow in df_lines_galfit.iterrows():

        near_redshift_galaxies_i = []

        # Looks for galaxies within the recessional velocity constraint (500 km/s) of the line, gets their galaxy number
        for jndex, jrow in df_galfit.iterrows():
            z1 = irow['zsys']
            b1 = (1 + z1)**2
            v1 = (b1 - 1) / ((1 / c) + (b1 / c))

            z2 = jrow['redshift']
            b2 = (1 + z2)**2
            v2 = (b2 - 1) / ((1 / c) + (b2 / c))

            diff = np.abs(v1 - v2)

            if diff <= 500:
                near_redshift_galaxies_i.append(jrow['gal_num'])

        # Insert a column for the "one true host" for the line (within recessional velocity constraint and closest rho_rvir)
        # If there are no galaxies nearby in redshift
        if len(near_redshift_galaxies_i) == 0:
            one_true_host.append('None')

        # If there is only one galaxy nearby in redshift
        elif len(near_redshift_galaxies_i) == 1:
            one_true_host.append(near_redshift_galaxies_i[0])

        # If there are multiple galaxies nearby in redshift, choose the one with the lowest rho_rvir as given in the full_galaxy_table
        else:
            rho_rvir = []
            for k in near_redshift_galaxies_i:
                rho_rvir_k = df_full_galaxy.iloc[k]['rho_rvir']
                rho_rvir.append(rho_rvir_k)

            min_rho_rvir_index = np.argmin(rho_rvir)
            one_true_host.append(near_redshift_galaxies_i[min_rho_rvir_index])

    df_lines_galfit['true_host_num'] = one_true_host
    
    # Adding on the parameters of each line's host galaxy
    # Parameters included: azimuthal angle, inclination angle

    azangle = []
    inclangle = []

    for index, row in df_lines_galfit.iterrows():
        gal_num_i = row['true_host_num']

        if gal_num_i == 'None':
            azangle.append('None')
            inclangle.append('None')

        else:
            azangle_i = float(df_galfit[df_galfit['gal_num'] == gal_num_i]['azimuthal_angle'])
            inclangle_i = float(df_galfit[df_galfit['gal_num'] == gal_num_i]['inclination_angle'])

            azangle.append(azangle_i)
            inclangle.append(inclangle_i)

    df_lines_galfit['host_azimuthal_angle'] = azangle
    df_lines_galfit['host_inclination_angle'] = inclangle
    
    return df_lines_galfit
    