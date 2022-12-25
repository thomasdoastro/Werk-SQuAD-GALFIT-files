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

    # Initialize lists to hold full_galaxy_table properties
    rho = []
    rho_rvir = []
    mstar = []
    logmhalo = []
    
    # Initialize lists to hold GALFIT properties
    azangle = []
    inclangle = []
    axis_ratio = []
    ar_error= []
    
    # Initialize lists to hold flags
    good = []
    struct = []
    overlap = []
    big = []
    edited = []
    double = []

    for index, row in df_lines_galfit.iterrows():
        gal_num_i = row['true_host_num']

        if gal_num_i == 'None':
            # full_galaxy_table properties if there is no host galaxy
            rho.append('None')
            rho_rvir.append('None')
            mstar.append('None')
            logmhalo.append('None')
            
            # GALFIT properties if there is no host galaxy
            azangle.append('None')
            inclangle.append('None')
            axis_ratio.append('None')
            ar_error.append('None')
            
            # Flags if there is no host galaxy
            good.append('None')
            struct.append('None')
            overlap.append('None')
            big.append('None')
            edited.append('None')
            double.append('None')

        else:
            # full_galaxy_table properties for each host galaxy
            rho_i = float(df_full_galaxy.iloc[gal_num_i]['rho'])
            rho_rvir_i = float(df_full_galaxy.iloc[gal_num_i]['rho_rvir'])
            mstar_i = float(df_full_galaxy.iloc[gal_num_i]['mstars'])
            logmhalo_i = float(df_full_galaxy.iloc[gal_num_i]['logmhalo'])
            
            # GALFIT properties for each host galaxy
            azangle_i = float(df_galfit[df_galfit['gal_num'] == gal_num_i]['azimuthal_angle'])
            inclangle_i = float(df_galfit[df_galfit['gal_num'] == gal_num_i]['inclination_angle'])
            axis_ratio_i = float(df_galfit[df_galfit['gal_num'] == gal_num_i]['axis_ratio'])
            axis_ratio_error_i = float(df_galfit[df_galfit['gal_num'] == gal_num_i]['axis_ratio_error'])
            
            # Adding our 6 flags (as of 12/23)
            good_i = df_galfit[df_galfit['gal_num'] == gal_num_i]['is_good_fit']
            good_i = bool(good_i[good_i.index[0]])
            struct_i = df_galfit[df_galfit['gal_num'] == gal_num_i]['shows_spiral_struct']
            struct_i = bool(struct_i[struct_i.index[0]])
            overlap_i = df_galfit[df_galfit['gal_num'] == gal_num_i]['overlaps_w_object']
            overlap_i = bool(overlap_i[overlap_i.index[0]])
            big_i = df_galfit[df_galfit['gal_num'] == gal_num_i]['is_big_galaxy']
            big_i = bool(big_i[big_i.index[0]])
            edited_i = df_galfit[df_galfit['gal_num'] == gal_num_i]['edited']
            edited_i = bool(edited_i[edited_i.index[0]])
            double_i= df_galfit[df_galfit['gal_num'] == gal_num_i]['double_model']
            double_i = bool(double_i[double_i.index[0]])

            # full_galaxy_table properties appended
            rho.append(rho_i)
            rho_rvir.append(rho_rvir_i)
            mstar.append(mstar_i)
            logmhalo.append(logmhalo_i)
            
            # GALFIT properties appended
            azangle.append(azangle_i)
            inclangle.append(inclangle_i)
            axis_ratio.append(axis_ratio_i)
            ar_error.append(axis_ratio_error_i)

            # Flags appended
            good.append(good_i)
            struct.append(struct_i)
            overlap.append(overlap_i)
            big.append(big_i)
            edited.append(edited_i)
            double.append(double_i)

    # full_galaxy_table property columns added
    df_lines_galfit['host_rho'] = rho
    df_lines_galfit['host_rho_rvir'] = rho_rvir
    df_lines_galfit['host_mstars'] = mstar
    df_lines_galfit['host_logmhalo'] = logmhalo
    
    # GALFIT property columns added
    df_lines_galfit['host_azimuthal_angle'] = azangle
    df_lines_galfit['host_inclination_angle'] = inclangle
    df_lines_galfit['host_axis_ratio'] = axis_ratio
    df_lines_galfit['host_axis_ratio_error'] = ar_error
    
    # Flag columns added
    df_lines_galfit['host_good_fit'] = good
    df_lines_galfit['host_shows_structure'] = struct
    df_lines_galfit['host_overlaps'] = overlap
    df_lines_galfit['host_big'] = big
    df_lines_galfit['host_edited'] = edited
    df_lines_galfit['host_double_model'] = double
    
    return df_lines_galfit
    
