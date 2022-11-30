# Werk-SQuAD-GALFIT-files
---

# Requirements

To run these files, you need to:
* have GALFIT aliased or have the path to GALFIT
* have Pandas, Numpy, Astropy, and Importlib installed

---

# Usage

Once you download these files, put them in a directory where you can duplicate or copy/paste these files to whichever QSO sightline folder you have.

If you do not have GALFIT aliased, then you will need to go into the WerkSQuAD_galfit_functions.py file. In line 357 (under Function 7) change the string 'galfit ' to '[path] ' where [path] represents the path from your home directory to where you have the galfit executable stored.

Once you have satisfied the above requirements, you can follow along with the instructions outlined in the WerkSQuAD_galfit_notebook.ipynb file. They should guide you along the process of creating GALFIT input files, running GALFIT on these files, and reading in the best fit parameters.

---

# Limitations

There are some limitations for this code that are important to keep in mind when using it:
* The fit.log file must containn only one of each galaxy before using the extract_best_fit_param function. Duplicates will be produced in the resulting table if this warning is not heeded.
* In cases where you have fitted multiple profiles for a single input file, the extract_best_fit_param function will always get the parameters of the first profile. This first profile is usually (but not always!) the galaxy you want to fit. Be wary of cases where the first profile is not the galaxy you want model parameters from.
