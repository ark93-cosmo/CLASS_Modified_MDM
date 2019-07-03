# CLASS_Modified_MDM
This is the modified version of CLASS to study Mimetic Dark Matter Model. The files that have been modified are:
1-explanatory.ini: in this file we added 3 parameters to specify mimetic field initial conditions (energy density and velocity perturbation)
2-source/input.c: In this file , we've added the same lines as for cdm, and named it "mimetic". Moreover, we added lines for the code to read the relevant parameters from the .ini file
3-source/perturbations.c: In addition to adding the same line as for cdm, we added the options of having two different kinds of initial conditions for the velocity perturbations, which can be chosen by specifying their corresponding parameter in the .ini file
Finally, in source/spectra.c, include/perturbations.h and include/spectra.h, the lines of cdm has been copied and pasted as "mimetic"
