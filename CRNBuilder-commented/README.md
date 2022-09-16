# CRNBuilder

CRNBuilder is a Python program that is able to automatically generate a CRN of ideal reactors from the results of a CFD
simulation.

## Index

1. [Dependencies](#Dependencies)
2. [Folder structure](#Folder structure)
3. [Usage](#Usage)
4. [Settings file](#Settings file)
5. [Examples](#Examples)
6. [License](#License)

## Dependencies

All the libraries required by the program are present in the file `requirements.txt`, to install them through pip just
run the command `pip install -r requirements.txt`.

**WARNING** - To make the code work it is required python >= 3.8

## Folder structure

All the python scripts are contained in the main folder (except for `SagittalAngle.py`, which is outdated and not used
anymore). The input cases are contained in the folder `data`, where there is a folder for each one of the possible input
cases. Inside each one of these folders, there is a folder named `case` containing the OpenFOAM case and a file
called `CRNBinput.dic` containing the settings of the program. Inside the `NetSMOKE` folder the program NetSMOKE can be
put in order to solve the generated CRN automatically through the file `run.sh`.

In the folder `.old` there is a script that can be used to compute the sagittal and azimuth angles, however it is
outdated since it is using an old method to cycle through all the cells

## Usage

_It is advised to operate inside a virtual environment, for more information
check [the official documentation](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system.)
._

This program in its current state is able to generate CRNs starting from solved OpenFOAM cases. In order to do that, you need to:

1. Create, inside the `data` folder, a subfolder named accordingly to the case name (for instance, the case could be
   named `example-case`).
2. Inside the `example-case` folder, a subfolder should be created named `case`, containing the solved OpenFOAM case.
3. The OpenFOAM case should be pre-processed through the following (OpenFOAM) commands:
    - `postProcess -func writeCellCentres`
    - `postProcess -func writeCellVolumes`
4. Inside the `example-case` folder, a file called `CRNBInput.dic` should be created and filled according to the next
   section (cf. _Settings file_)
5. Inside the file `CRNBuilder.py`, at line `755`, change the variable `case_name` according to the case name (for
   instance, `case_name = "example-case"`)
6. From the terminal, run the command `python .\CRNBuilder.py x`, where `x` is the number of objective reactors of the
   CRN (for instance, to obtain a CRN of 10 reactors run the command `python .\CRNBuilder.py 10`)
7. Since at the current version the program only supports structured grid, the user is required to input the number of
   cells along the radial and axial direction.
8. The program is then going to ask whether to save the case or not, it is usually better to save it by writing `y` in
   the terminal and confirming by pressing `Enter`.
9. When the script has finished running, inside the `example-case` folder two new subfolders are present:
    - `cache`: containing the cache of the program, it is not of interest for the end user.
    - `out`: this folder contains the file `inputRN.dic`, to be used as an input for NetSMOKE, and a few files
      representing the system.
10. In order to solve the generated CRN in NetSMOKE, the `inputRN.dic` file should be modified to correct the input
    species (at the moment the program is not able to set them automatically).

## Settings file

The file `CRNB_input.dic` contains the settings for a specific case.\
Each command is structured as follows:

```
@Command
variable1: value
variable2: value, value
//
```

At the current state every command is mandatory. The full list of commands is:

### Inlets

Through this command the inlets present in the OpenFOAM case need to be specified. Each variable is the actual name used
for the inlet in the OpenFOAM case. The first value of each variable is a string and can be north, south, east or west;
it is used to specify on which side the inlet is. The second and third values are floats, and are used to specify where
the inlet starts and ends (for instance, for an inlet present on the bottom part of the system, the first value is the
string "south", the second value is the starting _x_ coordinate, and the third value is the ending _x_ coordinate).

### Outlets

Through this command the outlets present in the OpenFOAM case need to be specified. The variables and values are the
same as in the **Inlets** command.

### Dimensions

This command is used to define some geometrical properties of the system. The variables are:

- min_y : this variable requires only one value, that is a float corresponding to the minimum value along the radial
  direction.
- max_y : this variable requires only one value, that is a float corresponding to the maximum value along the radial
  direction.
- min_z : this variable requires only one value, that is a float corresponding to the maximum value along the axial
  direction.
- max_z : this variable requires only one value, that is a float corresponding to the maximum value along the axial
  direction.
- angle : this variable requires only one value, that is a float corresponding to the angle of the slice used to
  simulate the OpenFOAM case. If the case is not axi-symmetric, this value should be 360.

### OperativeConditions

This command is used to define the operative conditions of the system. The only variable is:

- pressure : this variable requires only one value, that is a float corresponding to the pressure of the system.

### Geometry

This command is used to define some other geometrical properties of the system. The variables are:

- radial_dir : this variable requires only one value, that is a string that can be "x", "y" or "z", and is used to
  define which axis corresponds to the radial direction in the OpenFOAM case.
- axial_dir : this variable requires only one value, that is a string that can be "x", "y" or "z", and is used to define
  which axis corresponds to the axial direction in the OpenFOAM case.

### Parameters

This command is used to define some parameters of the system. The only variable is:

- T_threshold : this variable requires only one value, that is a float corresponding to the threshold Temperature used
  to cluster all the cold zones. It is generally a good idea to put a value under which the reactivity of the system is
  limited or the minimum Temperature of the system. For more information, it is advised to read the thesis on which this
  program is based.

## Examples

The program comes with two test cases, described thoroughly in the thesis on which this work is based: the HM1
bluff-body flame and the Sandia flame D.

### HM1 bluff-body flame

This case is based on the HM1 bluff-body flame described [here](https://web.aeromech.usyd.edu.au/thermofluids/bluff.php)
. It is a CH4 and H2 flame with a coflow of air, the reader is redirected to the link for further information. The
OpenFOAM case is found in the folder `data/HM1_bluff-body_flame/case`. To run it, be sure to change the `case_name`
variable to `HM1_bluff-body_flame` in the file `CRNBuilder.py`, then run from the terminal the
command `python .\CRNBuilder.py 10`, to obtain a CRN of 10 reactors.

The original `Ny` is `290`, and the original `Nz` is `300` (the user is required to insert those values when requested
by the program). When prompted whether to save the cache or not, it is better to answer "yes" by writing `y` in the
terminal and confirming with `Enter`.

### Sandia flame D

This case is based on the Sandia flame D described [here](https://tnfworkshop.org/data-archives/pilotedjet/ch4-air/). It
is a CH4 flame with a coflow of air and the presence of a pilot flame, the reader is redirected to the link for further
information. The OpenFOAM case is found in the folder `data/Sandia_flame_D/case`. To run it, be sure to change
the `case_name` variable to `Sandia_flame_D` in the file `CRNBuilder.py`, then run from the terminal the
command `python .\CRNBuilder.py 10`, to obtain a CRN of 10 reactors.

The original `Ny` is `252`, and the original `Nz` is `450` (the user is required to insert those values when requested
by the program). When prompted whether to save the cache or not, it is better to answer "yes" by writing `y` in the
terminal and confirming with `Enter`.

## License
