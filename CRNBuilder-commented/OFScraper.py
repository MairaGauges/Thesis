import numpy as np
import Ofpp_patched
import os

from InputParser import parse_input


def get_last_time(case_path: str):
    """
    search for the last time of the simulation
    :param case_path: path to the OpenFOAM case, default to the current working dir
    :return: float corresponding to the last time found
    """
    dirs = os.listdir(case_path)
    timeDirs = []
    for dir in dirs:
        try:
            float(dir)
            timeDirs.append(dir)
        except ValueError:
            continue
    return max(timeDirs)


def read_case(case_name: str, path: str):
    """
    read and parse an OpenFOAM case, extracting the data needed to build a data.dat file\n
    WARNING: the case should be preprocessed by running the `postProcess -func 'writeCellCentres' -time 0' command
    :param case_name: the name of the case to be processed
    :param path: path to the OpenFOAM case
    :return:
     1) numpy array: [Nc x 3] containing the cells coordinates
     2) numpy array: [Nc] containing the T field
     3) numpy array: [Nc x 3] containing the U field
     4) int: total number of cells
     5) str: the name of the case
    """
    print(f"Reading the case named '{case_name}'...")
    last_time = get_last_time(os.path.join(f"{path}", f"case", f""))

    # Read the mesh infos and get X, Y, and Z data
    mesh = Ofpp_patched.FoamMesh(os.path.join(f"{path}", f"case", f""))

    if not os.path.isfile(os.path.join(f"{path}", f"case", f"{last_time}", f"C")):
        raise RuntimeError(f"Could not find cell centres data, please make sure to run the command 'postProcess -func writeCellCentres' from OpenFOAM and check if the file 'C' is present in the last "
                           f"time folder")
    if not os.path.isfile(os.path.join(f"{path}", f"case", f"{last_time}", f"V")):
        raise RuntimeError(f"Could not find cell volumes data, please make sure to run the command 'postProcess -func writeCellVolumes' from OpenFOAM and check if the file 'V' is present in the last "
                           f"time folder")
    mesh.read_cell_centres(os.path.join(f"{path}", f"case", f"{last_time}", f"C"))
    cells = mesh.cell_centres
    Nc = cells.shape[0]

    # Read T, Ux, Uy and Uz data
    V = Ofpp_patched.parse_internal_field(os.path.join(f"{path}", f"case", f"{last_time}", f"V"))
    T = Ofpp_patched.parse_internal_field(os.path.join(f"{path}", f"case", f"{last_time}", f"T"))
    U = Ofpp_patched.parse_internal_field(os.path.join(f"{path}", f"case", f"{last_time}", f"U"))
    rho = Ofpp_patched.parse_internal_field(os.path.join(f"{path}", f"case", f"{last_time}", f"rho"))

    return cells, T, U, rho, V, Nc


def build_dat(case_name: str):
    """
    build a data.dat file that can act as an input
    WARNING: the case should be preprocessed by running the `postProcess -func 'writeCellCentres' -time 0` and `postProcess -func 'writeCellVolumes' -time 0` commands
    :param case_name: the name of the case to be processed
    :return: None
    """
    # Read input file
    _, _, _, _, _, _, radial_dir, axial_dir, _ = parse_input(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"CRNB_input.dic"))
    other_dir = list({0, 1, 2} - {axial_dir, radial_dir})[0]

    path = os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"")
    cells, T, U, rho, V, Nc = read_case(case_name, path)
    with open(os.path.join(f"{path}", f"{case_name}.dat"), 'w') as f:
        print(os.path.join(f"Writing file: {path}", f"{case_name}.dat"))
        f.write("\t\tX\t\t\t\t\tY\t\t\t\t\tZ\t\t\t\t\tV\t\t\t\t\tUx\t\t\t\t\tUy\t\t\t\t\tUz\t\t\t\t\tT\t\t\t\t\trho\n")
        for i in range(Nc):
            f.write(f"{cells[i, other_dir]:.12e}\t"  # TODO: insert a way to select the first and secondary direction
                    f"{cells[i, radial_dir]:.12e}\t"
                    f"{cells[i, axial_dir]:.12e}\t"
                    f"{V[i]:.12e}\t"
                    f"{U[i, 2]:.12e}\t"
                    f"{U[i, 0]:.12e}\t"
                    f"{U[i, 1]:.12e}\t"
                    f"{T[i]:.12e}\t"
                    f"{rho[i]:.12e}\n")


def get_boundary_data(case_name: str, boundaries: list):
    properties = ['U', 'T', 'rho', 'C']
    path = os.path.join(f"{os.getcwd()}", f"data", f"{case_name}")
    last_time = get_last_time(os.path.join(f"{path}", f"case"))
    mesh = Ofpp_patched.FoamMesh(os.path.join(f"{path}", f"case"))
    boundary_data = {}
    for boundary_name in boundaries:
        boundary_data[boundary_name] = {}       #creates empty dictionary for each boundary_name within boundary_data dict
        for property_ in properties:
            boundary_data_temp = Ofpp_patched.parse_boundary_field(os.path.join(f"{path}", f"case", f"{last_time}", f"{property_}"))
            if boundary_data_temp[bytes(boundary_name, 'utf-8')] != {}:
                boundary_data[boundary_name][property_] = boundary_data_temp[bytes(boundary_name, 'utf-8')][b'value']
                if type(boundary_data[boundary_name][property_]).__name__ != 'ndarray':
                    boundary_data[boundary_name][property_] = np.array([boundary_data[boundary_name][property_] for _ in range(mesh.boundary[bytes(boundary_name, 'utf-8')].num)])
    return boundary_data


if __name__ == '__main__':
    caseName = "CH4Piloted"
    # build_dat(caseName)
    print(get_boundary_data(caseName, ['fuelinlet', 'airinlet', 'pilotinlet', 'leftside']))
