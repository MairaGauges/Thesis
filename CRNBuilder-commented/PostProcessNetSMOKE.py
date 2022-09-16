import os
import subprocess

import matplotlib.pyplot as plt

import Clustering

from Initializer import initialize
from EddyDetection import detect_eddy


def utility(n):
    out_string = post_process_NetSMOKE()
    with open(os.path.join(f"{os.getcwd()}", 'data', f'{case_name}', 'cache', f'output_{n}cl.txt'), 'w') as f:
        f.write(out_string)
    if not os.path.isfile(os.path.join(f"{os.getcwd()}", 'data', f'{case_name}', 'cache', f'output.txt')):
        with open(os.path.join(f"{os.getcwd()}", 'data', f'{case_name}', 'cache', f'output.txt'), 'w') as f:
            for _ in range(10):
                f.write("\n")
    with open(os.path.join(f"{os.getcwd()}", 'data', f'{case_name}', 'cache', f'output.txt'), 'r') as f0:
        with open(os.path.join(f"{os.getcwd()}", 'data', f'{case_name}', 'cache', f'output_new.txt'), 'w') as f1:
            lines = f0.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].rstrip('\n')
            new_vals = out_string.split('\n')
            new_vals[-1] = (f"{n}")
            for i in range(len(new_vals)):
                lines[i] += f'{new_vals[i]},\n'
                f1.write(lines[i])
    os.remove(os.path.join(f"{os.getcwd()}", 'data', f'{case_name}', 'cache', f'output.txt'))
    os.rename(os.path.join(f"{os.getcwd()}", 'data', f'{case_name}', 'cache', f'output_new.txt'), os.path.join(f"{os.getcwd()}", 'data', f'{case_name}', 'cache', f'output.txt'))


def post_process_NetSMOKE():
    species_to_get = ['CH4', 'H2', 'O2', 'N2', 'H2', 'H2O', 'CO2', 'CO', 'O', 'OH', 'CH2O']

    in_streams = []
    out_streams = []

    with open(os.path.join(os.getcwd(), f"NetSMOKE", f"ReactorNetwork_graph", f"NetworkMap_graphvizDot.dot")) as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip('\n')
        if line[0:8] == "INPUT ->":
            print(f"Found input\n   line: {line}")
            for j in range(len(line) - 8):
                c = j
                if line[c:c + 8] == 'label ="':
                    numb_string = ""
                    c += 8
                    while line[c] != "," and line[c] != '"':
                        numb_string += line[c]
                        c += 1
                    print(f"   stream id = {int(numb_string)}")
                    in_streams.append(int(numb_string))
                    break
            continue
        for j in range(len(line) - 9):
            if line[j:j + 9] == "-> OUTPUT":
                print(f"Found output\n   line: {line}")
                for j in range(len(line) - 8):
                    c = j
                    if line[c:c + 8] == 'label ="':
                        numb_string = ""
                        c += 8
                        while line[c] != "," and line[c] != '"':
                            numb_string += line[c]
                            c += 1
                        print(f"   stream id = {int(numb_string)}")
                        out_streams.append(int(numb_string))
                        break

    if not os.path.isdir(os.path.join(os.getcwd(), 'NetSMOKE', 'output')):
        raise RuntimeError(f"Cannot find {os.path.join(os.getcwd(), 'NetSMOKE', 'output')}, run NetSMOKE before trying to post-process.")
    if not os.path.isfile(os.path.join(os.getcwd(), 'NetSMOKE', 'output', 'StreamSummary.out')):
        raise RuntimeError(f"Cannot find {os.path.join(os.getcwd(), 'NetSMOKE', 'output', 'StreamSummary.out')}, run NetSMOKE before trying to post-process.")

    with open(os.path.join(os.getcwd(), 'NetSMOKE', 'output', 'StreamSummary.out'), 'r') as stream_file:
        lines = stream_file.readlines()

    headers = (lines.pop(0)).split(')')  # Remove header line and get the column index to analyze
    species_to_get_index = {}
    # species_to_get_index = [0 for _ in range(len(species_to_get))]
    for i in range(len(headers)):
        headers[i] = headers[i].lstrip(' ').split('(')[0]
        for species in species_to_get:
            if headers[i] == species + '_w':
                species_to_get_index[species] = i

    inlet_massflow = 0.
    # species_inlet_massflow = np.zeros(len(species_to_get_index), dtype=float)
    species_inlet_massflow = {}
    outlet_massflow = 0.
    # species_massflow = np.zeros(len(species_to_get_index), dtype=float)
    species_massflow = {}
    for k in species_to_get_index:
        species_inlet_massflow[k] = 0.
        species_massflow[k] = 0.
    for line in lines:
        stream_id = int(line.split()[0])
        if stream_id in in_streams:
            print(f"Reading stream n° {stream_id}")
            inlet_massflow += float(line.split()[5])
            for k in species_to_get_index:
                species_inlet_massflow[k] += float(line.split()[species_to_get_index[k]]) * float(line.split()[5])
        if stream_id in out_streams:
            print(f"Reading stream n° {stream_id}")
            outlet_massflow += float(line.split()[5])
            for k in species_to_get_index:
                species_massflow[k] += float(line.split()[species_to_get_index[k]]) * float(line.split()[5])

    out_string = f"\nInlet mass flow = {inlet_massflow:.5e} kg/s\n"
    for k in species_massflow:
        out_string += f"  {k} mass flow = {species_inlet_massflow[k]:.5e} kg/s\n"
    out_string += f"Mass fractions:\n"
    for k in species_massflow:
        out_string += f"  {k} mass fraction = {species_inlet_massflow[k] / inlet_massflow:.5e}\n"
    out_string += f"\nOutlet mass flow = {outlet_massflow:.5e} kg/s\n"
    for k in species_massflow:
        out_string += f"  {k} mass flow = {species_massflow[k]:.5e} kg/s\n"
    # for i in range(len(species_massflow)):
    #     out_string += f"{species_massflow[i]:.5e}\n"
    out_string += f"Mass fractions:\n"
    for k in species_massflow:
        out_string += f"  {k} mass fraction = {species_massflow[k] / outlet_massflow:.5e}\n"

    print(out_string)

    out_string = ""
    print("")
    out_string += f"{1 - species_massflow['CH4'] / species_inlet_massflow['CH4']}\n"
    out_string += f"{1 - species_massflow['O2'] / species_inlet_massflow['O2']}\n"
    out_string += f"{species_massflow['CO2'] / (species_inlet_massflow['CH4'] - species_massflow['CH4']) * 0.016 / 0.044}\n"
    out_string += f"{species_massflow['CO'] / (species_inlet_massflow['CH4'] - species_massflow['CH4']) * 0.016 / 0.028}\n"
    print(out_string)

    return out_string


if __name__ == '__main__':

    load_cache = True
    save_cache = True

    # Load data
    case_name = "CH4Piloted"
    Ny, Nz, y, z, V, vx, vy, vz, T, rho = initialize(case_name)

    # Get eddy data
    # Give id 0 to cells not in an eddy, 1 to cells in the firsts eddy, 2 to the cells in the second eddy, etc...
    eddy_id = detect_eddy(case_name, Ny, Nz, y, z, vy, vz, load_cached=True)
    for i in range(3, 9):
        cluster_id_linkage = Clustering.cluster_data_linkage(case_name, Ny, Nz, y, z, T, i, eddy_id, threshold=750, load_cached=load_cache, save_cache=save_cache, to_cut=False, rework=False)

        # Plot of raw data and clusters
        print("Plotting results...")
        fig = plt.figure(figsize=(20, 15))

        # ax = fig.add_subplot(131, aspect='equal')
        # cs = ax.pcolormesh(y, z, cluster_id_kmeans, cmap='jet', shading='auto')
        # ax.set_title('Cluster ID - kmeans2')
        # ax.set_xlabel('y')
        # ax.set_ylabel('z')
        # fig.colorbar(cs)

        ax = fig.add_subplot(121, aspect='equal')
        cs = ax.pcolormesh(y, z, cluster_id_linkage, cmap='jet', shading='auto')
        ax.set_title('Cluster ID - linkage')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        fig.colorbar(cs)

        ax = fig.add_subplot(122, aspect='equal')
        cs = ax.pcolormesh(y, z, T, cmap='jet', shading='auto')
        ax.set_title('Temperature')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        fig.colorbar(cs)

        if not os.path.isdir(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f"graphs")):
            os.mkdir(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f"graphs"))
        fig.savefig(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f"graphs", f"clusters_{i}cl.pdf"))
        subprocess.call(os.path.join(os.getcwd(), f'run.sh {i}'), shell=True)
        utility(i)
#    post_process_NetSMOKE()
