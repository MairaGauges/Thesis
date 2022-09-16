import matplotlib.pyplot as plt
import numpy as np
import os

from DATScraper import scrapeDAT
from OFScraper import read_case


def initialize(case_name):
    print("Starting initialization...")

    path = os.path.join(os.getcwd(), "data", case_name)

    if os.path.isdir(os.path.join(path, "case")):
        print("OpenFOAM case found. Loading data directly from it...")

        if os.path.isdir(os.path.join(path, 'cache')):
            print("Cache found, loading it...")
            y = np.load(os.path.join(path, 'cache', 'Y.npy'))
            z = np.load(os.path.join(path, 'cache', 'Z.npy'))
            V = np.load(os.path.join(path, 'cache', 'V.npy'))
            vx = np.load(os.path.join(path, 'cache', 'Ux.npy'))
            vy = np.load(os.path.join(path, 'cache', 'Uy.npy'))
            vz = np.load(os.path.join(path, 'cache', 'Uz.npy'))
            T = np.load(os.path.join(path, 'cache', 'T.npy'))
            rho = np.load(os.path.join(path, 'cache', 'rho.npy'))

            Ny = y.shape[0]
            Nz = y.shape[1]

            return Ny, Nz, y, z, V, vx, vy, vz, T, rho

        cells, T_of, U_of, rho_of, V_of, Nc = read_case(case_name, path)
        if os.path.isfile(os.path.join(path, "CRNB_input.dic")):
            radial_dir = -1
            axial_dir = -1
            directions = {'x': 0, 'y': 1, 'z': 2}
            with open(os.path.join(path, "CRNB_input.dic"), 'r') as f:
                lines = f.readlines()
            command = ""
            for c in range(len(lines)):
                lines[c] = lines[c].rstrip('\n').replace(' ', '').replace('\t', '')
                if command == "":
                    if lines[c] == '':
                        continue
                    if lines[c][0] != '@':
                        raise ValueError(f"Command (starting with '@') expected in {os.path.join(path, 'CRNB_input.dic')} at line {c}")
                    command = lines[c][1:]
                    print(f"Found command @{command}")
                elif lines[c] == '//':
                    print(f"Ended command @{command}")
                    command = ""
                elif command == 'Geometry':
                    if lines[c].split(':')[0] == 'radial_dir':
                        radial_dir = directions[lines[c].split(':')[1]]
                    if lines[c].split(':')[0] == 'axial_dir':
                        axial_dir = directions[lines[c].split(':')[1]]
            if radial_dir == -1 or axial_dir == -1:
                radial_dir = int(input("Which is the radial direction? (Write 0 for x, 1 for y, 2 for z) "))
                axial_dir = int(input("Which is the axial direction? (Write 0 for x, 1 for y, 2 for z) "))
        else:
            radial_dir = int(input("Which is the radial direction? (Write 0 for x, 1 for y, 2 for z) "))
            axial_dir = int(input("Which is the axial direction? (Write 0 for x, 1 for y, 2 for z) "))

        print("WARNING - From now on, the radial direction will be referred as 'y' and the axial direction will be referred as 'z'")

        # Get dimensions
        print("Initializing grid...")
        Ny0 = int(input("Enter original Ny: "))
        Nz0 = int(input("Enter original Nz: "))
        Ny = 0
        Nz = 0

        prec = 12
        cells_norm = cells.copy()
        y_temp = []
        z_temp = []
        for i in range(len(cells_norm[0, :])):
            cells_norm[:, i] -= min(cells_norm[:, i])
            cells_norm[:, i] /= max(cells_norm[:, i])
        while prec > 3:
            print(f"Precision = {prec}")
            Ny = 0
            Nz = 0
            y_temp = []
            for yf in cells_norm[:, radial_dir]:
                if not any(y_temp == round(yf, prec)):
                    Ny += 1
                    y_temp.append(round(yf, prec))
            z_temp = []
            for zf in cells_norm[:, axial_dir]:
                if not any(z_temp == round(zf, prec)):
                    Nz += 1
                    z_temp.append(round(zf, prec))
            print(f" Found {Ny = }, {Nz = }")
            if Ny == Ny0 and Nz == Nz0:
                break
            else:
                print(" Reducing precision and retrying...")
                prec -= 1
        if prec <= 3:
            raise RuntimeError("Minimum precision reached. Check if the grid is structured.")

        # Initialize matrices
        y = np.empty((Ny, Nz), dtype=float)
        z = np.empty((Ny, Nz), dtype=float)
        V = np.empty((Ny, Nz), dtype=float)
        vx = np.empty((Ny, Nz), dtype=float)
        vy = np.empty((Ny, Nz), dtype=float)
        vz = np.empty((Ny, Nz), dtype=float)
        T = np.empty((Ny, Nz), dtype=float)
        rho = np.empty((Ny, Nz), dtype=float)

        temp = [0, 1, 2]
        temp.remove(radial_dir)
        temp.remove(axial_dir)
        third_dir = temp[0]
        del temp

        for k in range(len(cells[:, 0])):
            i = y_temp.index(round(cells_norm[k, radial_dir], prec))
            j = z_temp.index(round(cells_norm[k, axial_dir], prec))
            y[i, j] = cells[k, radial_dir]
            z[i, j] = cells[k, axial_dir]
            V[i, j] = V_of[k]
            vx[i, j] = U_of[k, third_dir]
            vy[i, j] = U_of[k, radial_dir]
            vz[i, j] = U_of[k, axial_dir]
            T[i, j] = T_of[k]
            rho[i, j] = rho_of[k]

        if input("Save cache? (y/n) ") == 'y':
            if not os.path.isdir(os.path.join(path, 'cache')):
                os.mkdir(os.path.join(path, 'cache'))
            np.save(os.path.join(path, 'cache', 'Y.npy'), y)
            np.save(os.path.join(path, 'cache', 'Z.npy'), z)
            np.save(os.path.join(path, 'cache', 'V.npy'), V)
            np.save(os.path.join(path, 'cache', 'Ux.npy'), vx)
            np.save(os.path.join(path, 'cache', 'Uy.npy'), vy)
            np.save(os.path.join(path, 'cache', 'Uz.npy'), vz)
            np.save(os.path.join(path, 'cache', 'T.npy'), T)
            np.save(os.path.join(path, 'cache', 'rho.npy'), rho)

    elif os.path.isfile(os.path.join(path, f"{case_name}.dat")):
        # This is somewhat outdated but still useful, it is used to read a .dat file containing all the information about the case (theoretically to use the program with CFD simulations made with
        # tools other than OpenFOAM), however since it hasn't been used for a long time there could be some errors
        print("DAT file found. Loading data...")
        dataSets, dataNames = scrapeDAT(case_name)

        # Get dimensions
        Ny = 0
        Nz = 0

        prec = 9
        # Get Ny and Nz
        #  y and z are useless arrays used only to store whenever a cell has already been counted
        y = []
        for yf in dataSets["Y"]:
            if not any(y == round(yf, prec)):
                Ny += 1
                y.append(round(yf, prec))
        z = []
        for zf in dataSets["Z"]:
            if not any(z == round(zf, prec)):
                Nz += 1
                z.append(round(zf, prec))

        # Initialize matrices
        y = np.empty((Ny, Nz), dtype=float)
        z = np.empty((Ny, Nz), dtype=float)
        V = np.empty((Ny, Nz), dtype=float)
        vx = np.empty((Ny, Nz), dtype=float)
        vy = np.empty((Ny, Nz), dtype=float)
        vz = np.empty((Ny, Nz), dtype=float)
        T = np.empty((Ny, Nz), dtype=float)
        rho = np.empty((Ny, Nz), dtype=float)

        print("Filling the matrices...")

        # Fill matrices with data
        j = 0
        k = 0
        for i in range(len(dataSets["X"])):
            y[j, k] = dataSets["Y"][i]
            z[j, k] = dataSets["Z"][i]
            V[j, k] = dataSets["V"][i]
            vx[j, k] = dataSets["Ux"][i]
            vy[j, k] = dataSets["Uy"][i]
            vz[j, k] = dataSets["Uz"][i]
            T[j, k] = dataSets["T"][i]
            rho[j, k] = dataSets["rho"][i]

            j += 1
            if j >= Ny:
                j = 0
                k += 1
    else:
        raise RuntimeError(f"Neither an OpenFOAM case or a DAT file have been found in {path}")

    print("Data initialized.")

    return Ny, Nz, y, z, V, vx, vy, vz, T, rho


if __name__ == '__main__':
    # Load data
    caseName = "Finale"
    Ny, Nz, y, z, V, vx, vy, vz, T, rho = initialize(caseName)
    print(f"Ny = {Ny}\n"
          f"Nz = {Nz}\n"
          f"{y.shape = }")

    print("Plotting...")

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(131, aspect='equal')
    cs = ax.pcolormesh(y, z, np.sqrt(np.square(vy) + np.square(vz)), cmap='jet', shading='auto')
    ax.set_title(r'Velocity [m/s]')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_xlim([y[0, 0], y[-1, -1]])
    ax.set_ylim([z[0, 0], z[-1, -1]])
    fig.colorbar(cs)
    ax.xaxis.set_ticks([0, y[-1, -1]])

    ax = fig.add_subplot(132, aspect='equal')
    cs = ax.pcolormesh(y, z, T, cmap='jet', shading='auto')
    ax.set_title('Temperature [K]')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    fig.colorbar(cs)
    ax.xaxis.set_ticks([0, y[-1, -1]])

    ax = fig.add_subplot(133, aspect='equal')
    cs = ax.pcolormesh(y, z, rho, cmap='jet', shading='auto')
    ax.set_title('Density [kg/m3]')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.xaxis.set_ticks([0, y[-1, -1]])
    fig.colorbar(cs)
    fig.tight_layout()

    fig = plt.figure()

    ax = fig.add_subplot(131, aspect='equal')
    cs = ax.pcolormesh(y, z, vx, cmap='jet', shading='auto')
    ax.set_title('vx [m/s]')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    fig.colorbar(cs)

    ax = fig.add_subplot(132, aspect='equal')
    cs = ax.pcolormesh(y, z, vy, cmap='jet', shading='auto')
    ax.set_title('vy [m/s]')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    fig.colorbar(cs)

    ax = fig.add_subplot(133, aspect='equal')
    cs = ax.pcolormesh(y, z, vz, cmap='jet', shading='auto')
    ax.set_title('vz [m/s]')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    fig.colorbar(cs)
    fig.tight_layout()

    plt.show()
