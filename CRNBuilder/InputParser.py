from math import prod
import os


def parse_input(filename):
    inlets = []
    outlets = []
    y_lim = [0, 0]
    z_lim = [0, 0]
    angle = 0.
    pressure = 0.
    radial_dir = -1
    axial_dir = -1
    T_threshold = 0.
    directions = {'x': 0, 'y': 1, 'z': 2}
    all_set = [False for _ in range(6)]
    with open(filename, 'r') as f:
        lines = f.readlines()
    command = ""
    for c in range(len(lines)):
        lines[c] = lines[c].rstrip('\n').replace(' ', '').replace('\t', '')
        if command == "":
            if lines[c] == '':
                continue
            if lines[c][0] != '@':
                raise ValueError(f"Command (starting with '@') expected in {filename} at line {c}")
            command = lines[c][1:]
            print(f"Found command @{command}")
        elif lines[c] == '//':
            print(f"Ended command @{command}")
            command = ""
        elif command == 'Inlets':
            all_set[0] = True
            inlet_name = lines[c].split(':')[0]
            inlet_data = lines[c].split(':')[1].split(',')
            inlets.append([inlet_name, inlet_data])
            if len(inlet_data) != 3:
                raise RuntimeError("Cannot scrape inlet data, expected arguments number is 3")
            if inlet_data[0] == 'south':
                inlet_data[0] = 0
            elif inlet_data[0] == 'west':
                inlet_data[0] = 1
            elif inlet_data[0] == 'north':
                inlet_data[0] = 2
            elif inlet_data[0] == 'east':
                inlet_data[0] = 3
            else:
                raise RuntimeError(f"{inlet_data[0]} is not a valid inlet argument. Valid arguments are 'south', 'west', 'north' and 'east'")
            inlet_data[1] = float(inlet_data[1])
            inlet_data[2] = float(inlet_data[2])
            print(f"Found inlet named {inlet_name}")
        elif command == 'Outlets':
            all_set[1] = True
            outlet_name = lines[c].split(':')[0]
            outlet_data = lines[c].split(':')[1].split(',')
            outlets.append([outlet_name, outlet_data])
            if len(outlet_data) != 3:
                raise RuntimeError("Cannot scrape outlet data, expected arguments number is 3")
            if outlet_data[0] == 'south':
                outlet_data[0] = 0
            elif outlet_data[0] == 'west':
                outlet_data[0] = 1
            elif outlet_data[0] == 'north':
                outlet_data[0] = 2
            elif outlet_data[0] == 'east':
                outlet_data[0] = 3
            else:
                raise RuntimeError(f"{outlet_data[0]} is not a valid inlet argument. Valid arguments are 'south', 'west', 'north' and 'east'")
            outlet_data[1] = float(outlet_data[1])
            outlet_data[2] = float(outlet_data[2])
            print(f"Found outlet named {outlet_name}")
        elif command == 'Dimensions':
            all_set[2] = True
            if lines[c].split(':')[0] == 'min_y':
                y_lim[0] = float(lines[c].split(':')[1])
            if lines[c].split(':')[0] == 'max_y':
                y_lim[1] = float(lines[c].split(':')[1])
            if lines[c].split(':')[0] == 'min_z':
                z_lim[0] = float(lines[c].split(':')[1])
            if lines[c].split(':')[0] == 'max_z':
                z_lim[1] = float(lines[c].split(':')[1])
            if lines[c].split(':')[0] == 'angle':
                angle = float(lines[c].split(':')[1])
        elif command == 'OperativeConditions':
            all_set[3] = True
            if lines[c].split(':')[0] == 'pressure':
                pressure = float(lines[c].split(':')[1])
        elif command == 'Geometry':
            all_set[4] = True
            if lines[c].split(':')[0] == 'radial_dir':
                radial_dir = directions[lines[c].split(':')[1]]
            if lines[c].split(':')[0] == 'axial_dir':
                axial_dir = directions[lines[c].split(':')[1]]
        elif command == 'Parameters':
            all_set[5] = True
            if lines[c].split(':')[0] == 'T_threshold':
                T_threshold = float(lines[c].split(':')[1])
    if prod(all_set) == 0:
        raise RuntimeError("Not all the settings have been defined in CRNB_input.dic, check there for errors")
    print(f"{inlets = }\n"
          f"{outlets = }\n"
          f"{y_lim = }\n"
          f"{z_lim = }\n"
          f"{angle = }\n"
          f"{pressure = }\n"
          f"{radial_dir = }\n"
          f"{axial_dir = }\n"
          f"{T_threshold = }\n")
    return inlets, outlets, y_lim, z_lim, angle, pressure, radial_dir, axial_dir, T_threshold


def main():
    case_name = "Finale"
    parse_input(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"CRNB_input.dic"))


if __name__ == '__main__':
    main()
