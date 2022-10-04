from __future__ import annotations  # This is used to enable type hinting in python versions from 3.7 to 3.9 (it is built-in in 3.10 and above)

from math import pi, sqrt, prod
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import networkx as nx

from Clustering import cluster_data_linkage
from ClusterAnalysis import get_pxr_score
from EddyDetection import detect_eddy
from Initializer import initialize
from InputParser import parse_input
from OFScraper import get_boundary_data
from readVTK import initializeFromVTK


class ReactorNetworkObject:

    def __init__(self, id_: int, inlet: list, outlet: list):
        self.id_ = id_
        self.inlet = inlet
        self.outlet = outlet

    def generate_out_string(self):
        """
        Every class inheriting from the class ReactorNetworkObject should have a function to generate the string to be put into InputRN.dic, however since it differs
        based on the object type (e.g. Reactors, Mixers, etc...) it cannot be implemented directly here. This function is then a placeholder that throws an error if
        the user forgets to implement it.
        """
        raise NotImplementedError(f"The function generate_out_string has not been implemented yet for the class {type(self).__name__}.")


class Reactor(ReactorNetworkObject):

    def __init__(self, id_: int, inlet: list, outlet: list, T: float):
        super().__init__(id_, inlet, outlet)
        self.T = T
        self.volume_string = ""
        print(f"Initialized reactor of type {type(self).__name__} with {id_ = } and {T = :.2f} K")

    def generate_out_string(self) -> str:
        if type(self).__name__ == "Reactor":
            raise RuntimeError("generate_out_string can not be called on a general reactor. Search for a bug in CRNBuilder.py")
        assert self.volume_string != ""  # Check whether the volume string has been set, if not throw an error
        if len(self.inlet) > 1:
            raise ValueError("Trying to generate a reactor with more than one inlet stream. Check CRNBuilder.py, class Reactor.")
        if len(self.outlet) > 1:
            raise ValueError("Trying to generate a reactor with more than one outlet stream. Check CRNBuilder.py, class Reactor.")
        for crn_object in (self.inlet + self.outlet):
            if type(crn_object).__name__ != "Stream" and type(crn_object).__name__ != "CRNInlet":
                raise ValueError(f"{type(self).__name__} {self.id_} is trying to write as input or output stream an object that is not a Stream or CRNInlet.\n"
                                 f"The object is of type {type(crn_object).__name__} and id {crn_object.id_}")
        return f"@Reactor       R{self.id_}\n" \
               f"Type           {type(self).__name__}\n" \
               f"Phase          Gas\n" \
               f"Energy         Isothermal\n" \
               f"Temperature    {self.T:.2f} K\n" \
               f"{self.volume_string}\n" \
               f"Inlet_stream   {','.join(str(inlet.id_) for inlet in self.inlet)}\n" \
               f"Outlet_stream  {','.join(str(outlet.id_) for outlet in self.outlet)}\n" \
               f"//\n"


class PFR(Reactor):

    def __init__(self, id_: int, inlet: list, outlet: list, T: float, L: float, D: float):
        super().__init__(id_, inlet, outlet, T)
        self.volume_string = f"Length         {L:.5e} m\n" \
                             f"Diameter       {D:.5e} m"


class PSR(Reactor):

    def __init__(self, id_: int, inlet: list, outlet: list, T: float, V: float):
        super().__init__(id_, inlet, outlet, T)
        self.volume_string = f"Volume         {V:.5e} m3"


class Mixer(ReactorNetworkObject):

    def __init__(self, id_: int, inlet: list, outlet: list, T: float):
        super().__init__(id_, inlet, outlet)
        self.T = T

    def generate_out_string(self):
        if len(self.outlet) > 1:
            raise ValueError("Trying to generate a mixer with more than one outlet stream. Check CRNBuilder.py, class Mixer.")
        for crn_object in (self.inlet + self.outlet):
            if type(crn_object).__name__ != "Stream" and type(crn_object).__name__ != "CRNInlet":
                raise ValueError(f"{type(self).__name__} {self.id_} is trying to write as input or output stream an object that is not a Stream or CRNInlet.\n"
                                 f"The object is of type {type(crn_object).__name__} and id {crn_object.id_}")
        return f"@Mixer         M{self.id_}\n" \
               f"Energy         Isothermal\n" \
               f"Temperature    {self.T:.2f} K\n" \
               f"Inlet_stream   {','.join(str(inlet.id_) for inlet in self.inlet)}\n" \
               f"Outlet_stream  {','.join(str(outlet.id_) for outlet in self.outlet)}\n" \
               f"//\n"


class Splitter(ReactorNetworkObject):

    def __init__(self, id_: int, inlet: list, outlet: list, splitting_ratio: list[float]):
        super().__init__(id_, inlet, outlet)
        self.splitting_ratio = splitting_ratio

    def generate_out_string(self) -> str:
        for crn_object in (self.inlet + self.outlet):
            if type(crn_object).__name__ != "Stream" and type(crn_object).__name__ != "CRNInlet":
                raise ValueError(f"{type(self).__name__} {self.id_} is trying to write as input or output stream an object that is not a Stream or CRNInlet.\n"
                                 f"The object is of type {type(crn_object).__name__} and id {crn_object.id_}")
        outlet_streams_str = "Outlet_stream  "
        if len(self.inlet) > 1:
            raise ValueError("Trying to generate a splitter with more than one inlet stream. Check CRNBuilder.py, class Splitter.")
        if sum(self.splitting_ratio) != 1.:
            print(f"splitting_ratio of Splitter {self.id_} doesn't sum to 1, correcting the last value from {self.splitting_ratio[-1]} to {1 - sum(self.splitting_ratio[:-1])}")
            self.splitting_ratio[-1] = 1 - sum(self.splitting_ratio[:-1])
        for i in range(len(self.outlet)):
            outlet_streams_str += f"{self.outlet[i].id_},{self.splitting_ratio[i]:.9f}"
            if i < len(self.outlet) - 1:
                outlet_streams_str += f"\n               "
        return f"@Splitter      S{self.id_}\n" \
               f"Inlet_stream   {','.join(str(inlet.id_) for inlet in self.inlet)}\n" \
               f"{outlet_streams_str};\n" \
               f"//\n"


class CRNInlet(ReactorNetworkObject):

    def __init__(self, id_: int, inlet: list, outlet: list, T: float, mass_flowrate: list):
        super().__init__(id_, inlet, outlet)
        if len(self.inlet) > 0:
            raise RuntimeError(f"Trying to set an inlet to CRNInlet {self.id_}")
        self.T = T
        self.mass_flowrate = mass_flowrate
        print(f"Initialized inlet with {id_ = :d}, {T = :.2f} K and total mass_flowrate = {sum(self.mass_flowrate[k] for k in self.mass_flowrate):.5e} kg/s\n"
              f"  f{mass_flowrate = }")

    def generate_out_string(self) -> str:
        print("----WARNING----\n  MassFraction for InletStream has not been implemented yet. Setting MassFraction to dry air...")
        if len(self.inlet) > 0:
            raise ValueError("Trying to generate a CRNInlet with an inlet stream. Check CRNBuilder.py, class CRNInlet.")
        mass_fraction_str = f"MassFraction   N2  0.765\n" \
                            f"               O2  0.235;\n"
        return f"@Stream        {self.id_}\n" \
               f"Phase          Gas\n" \
               f"MassFlowRate   {sum([self.mass_flowrate[k] for k in self.mass_flowrate]):.5e} kg/s\n" \
               f"Temperature    {self.T:.2f} K\n" \
               f"{mass_fraction_str}" \
               f"//\n"


class CRNOutlet(ReactorNetworkObject):

    def __init__(self, id_: int, inlet: list, outlet: list, mass_flowrate: list):
        super().__init__(id_, inlet, outlet)
        self.mass_flowrate = mass_flowrate
        if len(self.outlet) > 0:
            raise RuntimeError(f"Trying to set an outlet to CRNOutlet {self.id_}")
        print(f"Initialized outlet with {id_ = :d} and total mass_flowrate{sum(self.mass_flowrate[k ]for k in self.mass_flowrate):.5e} kg/s\n ")

    def generate_out_string(self) -> str:
        # Since CRNOutlet objects are not needed in InputCRN.dic, the output string is empty.
        return f""


class Stream(ReactorNetworkObject):

    def __init__(self, id_: int, inlet: list, outlet: list):
        super().__init__(id_, inlet, outlet)


class ChemicalReactorNetwork:

    def __init__(self,
                 Ny, Nz, n_clusters, cluster_id, pxr_score, T,
                 mass_flows, inlets, outlets, axial_n, y, z, my,
                 mz, V, vy, vz, boundary_data):
        self.inlet_id = np.zeros((Ny, Nz), dtype=int)
        self.outlet_id = np.zeros((Ny, Nz), dtype=int)
        self.reactors = []
        self.mixers = []
        self.splitters = []
        self.streams = []
        self.crn_inlets = []
        self.crn_outlets = []
        # mass_flows acts as a weighted adjacency matrix
        self.mass_flows = mass_flows

        self.generate_inlets(Ny, Nz, y, z, T, my, mz, inlets, cluster_id, boundary_data)
        self.generate_outlets(Ny, Nz, y, z, outlets, my, mz, cluster_id, n_clusters)
        self.generate_reactors(Ny, Nz, n_clusters, cluster_id, pxr_score, T, axial_n, V, vy, vz, y, z)
        self.generate_splitters()
        self.generate_mixers()
        # Merge all lists in a single object list
        self.crn_objects =self.crn_outlets + self.reactors  + self.mixers + self.splitters + self.crn_inlets

        self.generate_streams()

    # TODO: Entrambe le funzioni generate_inlet e generate_outlet dovrebbero essere sostituite da un'unica funzione generate_boundary che sommi tutto il mass flow e decida se mettere
    #  un inlet o un outlet in base al segno
    def generate_inlets(self, Ny, Nz, y, z, T, my, mz, inlets, cluster_id, boundary_data):
        for n in range(len(inlets)):
            inlet = inlets[n][0]
            print(f"Processing inlet {inlet}")
            T_mean = 0.
            counter = 0.
            mass_flowrate = {}
            # Read the minimum and maximum y/z from the inlets list
            lower_bound = inlets[n][1][1]
            upper_bound = inlets[n][1][2]
            N = -1
            base_ind = np.zeros(2, dtype=int)
            x = np.empty((Ny, Nz), dtype=float)
            mx = np.empty((Ny, Nz), dtype=float)
            mx_name = ''
            f = 1
            if inlets[n][1][0] == 0:
                N = Ny
                base_ind = np.array([1, 0])
                x = y
                mx = mz
                mx_name = 'mz'
            elif inlets[n][1][0] == 1:
                N = Nz
                base_ind = np.array([0, 1])
                x = z
                mx = my
                mx_name = 'my'
            elif inlets[n][1][0] == 2:
                N = Ny
                base_ind = np.array([1, -1])
                x = y
                mx = mz
                mx_name = 'mz'
                f *= -1
            elif inlets[n][1][0] == 3:
                N = Nz
                base_ind = np.array([-1, 1])
                x = z
                mx = my
                mx_name = 'my'
                f *= -1
            boundary_data_present = True
            # This try-except is used to check whether the required boundary data has been read from the OpenFOAM case
            try:
                properties = ['U', 'T', 'rho', 'C']
                for property_ in properties:
                    boundary_data[inlet][property_]
            except KeyError:
                boundary_data_present = False
            if not boundary_data_present:
                for i in range(N):
                    ind = tuple([base_ind[k] * i if base_ind[k] > 0 else base_ind[k] for k in range(len(base_ind))])
                    if not lower_bound <= x[ind] <= upper_bound:
                        continue
                    T_mean += T[ind]
                    if f * mx[ind] < 0:
                        raise RuntimeError(f"f * mx[ind] < 0, there must be a problem in the inlet reading. Check generate_inlet under ChemicalReactorNetwork in CRNBuilder.py")
                    if cluster_id[ind] not in mass_flowrate:
                        mass_flowrate[cluster_id[ind]] = 0
                    mass_flowrate[cluster_id[ind]] += f * max(mx[ind], 0.)
                    self.inlet_id[ind] = n + 1
                    counter += 1
            else:
                i0 = 0
                for i in range(N):
                    ind = tuple([base_ind[k] * i if base_ind[k] > 0 else base_ind[k] for k in range(len(base_ind))])
                    if x[ind] > lower_bound:
                        i0 = i
                        break
                for i in range(len(boundary_data[inlet]['U'])):
                    ind = tuple([base_ind[k] * i + i0 if base_ind[k] > 0 else base_ind[k] for k in range(len(base_ind))])
                    T_mean += boundary_data[inlet]['T'][i]
                    if f * boundary_data[inlet][mx_name][i] < 0:
                        print(f"WARNING: f * boundary_data[inlet]['{mx_name}'][i] < 0, there must be a problem in the inlet reading. Check generate_inlet under ChemicalReactorNetwork in "
                              f"CRNBuilder.py")
                    if cluster_id[ind] not in mass_flowrate:
                        mass_flowrate[cluster_id[ind]] = 0
                    mass_flowrate[cluster_id[ind]] += f * boundary_data[inlet][mx_name][i]
                    self.inlet_id[ind] = n + 1
                    counter += 1
            T_mean /= counter
            for k in mass_flowrate:
                if mass_flowrate[k] <= 0:
                    raise RuntimeError(f"mass_flowrate in {inlet} is <= 0")
            self.crn_inlets.append(CRNInlet(n, [], [], T_mean, mass_flowrate))
            print('here mass_flowrate')
            print(mass_flowrate)

	    
    def generate_outlets(self, Ny, Nz, y, z, outlets, my, mz,cluster_id,n_clusters):
        
        
        for n in range(len(outlets)):
            lower_bound = outlets[n][1][1]
            upper_bound = outlets[n][1][2]
            mass_flowrate={}
            for r in range(n_clusters):
                
                if outlets[n][1][0] == 0:
                    for i in range(Ny):
                        if not lower_bound <= y[i, 0] <= upper_bound or r!= cluster_id[i,0]:
                            continue
                        self.outlet_id[i, 0] = n + 1
                        if r not in mass_flowrate:
                            mass_flowrate[r]=0
                        mass_flowrate[r]-=mz[i,0]
                elif outlets[n][1][0] == 1:
                    for j in range(Nz):
                        if not lower_bound <= z[0, j] <= upper_bound or r!= cluster_id[0,j]:
                            continue
                        self.outlet_id[0, j] = n + 1
                        if r not in mass_flowrate:
                            mass_flowrate[r]=0
                        mass_flowrate[r]-=my[0,j]
                         
                elif outlets[n][1][0] == 2:
                    for i in range(Ny):
                        if not lower_bound <= y[i, -1] <= upper_bound or r!= cluster_id[i,-1]:
                            continue
                        self.outlet_id[i, -1] = n + 1
                        if r not in mass_flowrate:
                            mass_flowrate[r]=0
                        mass_flowrate[r]+=mz[i,-1]
                elif outlets[n][1][0] == 3:
                    for j in range(Nz):
                        if not lower_bound <= z[-1, j] <= upper_bound or r!= cluster_id[-1,j]:
                            continue
                        self.outlet_id[-1, j] = n + 1
                        if r not in mass_flowrate:
                            mass_flowrate[r]=0
                        mass_flowrate[r]+=mz[-1,j]

            self.crn_outlets.append(CRNOutlet(n, [], [], mass_flowrate))
            print('mass_flowrate outlet')
            print(mass_flowrate)

    def generate_reactors(self, Ny, Nz, n_clusters, cluster_id, pxr_score, T, axial_n, V, vy, vz, y, z):
        cluster_volumes = get_cluster_volume(Ny, Nz, V, n_clusters, cluster_id, axial_n)
        for n in range(n_clusters):
            # Get the Temperature of the reactor
            T_mean = 0.
            V_counter = 0.
            for i in range(Ny):
                for j in range(Nz):
                    if cluster_id[i, j] != n:
                        continue
                    T_mean += T[i, j] * V[i, j]
                    V_counter += V[i, j]
            T_mean /= V_counter
            # Set the volume and instantiate the object
            if pxr_score[n]:
                self.reactors.append(PSR(n, [], [], T_mean, cluster_volumes[n]))
            else:
                pfr_L = get_PFR_length(Ny, Nz, vy, vz, cluster_id, n, y, z)
                pfr_D = sqrt(cluster_volumes[n] / pfr_L * 4 / pi)
                self.reactors.append(PFR(n, [], [], T_mean, pfr_L, pfr_D))
            # Connect the reactor to the inlets and outlets of the system (if necessary)
            for i in range(Ny):
                for j in range(Nz):
                    if self.inlet_id[i, j] == 0 or n != cluster_id[i, j]:
                        continue
                    inlet_already_added = False
                    for reactor_inlet in self.reactors[-1].inlet:
                        if type(reactor_inlet).__name__ == 'CRNInlet' and reactor_inlet.id_ == self.inlet_id[i, j] - 1:
                            inlet_already_added = True
                            break
                    if not inlet_already_added:
                        print(f"Adding CRNInlet {self.crn_inlets[self.inlet_id[i, j] - 1].id_} as an inlet to {type(self.reactors[-1]).__name__} {self.reactors[-1].id_}")
                        self.reactors[-1].inlet.append(self.crn_inlets[self.inlet_id[i, j] - 1])
                        self.crn_inlets[self.inlet_id[i, j] - 1].outlet.append(self.reactors[-1])
            for i in range(Ny):
                for j in range(Nz):
                    if self.outlet_id[i, j] == 0 or n != cluster_id[i, j]:
                        continue
                    outlet_already_added = False
                    for reactor_outlet in self.reactors[-1].outlet:
                        if type(reactor_outlet).__name__ == 'CRNOutlet' and reactor_outlet.id_ == self.outlet_id[i, j] - 1:
                            outlet_already_added = True
                            break
                    if not outlet_already_added:
                        print(f"Adding CRNOutlet {self.crn_outlets[self.outlet_id[i, j] - 1].id_} as an outlet to {type(self.reactors[-1]).__name__} {self.reactors[-1].id_}")
                        self.reactors[-1].outlet.append(self.crn_outlets[self.outlet_id[i, j] - 1])
                        self.crn_outlets[self.outlet_id[i, j] - 1].inlet.append(self.reactors[-1])
        # Connect the reactors inside the CRN
        for i in range(self.mass_flows.shape[0]):
            for j in range(self.mass_flows.shape[1]):
                if self.mass_flows[i, j] != 0:
                    self.reactors[i].outlet.append(self.reactors[j])
                    self.reactors[j].inlet.append(self.reactors[i])
                    print(f"Added {type(self.reactors[j]).__name__} {self.reactors[j].id_} as outlet to {type(self.reactors[i]).__name__} {self.reactors[i].id_}")
        # Disconnect the reactors that are connected by a mass flow less than 10% of the max mass flow, also checks if they are connected to an output
        
        temp_mass_flows = self.mass_flows.copy()
        max_mass_flow = np.max(temp_mass_flows)
        while True:
            min_mass_flow = np.min(temp_mass_flows[temp_mass_flows > 0.])
            if min_mass_flow >= 0.1 * max_mass_flow:
                print(f"min_mass_flow >= 0.1* max_mass_flow\n  {min_mass_flow       = :.3e} kg/s\n  {0.1 * max_mass_flow = :.3e} kg/s")
                break
            for i in range(self.mass_flows.shape[0]):
                for j in range(self.mass_flows.shape[1]):
                    if self.mass_flows[i, j] == min_mass_flow:
                        self.reactors[i].outlet.remove(self.reactors[j])
                        self.reactors[j].inlet.remove(self.reactors[i])
                        temp_mass_flows[i, j] = 0.
                        if not self.connected_to_output(i) or not self.connected_to_input(j) or len(self.reactors[i].outlet) == 0 or len(self.reactors[j].inlet) == 0:
                            self.reactors[i].outlet.append(self.reactors[j])
                            self.reactors[j].inlet.append(self.reactors[i])
                        else:
                            print(f"Disconnected {type(self.reactors[j]).__name__} {self.reactors[j].id_} from {type(self.reactors[i]).__name__} {self.reactors[i].id_} "
                                  f"(mass flow = {self.mass_flows[i, j]:.3e} kg/s)")
        del temp_mass_flows

    def connected_to_input(self, i):
        # BFS algorithm to check if the reactor is connected to an inlet
        visited = [False] * len(self.reactors)
        queue = [i]
        visited[i] = True
        while queue:
            n = queue.pop(0)
            for inlet in self.reactors[n].inlet:
                if type(inlet).__name__ == 'CRNInlet':
                    return True
                if not visited[inlet.id_]:
                    queue.append(inlet.id_)
                    visited[inlet.id_] = True
        return False

    def connected_to_output(self, i):
        # BFS algorithm to check if the reactor is connected to an output
        visited = [False] * len(self.reactors)
        queue = [i]
        visited[i] = True
        while queue:
            n = queue.pop(0)
            for outlet in self.reactors[n].outlet:
                if type(outlet).__name__ == 'CRNOutlet':
                    return True
                if not visited[outlet.id_]:
                    queue.append(outlet.id_)
                    visited[outlet.id_] = True
        return False

    def generate_splitters(self):
        counter = 0
        for crn_object in self.reactors + self.crn_inlets:
            if len(crn_object.outlet) > 1:
                # WARNING: This function must be called before generate_mixers, or else the next line won't work
                splitting_ratio = []
                for outlet in crn_object.outlet:
                    if isinstance(crn_object, Reactor):
                        if isinstance(outlet, Reactor):
                            splitting_ratio.append(self.mass_flows[crn_object.id_, outlet.id_])
                        elif isinstance(outlet, CRNOutlet):
                            splitting_ratio.append(outlet.mass_flowrate[crn_object.id_])
                        else:
                            raise RuntimeError("The reactor outlet is neither a Reactor or a CRNOutlet, check generate_splitters() in object ChemicalReactorNetwork in CRNBuilder.py")
                    elif isinstance(crn_object, CRNInlet):
                        if isinstance(outlet, Reactor):
                            splitting_ratio.append(crn_object.mass_flowrate[outlet.id_])
                        else:
                            raise RuntimeError("The CRNInlet outlet is not a Reactor, check generate_splitters() in object ChemicalReactorNetwork in CRNBuilder.py")
                    else:
                        raise RuntimeError("Check generate_splitters() in object ChemicalReactorNetwork in CRNBuilder.py")
                normalize = True
                while normalize:
                    splitting_ratio /= sum(splitting_ratio)
                    normalize = False
                    for i in range(len(splitting_ratio)):
                        splitting_ratio[i] = round(splitting_ratio[i], 9)
                        if splitting_ratio[i] == 0:
                            splitting_ratio[i] += 1e-9
                            normalize = True
                self.splitters.append(Splitter(counter, [crn_object], [], splitting_ratio))
                self.splitters[-1].outlet = crn_object.outlet.copy()
                crn_object.outlet = [self.splitters[-1]]
                for outlet in self.splitters[-1].outlet:
                    outlet.inlet.remove(crn_object)
                    outlet.inlet.append(self.splitters[-1])
                print(f"Added splitter from {type(crn_object).__name__} {crn_object.id_} to {[f'{type(o).__name__} {o.id_}' for o in self.splitters[-1].outlet]} "
                      f"with splitting ratio {self.splitters[-1].splitting_ratio}.")
                counter += 1

    def generate_mixers(self):
        counter = 0
        for reactor in self.reactors:
            if len(reactor.inlet) > 1:
                self.mixers.append(Mixer(counter, [], [reactor], 300.))
                self.mixers[-1].inlet = reactor.inlet.copy()
                reactor.inlet = [self.mixers[-1]]
                for inlet in self.mixers[-1].inlet:
                    inlet.outlet.remove(reactor)
                    inlet.outlet.append(self.mixers[-1])
                print(f"Added mixer from {[f'{type(o).__name__} {o.id_}' for o in self.mixers[-1].inlet]} to {type(reactor).__name__} {reactor.id_}")
                counter += 1

    def generate_streams(self):
        counter = len(self.crn_inlets)

        list = self.crn_outlets
        for reactor in self.reactors:
            for inlet in reactor.inlet:
                if isinstance(inlet,Mixer):
                    list.append(inlet)
            list.append(reactor)
        for splitter in self.splitters:
            list.append(splitter)

        for crn_object in list:
            
            for inlet in crn_object.inlet.copy():
                if isinstance(inlet, CRNInlet):
                    continue
                elif not isinstance(inlet, Stream):
                    self.streams.append(Stream(counter, [inlet], [crn_object]))
                    crn_object.inlet.remove(inlet)
                    crn_object.inlet.append(self.streams[-1])
                    inlet.outlet.remove(crn_object)
                    inlet.outlet.append(self.streams[-1])
                    print(f"Added stream with id {self.streams[-1].id_} from {type(inlet).__name__} {inlet.id_} to {type(crn_object).__name__} {crn_object.id_}")
                    counter += 1
                else:
                    print(f"----WARNING----\n"
                          f"  There may be a problem, check generate_streams() in ChemicalReactorNetwork in CRNBuilder.py\n"
                          f"  The inlet to {type(crn_object).__name__} {crn_object.id_} is {type(inlet).__name__} {inlet.id_}")

    def get_PSRs(self):
        return [reactor for reactor in self.reactors if type(reactor).__name__ == "PSR"]

    def get_PFRs(self):
        return [reactor for reactor in self.reactors if type(reactor).__name__ == "PFR"]


def generate_input_dic(crn, path, pressure):
    general_options = f"@GeneralOptions\n" \
                      f"SystemPressure {pressure} atm\n" \
                      f"TearStream automatic\n" \
                      f"//\n"

    crn_object_data = ""
    for crn_object in crn.crn_objects:
        crn_object_data += crn_object.generate_out_string()

    if not os.path.isdir(os.path.join(f"{path}", f"out")):
        os.mkdir(os.path.join(f"{path}", f"out"))
    path = os.path.join(path, f"out")
    with open(os.path.join(f"{path}", f"inputRN.dic"), 'w') as f:
        f.write(f"{general_options}\n\n"
                f"{crn_object_data}\n\n\n")


def get_massflow(y, z, vy, vz, rho, axial_n, y_lim, z_lim, Ny, Nz, boundary_data, radial_dir, axial_dir, inlets):
    my = np.empty(y.shape, dtype=float)
    mz = np.empty(y.shape, dtype=float)
    Ay = np.empty(y.shape, dtype=float)
    Az = np.empty(y.shape, dtype=float)
    print(f"{axial_n = }")
    for i in range(Ny):
        for j in range(Nz):
            if i > 0:
                y1 = (y[i, j] + y[i - 1, j]) / 2
            else:
                y1 = y_lim[0]
            if i < Ny - 1:
                y2 = (y[i + 1, j] + y[i, j]) / 2
            else:
                y2 = y_lim[1]
            if j > 0:
                z1 = (z[i, j] + z[i, j - 1]) / 2
            else:
                z1 = z_lim[0]
            if j < Nz - 1:
                z2 = (z[i, j + 1] + z[i, j]) / 2
            else:
                z2 = z_lim[1]
            Ay[i, j] = 2 * pi / axial_n * y[i, j] * (z2 - z1)
            Az[i, j] = pi / axial_n * (y2 * y2 - y1 * y1)
            my[i, j] = rho[i, j] * vy[i, j] * Ay[i, j]
            mz[i, j] = rho[i, j] * vz[i, j] * Az[i, j]
    for n in range(len(inlets)):
        inlet = inlets[n][0]
        boundary_data[inlet]['Ay'] = np.zeros(boundary_data[inlet]['C'].shape[0], dtype=float)
        boundary_data[inlet]['Az'] = np.zeros(boundary_data[inlet]['C'].shape[0], dtype=float)
        boundary_data[inlet]['my'] = np.zeros(boundary_data[inlet]['C'].shape[0], dtype=float)
        boundary_data[inlet]['mz'] = np.zeros(boundary_data[inlet]['C'].shape[0], dtype=float)
        lower_bound = inlets[n][1][1]
        print(f"{inlet} : {lower_bound = }")
        N = -1
        base_ind = np.zeros(2, dtype=int)
        x = np.empty((Ny, Nz), dtype=float)
        f = 1
        if inlets[n][1][0] == 0:
            N = Ny
            base_ind = np.array([1, 0])
            x = y
        elif inlets[n][1][0] == 1:
            N = Nz
            base_ind = np.array([0, 1])
            x = z
        elif inlets[n][1][0] == 2:
            N = Ny
            base_ind = np.array([1, -1])
            x = y
            f *= -1
        elif inlets[n][1][0] == 3:
            N = Nz
            base_ind = np.array([-1, 1])
            x = z
            f *= -1
        i0 = 0
        for i in range(N):
            ind = tuple([base_ind[k] * i if base_ind[k] > 0 else base_ind[k] for k in range(len(base_ind))])
            if x[ind] > lower_bound:
                i0 = i
                break
        for i in range(len(boundary_data[inlet]['C'])):
            ind = tuple([base_ind[k] * i + i0 if base_ind[k] > 0 else base_ind[k] for k in range(len(base_ind))])
            if ind[0] > 0:
                y1 = (y[ind] + y[ind[0] - 1, ind[1]]) / 2
            else:
                y1 = y_lim[0]
            if ind[0] < Ny - 1:
                y2 = (y[ind[0] + 1, ind[1]] + y[ind]) / 2
            else:
                y2 = y_lim[1]
            if ind[1] > 0:
                z1 = (z[ind] + z[ind[0], ind[1] - 1]) / 2
            else:
                z1 = z_lim[0]
            if ind[1] < Nz - 1:
                z2 = (z[ind[0], ind[1] + 1] + z[ind]) / 2
            else:
                z2 = z_lim[1]
            # print(f"{inlet} : {y2 = }, {y1 = }, {y2 - y1 = }")
            boundary_data[inlet]['Ay'][i] = 2 * pi / axial_n * y[ind] * (z2 - z1)
            boundary_data[inlet]['Az'][i] = pi / axial_n * (y2 * y2 - y1 * y1)
            boundary_data[inlet]['my'][i] = boundary_data[inlet]['rho'][i] * boundary_data[inlet]['U'][i, radial_dir] * boundary_data[inlet]['Ay'][i]
            boundary_data[inlet]['mz'][i] = boundary_data[inlet]['rho'][i] * boundary_data[inlet]['U'][i, axial_dir] * boundary_data[inlet]['Az'][i]
    print(f'{np.sum(boundary_data["fuelinlet"]["Az"]) = }')
    print(f'{np.sum(boundary_data["airinlet"]["Az"]) = }')
    return my, mz, Ay, Az


def get_massflow_between_clusters(Ny, Nz, my, mz, n_clusters, cluster_id):
    mass_flows = np.zeros((n_clusters, n_clusters), dtype=float)
    for i in range(Ny):
        for j in range(Nz):
            if i > 0 and my[i, j] < 0 and cluster_id[i, j] != cluster_id[i - 1, j]:
                mass_flows[cluster_id[i, j], cluster_id[i - 1, j]] += abs(my[i, j])
            if i < Ny - 1 and my[i, j] > 0 and cluster_id[i, j] != cluster_id[i + 1, j]:
                mass_flows[cluster_id[i, j], cluster_id[i + 1, j]] += abs(my[i, j])
            if j > 0 and mz[i, j] < 0 and cluster_id[i, j] != cluster_id[i, j - 1]:
                mass_flows[cluster_id[i, j], cluster_id[i, j - 1]] += abs(mz[i, j])
            if j < Nz - 1 and mz[i, j] > 0 and cluster_id[i, j] != cluster_id[i, j + 1]:
                mass_flows[cluster_id[i, j], cluster_id[i, j + 1]] += abs(mz[i, j])
    return mass_flows


def get_axial_n(d):
    n = 360 / d
    return n


def get_cluster_volume(Ny, Nz, V, n_clusters, cluster_id, axial_n):
    cluster_volumes = np.zeros(n_clusters, dtype=float)
    for i in range(Ny):
        for j in range(Nz):
            cluster_volumes[cluster_id[i, j]] += V[i, j]
    return cluster_volumes * axial_n


def get_PFR_length(Ny, Nz, vy, vz, cluster_id, pfr_id, y, z):
    print(f"Computing PFR length for reactor {pfr_id}")
    inlet_cells = np.zeros((Ny, Nz), dtype=bool)
    for i in range(Ny):
        for j in range(Nz):
            if cluster_id[i, j] == pfr_id:
                if abs(vz[i, j]) >= abs(vy[i, j]):
                    if vz[i, j] > 0 and cluster_id[i, j - 1] != pfr_id or \
                            vz[i, j] < 0 and cluster_id[i, j + 1] != pfr_id:
                        inlet_cells[i, j] = True
                else:
                    if vy[i, j] > 0 and cluster_id[i - 1, j] != pfr_id or \
                            vy[i, j] < 0 and cluster_id[i + 1, j] != pfr_id:
                        inlet_cells[i, j] = True
    lengths = []
    for i in range(Ny):
        for j in range(Nz):
            if inlet_cells[i, j]:
                lengths.append(0)
                last_cell = (i, j)
                while True:
                    if abs(vz[i, j]) >= abs(vy[i, j]):
                        ii = last_cell[0]
                        if vz[i, j] > 0:
                            jj = last_cell[1] + 1
                        else:
                            jj = last_cell[1] - 1
                    else:
                        jj = last_cell[1]
                        if vy[i, j] > 0:
                            ii = last_cell[0] + 1
                        else:
                            ii = last_cell[0] - 1
                    new_cell = (ii, jj)
                    if ii < 0 or ii >= Ny or jj < 0 or jj >= Nz:
                        break
                    lengths[-1] += sqrt((y[new_cell] - y[last_cell]) * (y[new_cell] - y[last_cell]) + (z[new_cell] - z[last_cell]) * (z[new_cell] - z[last_cell]))
                    if cluster_id[new_cell] != pfr_id:
                        break
                    last_cell = new_cell
    return np.mean(lengths)


def post_process_boundary_data(Ny, Nz, y, z, T, vx, vy, vz, rho, boundary_data, inlets, radial_dir, axial_dir):
    for n in range(len(inlets)):
        boundary = inlets[n][0]
        for property_ in boundary_data[boundary]:
            if np.any(boundary_data[boundary][property_] == 'zeroGradient'):
                new_array = np.zeros(boundary_data[boundary][property_].shape, dtype=float)
                if property_ == 'T':
                    internal_property = T
                elif property_ == 'U':
                    new_array = np.zeros((boundary_data[boundary][property_].shape[0], 3), dtype=float)
                    internal_property_x = vx
                    internal_property_y = vy
                    internal_property_z = vz
                elif property_ == 'rho':
                    internal_property = rho
                lower_bound = inlets[n][1][1]
                N = -1
                base_ind = np.zeros(2, dtype=int)
                x = np.empty((Ny, Nz), dtype=float)
                f = 1
                if inlets[n][1][0] == 0:
                    N = Ny
                    base_ind = np.array([1, 0])
                    x = y
                elif inlets[n][1][0] == 1:
                    N = Nz
                    base_ind = np.array([0, 1])
                    x = z
                elif inlets[n][1][0] == 2:
                    N = Ny
                    base_ind = np.array([1, -1])
                    x = y
                    f *= -1
                elif inlets[n][1][0] == 3:
                    N = Nz
                    base_ind = np.array([-1, 1])
                    x = z
                    f *= -1
                i0 = 0
                for i in range(N):
                    ind = tuple([base_ind[k] * i if base_ind[k] > 0 else base_ind[k] for k in range(len(base_ind))])
                    if lower_bound > x[ind]:
                        i0 = i
                        break
                for i in range(len(boundary_data[boundary][property_])):
                    ind = tuple([base_ind[k] * i + i0 if base_ind[k] > 0 else base_ind[k] for k in range(len(base_ind))])
                    if property_ != 'U':
                        new_array[i] = internal_property[ind]
                    else:
                        new_array[i, radial_dir] = internal_property_y[ind]
                        new_array[i, axial_dir] = internal_property_z[ind]
                boundary_data[boundary][property_] = new_array
    return boundary_data


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(f"The number of reactors has not been defined, please run again the script with the desired number of reactors as a parameter.\n"
                           f"e.g. .\\CRNBuilder.py 5")
    elif len(sys.argv) > 2:
        raise RuntimeError(f"Too many parameters have been passed to the script.")
    else:
        n_clusters = int(sys.argv[1])
        print(f"Objective number of reactors: {n_clusters}")

    plot_graphs = True

    # Load data
    case_name = "HM1_bluff-body_flame"
    
    #Ny, Nz, y, z, V, vx, vy, vz, T, rho = initialize(case_name)
    Ny, Nz, y, z, V, vx, vy, vz, T, rho = initializeFromVTK('/shared_home/mgauges/ValidatingResults/HM1_bluff-body_flame/','case')
    np.save('vx.npy',vx)
    np.save('vy.npy',vy)
    np.save('vz.npy',vz)
    np.save('T.npy',T) 
    np.save('y.npy',y)    


    # Read input file
    inlets, outlets, y_lim, z_lim, angle, pressure, radial_dir, axial_dir, T_threshold = parse_input(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"CRNB_input.dic"))
    boundary_data = get_boundary_data(case_name, [inlets[i][0] for i in range(len(inlets))])
    boundary_data = post_process_boundary_data(Ny, Nz, y, z, T, vx, vy, vz, rho, boundary_data, inlets, radial_dir, axial_dir)
    print('Threshold Temp:')
    print(T_threshold)
    
    # Get eddy data
    # Give id 0 to cells not in an eddy, 1 to cells in the firsts eddy, 2 to the cells in the second eddy, etc...
    eddy_id = detect_eddy(case_name, Ny, Nz, y, z, vy, vz, load_cached=False, save_cache=False)

    # Find clusters
    cluster_id = cluster_data_linkage(case_name, Ny, Nz, y, z, T, n_clusters, eddy_id, threshold=T_threshold, load_cached=True, save_cache=True, to_cut=False, rework=False)
    n_clusters = int(np.max(cluster_id) + 1)
    np.save('clsuter_id',cluster_id)

    # Get mass flows
    axial_n = get_axial_n(angle)
    my, mz, Ay, Az = get_massflow(y, z, vy, vz, rho, axial_n, y_lim, z_lim, Ny, Nz, boundary_data, radial_dir, axial_dir, inlets)
    mass_flows = get_massflow_between_clusters(Ny, Nz, my, mz, n_clusters, cluster_id)
    np.save('my.npy',my)
    np.save('mz.npy_S',mz)
    np.save('mass_flows_S',mass_flows)


    
    # Compute the reactor type score (0: PFR, 1: PSR)
    pxr_score, clusters_sag_ang, var_sag_ang = get_pxr_score(Ny, Nz, y, z, vy, vz, cluster_id, extended_output=True)

    # Initialize the reactor network
    crn = ChemicalReactorNetwork(Ny, Nz, n_clusters, cluster_id, pxr_score, T, mass_flows, inlets, outlets, axial_n, y, z, my, mz, V, vy, vz, boundary_data)
    np.save('outlet_id_S',crn.outlet_id)
    np.save('cluster_id_S', cluster_id)

    # Generate an input.dic file to be used by NetSMOKE
    path = os.path.join(f"{os.getcwd()}", f"data", f"{case_name}")
    generate_input_dic(crn, path, pressure)

    if plot_graphs:
        # Plots
        print("Plotting results...")
        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(131, aspect='equal')
        cs = ax.pcolormesh(y, z, T, cmap='jet', shading='auto')
        ax.set_title('Temperature')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        fig.colorbar(cs)

        ax = fig.add_subplot(132, aspect='equal')
        cs = ax.pcolormesh(y, z, cluster_id, cmap='jet', shading='auto')
        ax.set_title('cluster_id')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        fig.colorbar(cs)


        ax = fig.add_subplot(133, aspect='equal')
        cs = ax.pcolormesh(y, z, vy, cmap='jet', shading='auto')
        ax.set_title('vy')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        fig.colorbar(cs)



        fig.savefig(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"out", f"fields.pdf"))

        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(111)
        G = nx.MultiDiGraph()
        G.add_nodes_from([(reactor, {"color": "red"}) for reactor in crn.reactors])
        G.add_nodes_from([(mixer, {"color": "blue"}) for mixer in crn.mixers])
        G.add_nodes_from([(splitter, {"color": "green"}) for splitter in crn.splitters])
        G.add_nodes_from([(crn_inlet, {"color": "purple"}) for crn_inlet in crn.crn_inlets])
        G.add_nodes_from([(crn_outlet, {"color": "cyan"}) for crn_outlet in crn.crn_outlets])
        for stream in crn.streams:
            G.add_edge(stream.inlet[0], stream.outlet[0])
        for inlet_stream in crn.crn_inlets:
            for outlet in inlet_stream.outlet:
                G.add_edge(inlet_stream, outlet)
        pos = nx.kamada_kawai_layout(G, scale=1.2)
        nx.draw_networkx(G, pos, with_labels=False, arrows=True)
        labels = dict((node, node.id_) for node in G.nodes)
        nx.draw(G, pos, labels=labels, node_color=[G.nodes[node]['color'] for node in G.nodes])

        fig.savefig(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"out", f"network.pdf"))

        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(111)
        G = nx.MultiDiGraph()
        G.add_nodes_from([(reactor, {"color": "red"}) for reactor in crn.reactors])
        G.add_nodes_from([(mixer, {"color": "blue"}) for mixer in crn.mixers])
        G.add_nodes_from([(splitter, {"color": "green"}) for splitter in crn.splitters])
        G.add_nodes_from([(crn_inlet, {"color": "purple"}) for crn_inlet in crn.crn_inlets])
        G.add_nodes_from([(stream, {"color": "yellow"}) for stream in crn.streams])
        for crn_object in crn.crn_objects + crn.streams:
            if not isinstance(crn_object, CRNOutlet):
                for inlet in crn_object.inlet:
                    G.add_edge(inlet, crn_object)
        pos = nx.kamada_kawai_layout(G, scale=1.2)
        nx.draw_networkx(G, pos, with_labels=False, arrows=True)
        labels = dict((node, node.id_) for node in G.nodes)
        nx.draw(G, pos, labels=labels, node_color=[G.nodes[node]['color'] for node in G.nodes])

        fig.savefig(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"out", f"network_with_streams.pdf"))

    print("Finished.")


if __name__ == '__main__':
    main()
