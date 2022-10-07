
import myUtils as mU
import numpy as np
import pyvista as pv
import math as m

def initializeFromVTK(path,folderName):

    #read VTK data
    timeL,fileL = mU.readVTKSeries(path,folderName)
    #VTK is a pyvista pointset Unstructured grid
    VTK = pv.read(fileL[-1])
    #slice is Polydata
    sl = VTK.slice(normal='z')

    #create new mesh
    x = np.arange(0,0.09,0.0001499)
    y = np.arange(0,0.36,0.00119999)
    x,y = np.meshgrid(x,y)
    z = np.zeros(x.shape)
    grid = pv.StructuredGrid(x,y,z)

    #interpolate from VTK to structured grid - interpolated is a structured grid, point set
    grid.clear_data()
    interpolated = grid.interpolate(sl, radius = 0.001, sharpness = 10)
    interpolated = interpolated.point_data_to_cell_data()


    #extract information, not yet in correct format
    cellCenters = interpolated.cell_centers()
    Coordinates = cellCenters.points
    pT = interpolated.get_array('T') 
    pU = interpolated.get_array('U')
    prho  = interpolated.get_array('rho')

    pz = []
    py = []
    for i in range(len(Coordinates)):
        py.append(Coordinates[i][0])
        pz.append(Coordinates[i][1])
   
    #Nz and Ny still hardcoded!
    Nz= 300
    Ny= 600
    pvy = []
    pvz = []
    pvx = []

    for k in range(len(pU)):  
        pvy.append(pU[k][0])
        pvz.append(pU[k][1])
        pvx.append(pU[k][2])

    #restructure data 

    #create empty arrays 

    y = np.empty(shape = (Ny,Nz))
    z = np.empty(shape = (Ny,Nz))
    T = np.empty(shape = (Ny,Nz))
    rho = np.empty(shape = (Ny,Nz))
    V = np.empty(shape = (Ny,Nz))
    vz = np.empty(shape = (Ny,Nz))
    vy = np.empty(shape = (Ny,Nz))
    vx = np.empty(shape = (Ny,Nz))

    for i in range(Ny):
        for j in range(Nz):
            counter= Nz*i+j
            y[i,j] = py[counter]
            z[i,j] = pz[counter]
            T[i,j] = pT[counter]
            rho[i,j] = prho[counter]
            vz[i,j] = pvz[counter]
            vy[i,j] = pvy[counter]
            vx[i,j] = pvx[counter]         
            bounds = interpolated.cell_bounds(counter)
            ri = bounds[0]
            ro = bounds[1]
            h = bounds[3]-bounds[2]
            V[i,j] = m.pi*h*(ro**2 - ri**2)



    return Ny, Nz, y, z, V, vx, vy, vz, T, rho




'''
Ny, Nz, y, z, V, vx, vy, vz, T, rho,grid, sl = initializeFromVTK('/shared_home/mgauges/ValidatingResults/HM1_bluff-body_flame/','case')


p = pv.Plotter()

p.add_mesh(grid, show_edges = True, opacity = 0.5)
p.add_mesh(sl, color = 'white')
p.show()


'''



'''
#compare temperaturs / validate interpolation 
Tint = interpolated.get_array('T')
To = sl.get_array('T')

point = [0.001, 0.1,0]

cello =sl.find_containing_cell(point)
cellI = interpolated.find_containing_cell(point)

print('Temp of orig')
print(To[cello])  

print('temp of interpolated')
print(Tint[cellI])
'''
'''
#Plotting
p = pv.Plotter()
#p.add_mesh(sl, show_edges = False,scalars = 'T',cmap = 'jet')
#p.add_mesh(grid, show_edges = True)
p.add_mesh(interpolated,show_edges = False ,scalars='T',cmap='jet')
p.show_grid()

#p.add_mesh(grid, scalars = Xcoordinate)



p.show()
'''
