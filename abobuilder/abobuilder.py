from pyqint import PyQInt, Molecule
from pytessel import PyTessel
from abobuilder.element_table import ElementTable
import numpy as np
import os

class AboBuilder:
    def __init__(self):
        # set transparency
        self.alpha = 0.97
        
        # specify colors for occupied and virtual orbitals
        self.colors = [
            np.array([0.592, 0.796, 0.369, self.alpha], dtype=np.float32),
            np.array([0.831, 0.322, 0.604, self.alpha], dtype=np.float32),
            np.array([1.000, 0.612, 0.000, self.alpha], dtype=np.float32),
            np.array([0.400, 0.831, 0.706, self.alpha], dtype=np.float32)
        ]
        
        self.orbtemplate = [
            '1s', 
            '2s', '2px', '2py', '2pz', 
            '3s', '3px', '3py', '3pz', '3dx2', '3dy2', '3dz2', '3dxy', '3dxz', '3dyz',
            '4s', '4px', '4py', '4pz'
        ]

        self.et = ElementTable()

    def build_abo_model(self, outfile, models, colors):
        """
        Build managlyph atom/bonds/orbitals file from raw model data
        """
        # build integrator
        integrator = PyQInt()
        
        # build pytessel object
        pytessel = PyTessel()

        # build output file
        f = open(outfile, 'wb')

        # write number of frames
        f.write(int(1).to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        # write frame_idx
        f.write(int(1).to_bytes(2, byteorder='little'))

        descriptor = 'Geometry'

        f.write(len(descriptor).to_bytes(2, byteorder='little'))
        f.write(bytearray(descriptor, encoding='utf8'))

        # write nr_atoms
        f.write(int(0).to_bytes(2, byteorder='little'))

        # write number of models
        f.write(int(len(models)).to_bytes(2, byteorder='little'))

        for i,model in enumerate(models):

            # write model idx
            f.write(int(i).to_bytes(2, byteorder='little'))
            
            # write model color
            color = np.array(colors[i])
            f.write(color.tobytes())
            
            # write number of vertices
            f.write(model['vertices'].shape[0].to_bytes(4, byteorder='little'))
            
            # write vertices
            vn = np.hstack([model['vertices'], model['normals']])
            f.write(vn.tobytes())
            
            # write number of indices
            f.write(int(len(model['indices'])/3).to_bytes(4, byteorder='little'))
            
            # write indices
            f.write(model['indices'].tobytes())

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))

    def build_abo_orbs(self, outfile, nuclei, orbs, isovalue=0.03, overwrite_nuclei = None):
        """
        Build managlyph atom/bonds/orbitals file from previous HF calculation
        """
        # build integrator
        integrator = PyQInt()
        
        # build pytessel object
        pytessel = PyTessel()

        # build output file
        f = open(outfile, 'wb')

        # write number of frames
        nr_frames = len(orbs) + 1
        f.write(nr_frames.to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        # write frame_idx
        f.write(int(1).to_bytes(2, byteorder='little'))

        descriptor = 'Geometry'

        f.write(len(descriptor).to_bytes(2, byteorder='little'))
        f.write(bytearray(descriptor, encoding='utf8'))

        # write nr_atoms
        f.write(len(nuclei).to_bytes(2, byteorder='little'))
        for i,atom in enumerate(nuclei):
            if overwrite_nuclei:
                f.write(self.et.atomic_number_from_element(overwrite_nuclei[i]).to_bytes(1, byteorder='little'))
            else:
                f.write(self.et.atomic_number_from_element(atom[1]).to_bytes(1, byteorder='little'))
            f.write(np.array(atom[0], dtype=np.float32).tobytes())

        # write number of models
        f.write(int(0).to_bytes(1, byteorder='little'))
        f.write(int(0).to_bytes(1, byteorder='little'))

        #
        # Write the geometry including the orbitals
        #
        for i,key in enumerate(orbs):
            # write frame_idx
            f.write((i+1).to_bytes(2, byteorder='little'))

            descriptor = key

            f.write(len(descriptor).to_bytes(2, byteorder='little'))
            f.write(bytearray(descriptor, encoding='utf8'))

            # write nr_atoms
            f.write(len(nuclei).to_bytes(2, byteorder='little'))
            for a,atom in enumerate(nuclei):
                if overwrite_nuclei:
                    f.write(self.et.atomic_number_from_element(overwrite_nuclei[a]).to_bytes(1, byteorder='little'))
                else:
                    f.write(self.et.atomic_number_from_element(atom[1]).to_bytes(1, byteorder='little'))
                f.write(np.array(atom[0], dtype=np.float32).tobytes())

            print('Writing MO #%02i' % (i+1))

            # write number of models
            nrorbs = len(orbs[key]['orbitals'])
            f.write(int(nrorbs * 2.0).to_bytes(2, byteorder='little'))
            for j in range(0, nrorbs):
                # grab basis functions
                orb = orbs[key]['orbitals'][j]
                nucleus = nuclei[orb[0]-1]
                at = Molecule()
                at.add_atom(nucleus[1], nucleus[0][0], nucleus[0][1], nucleus[0][2], unit='angstrom')
                cgfs, _ = at.build_basis('sto3g')
                print('    Reading %i CGFS' % len(cgfs))
                sc = orb[2] # scalar coefficient to multiply all coefs with
                coeffvec = np.array([1.0 if self.orbtemplate[i] == orb[1] else 0.0 for i in range(0,len(cgfs))])
                
                # build the pos and negative isosurfaces from the cubefiles
                usz = 7.0 # unitcell size
                sz = 100
                grid = integrator.build_rectgrid3d(-usz, usz, sz)
                scalarfield = np.reshape(integrator.plot_wavefunction(grid, sc * coeffvec, cgfs), (sz, sz, sz))
                unitcell = np.diag(np.ones(3) * (usz * 2.0))
                
                for k in range(0,2):
                    vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(), scalarfield.shape, unitcell.flatten(), isovalue if k ==0 else -isovalue)
                    vertices_normals = np.hstack([vertices * 0.529177, normals])
                    
                    # write model idx
                    f.write(int(j*2+k).to_bytes(2, byteorder='little'))
                    
                    # write model color
                    color = np.array(self.colors[k])
                    f.write(color.tobytes())
                    
                    # write number of vertices
                    f.write(vertices_normals.shape[0].to_bytes(4, byteorder='little'))
                    
                    # write vertices
                    f.write(vertices_normals.tobytes())
                    
                    # write number of indices
                    f.write(int(len(indices)/3).to_bytes(4, byteorder='little'))
                    
                    # write indices
                    f.write(indices.tobytes())
                    
                    if k == 0:
                        print('    Writing positive lobe: %i vertices and %i facets' % (vertices_normals.shape[0], indices.shape[0] / 3))
                    else:
                        print('    Writing negative lobe: %i vertices and %i facets' % (vertices_normals.shape[0], indices.shape[0] / 3))

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))

    def build_abo_hf(self, outfile, nuclei, cgfs, coeff, energies, isovalue=0.03, maxmo=-1, sz=5.0, nsamples=100):
        """
        Build managlyph atom/bonds/orbitals file from
        previous HF calculation
        """
        # build integrator
        integrator = PyQInt()
        
        # build pytessel object
        pytessel = PyTessel()

        # build output file
        f = open(outfile, 'wb')

        # write number of frames
        nr_frames = len(cgfs) + 1 if maxmo == -1 else maxmo + 1
        f.write(nr_frames.to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        # write frame_idx
        f.write(int(1).to_bytes(2, byteorder='little'))

        descriptor = 'Geometry'

        f.write(len(descriptor).to_bytes(2, byteorder='little'))
        f.write(bytearray(descriptor, encoding='utf8'))

        # write nr_atoms
        f.write(len(nuclei).to_bytes(2, byteorder='little'))
        for atom in nuclei:
            f.write(atom[1].to_bytes(1, byteorder='little'))
            f.write(np.array(atom[0] * 0.529177, dtype=np.float32).tobytes())

        # write number of models
        f.write(int(0).to_bytes(1, byteorder='little'))
        f.write(int(0).to_bytes(1, byteorder='little'))

        # calculate number of electrons
        nelec = np.sum([atom[1] for atom in nuclei])

        #
        # Write the geometry including the orbitals
        #
        for i in range(1, nr_frames):
            # write frame_idx
            f.write((i+1).to_bytes(2, byteorder='little'))

            descriptor = 'Molecular orbital %i\nEnergy: %.4f eV' % (i,energies[i-1])

            f.write(len(descriptor).to_bytes(2, byteorder='little'))
            f.write(bytearray(descriptor, encoding='utf8'))

            # write nr_atoms
            f.write(len(nuclei).to_bytes(2, byteorder='little'))
            for atom in nuclei:
                f.write(atom[1].to_bytes(1, byteorder='little'))
                f.write(np.array(atom[0] * 0.529177, dtype=np.float32).tobytes())

            print('Writing MO #%02i' % i)

            # write number of models
            f.write(int(2).to_bytes(2, byteorder='little'))
            for j in range(0, 2):
                # build the pos and negative isosurfaces from the cubefiles
                grid = integrator.build_rectgrid3d(-sz, sz, nsamples)
                scalarfield = np.reshape(integrator.plot_wavefunction(grid, coeff[:,i-1], cgfs), (nsamples, nsamples, nsamples))
                unitcell = np.diag(np.ones(3) * (sz * 2.0))
                vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(), scalarfield.shape, unitcell.flatten(), isovalue if j==1 else -isovalue)
                vertices_normals = np.hstack([vertices * 0.529177, normals])
                
                # write model idx
                f.write(j.to_bytes(2, byteorder='little'))
                
                # write model color
                if i <= nelec / 2:
                    color = np.array(self.colors[j])
                else:
                    color = np.array(self.colors[j+2])
                f.write(color.tobytes())
                
                # write number of vertices
                f.write(vertices_normals.shape[0].to_bytes(4, byteorder='little'))
                
                # write vertices
                f.write(vertices_normals.tobytes())
                
                # write number of indices
                f.write(int(len(indices)/3).to_bytes(4, byteorder='little'))
                
                # write indices
                f.write(indices.tobytes())
                
                if j == 0:
                    print('    Writing positive lobe: %i vertices and %i facets' % (vertices_normals.shape[0], indices.shape[0] / 3))
                else:
                    print('    Writing negative lobe: %i vertices and %i facets' % (vertices_normals.shape[0], indices.shape[0] / 3))

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))
