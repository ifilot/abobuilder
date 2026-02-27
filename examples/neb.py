from abobuilder import AboBuilder
import os

def main():
    ab = AboBuilder()
    path = os.path.join('/', 'mnt', 'd', 'data_bart_zijlstra', 'NEB', 'C1', 'C+H--CH', 'bz')
    ab.build_abo_neb_vasp('ch_hydr.abo', path, -4, (3,3))

    path = os.path.join('/', 'mnt', 'd', 'data_bart_zijlstra', 'NEB', 'C1O1', 'CO--C+O', 'b5rev')
    ab.build_abo_neb_vasp('co_diss.abo', path, -4, (3,3))

if __name__ == '__main__':
    main()
