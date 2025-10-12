#############################################################################
#
# Script to dump information from a GAUSSIAN baf file for the CCResPy
# program
#
# To run: python3 readgau.py <file_name>.baf <name_directory>
# It will form the directory called <name_directory>_txts with a bunch
# of files in it.
############################################################################
#
import sys
import os
import re
import math
import io
import numpy as np
import copy
# GAUOPEN path
sys.path.insert(0, '/Volumes/gaussian/gdv_j30p/')
np.set_printoptions(precision=16,threshold=sys.maxsize,floatmode='fixed')

from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
from gauopen import QCUtil as qcu
from gauopen import QCUtilH as su


def main():
# open the file
  FName = sys.argv[1]
  mol=sys.argv[2]
  os.system(f"mkdir {mol}_txts")
 
  baf = qcb.QCBinAr(file=FName)
  geometry = baf.c
  nb = baf.nbasis
  nae = (baf.ne+baf.multip-1)//2         # number of alpha electrons
  nbe = (baf.ne-baf.multip+1)//2         # number of beta electrons
  noa = nae - baf.nfc                    # number of active (not frozen
  nob = nbe - baf.nfc                    # core or virtual) electrons
  nrorb = nb - baf.nfc - baf.nfv         # number of active orbitals
  nva = nrorb - noa                      # number of active alpha virtuals
  nvb = nrorb - nob                      # number of active betavirtuals
  scfE = baf.scalar("escf")
  if (noa != nob):
    print(f" Only NOA=NOB case is allowed for now.")
    exit()
  ipbc = []
  if "File 733 Integers" in baf.matlist:
    #
    # This is a PBC calculation
    #
    ipbc = baf.matlist["File 733 Integers"].array
    nmtpbc = ipbc[1]
    nbx = nb*nmtpbc
    with open(f"{mol}_txts/pbc_info.txt","w") as writer:
      writer.write(str(ipbc))
    # twoeint = baf.matlist["PBC 2E INTEGRALS"]#.array.reshape((nbx,nbx,nbx,nbx))
    ntt = (nb*(nb+1))//2
    occ = [noa,nob,nva,nvb,baf.nfc,baf.nfv]
    with open(f"{mol}_txts/occ.txt","w") as writer:
      writer.write(str(occ))
    miller_ind = baf.matlist["K-POINT MILLER INDICES"].expand()
    with open(f"{mol}_txts/miller.txt","w") as writer:
      writer.write(str(miller_ind))
    k_phases = baf.matlist["K-POINT PHASES"].expand()
    with open(f"{mol}_txts/k_phases.txt","w") as writer:
      writer.write(str(k_phases))
    k_factors = baf.matlist["K-POINT FACTORS"].expand()
    with open(f"{mol}_txts/k_factors.txt","w") as writer:
      writer.write(str(k_factors))
    k_weights = baf.matlist["K-POINT WEIGHTS"].expand()
    with open(f"{mol}_txts/k_weights.txt","w") as writer:
      writer.write(str(k_weights))
    MO_weights = baf.matlist["PBC ORBITAL WEIGHTS"].expand()
    with open(f"{mol}_txts/MO_weights.txt","w") as writer:
      writer.write(str(MO_weights))
    orbE = baf.matlist["PBC ORBITAL ENERIES"].expand()
    mocoef=baf.matlist["PBC ALPHA ORBITALS"].array
  else:
    #
    # This is a molecular calculation
    #
    # # 2ERI
    # twoeint=baf.matlist["REGULAR 2E INTEGRALS"]
    # Orbital energies
    orbE = baf.matlist["ALPHA ORBITAL ENERGIES"].expand()
    with open(f"{mol}_txts/orbE.txt","w") as writer:
      writer.write(str(orbE))
    # Number of orbitals
    occ = [noa,nob,nva,nvb,baf.nfc,baf.nfv]
    # MO coefficients
    mocoef=baf.matlist["ALPHA MO COEFFICIENTS"].expand()

  # Write general stuff on disk
  with open(f"{mol}_txts/geometry.txt","w") as writer:
    writer.write(str(geometry))
  fock_r = baf.matlist["ALPHA FOCK MATRIX"].array#.reshape((nmtpbc,ntt))
  with open(f"{mol}_txts/fock.txt","w") as writer:
    writer.write(str(fock_r))
  overlap = baf.matlist["OVERLAP"].array#.reshape((nmtpbc,ntt))
  with open(f"{mol}_txts/overlap.txt","w") as writer:
    writer.write(str(overlap))
  core = baf.matlist["CORE HAMILTONIAN ALPHA"].array#.reshape((nmtpbc,ntt))
  with open(f"{mol}_txts/core.txt","w") as writer:
    writer.write(str(core))
  with open(f"{mol}_txts/scf.txt","w") as writer:
    writer.write(str(scfE))
  with open(f"{mol}_txts/occ.txt","w") as writer:
    writer.write(str(occ))
  with open(f"{mol}_txts/orbE.txt","w") as writer:
    writer.write(str(orbE))
  with open(f"{mol}_txts/mocoef.txt","w") as writer:
    writer.write(str(mocoef))
  # with open(f"{mol}_txts/twoeint.txt","w") as writer:
  #   writer.write(str(twoeint))
      
  # Dipole integrals, length gauge
  if "DIPOLE INTEGRALS" in baf.matlist:
    dip_r = baf.matlist["DIPOLE INTEGRALS"].array
    with open(f"{mol}_txts/dipole_r.txt","w") as writer:
      writer.write(str(dip_r))

  # Dipole integrals, velocity gauge
  if "DIP VEL INTEGRALS" in baf.matlist:
    dip_r = baf.matlist["DIP VEL INTEGRALS"].array
    with open(f"{mol}_txts/dipole_v.txt","w") as writer:
      writer.write(str(dip_r))

  
main()

