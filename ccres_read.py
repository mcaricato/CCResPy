############################################################################
#
# This file contains the functions used by the main CCResPy program
# v1.0.0 to read the reference wave function information
#
# This program is licensed under the terms of the GNU General Public
# License v3.0 or later
############################################################################
#
import numpy as np
import os
import sys
import re
import time
from scipy.constants import angstrom, physical_constants
np.set_printoptions(precision=16,threshold=sys.maxsize,floatmode='fixed')
from ccres_funct import fourier, basis_tran, fill_kl, square_m
from ccres_funct import denom, DEk, mem_check, mol_mass
# GAUOPEN path
sys.path.insert(0, '/Volumes/gaussian/gdv_j30p/')
np.set_printoptions(precision=16,threshold=sys.maxsize,floatmode='fixed')
from gauopen import QCBinAr as qcb
from gauopen import QCOpMat as qco
import gauopen.qcmio as qcmio


##########################################################################
#Get O, NB, SCF energy, MO coefficients, Orbital energies ################
##########################################################################

def getFort(mol,FreezeCore):
  # Read atoms list
  atoms_list = []
  with open(f"{mol}_txts/atoms_list.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  # Remove parentheses and empty spaces
  for i in range(len(text)):
    for j in range(len(text[i])):
      text[i][j] = text[i][j].replace("[","")
      text[i][j] = text[i][j].replace("]","")
  # Remove empty slots
  for i in range(len(text)):
    text[i][:] = [x for x in text[i] if x]
  for i in range(len(text)):
    for j in range(len(text[i])):
      atoms_list.append(int(text[i][j]))
  # Read geometry
  bohr_radius = physical_constants["Bohr radius"][0]
  toAng = bohr_radius/angstrom 
  geometry = []
  N_atoms = len(atoms_list)
  with open(f"{mol}_txts/geometry.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  # Remove parentheses and empty spaces
  for i in range(len(text)):
    for j in range(len(text[i])):
      text[i][j] = text[i][j].replace("[","")
      text[i][j] = text[i][j].replace("]","")
  # Remove empty slots
  for i in range(len(text)):
    text[i][:] = [x for x in text[i] if x]
  for i in range(len(text)):
    for j in range(len(text[i])):
      geometry.append(float(text[i][j]))
  geometry = np.array(geometry).reshape((N_atoms,3))
  geometry *= toAng
  # Read orbital numbers
  O=0
  V=0
  with open(f"{mol}_txts/occ.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split(","))
  # Remove parentheses and empty spaces
  for j in range(len(text[0])):
    text[0][j] = text[0][j].replace("[","")
    text[0][j] = text[0][j].replace("]","")
    text[0][j] = text[0][j].replace(" ","")
  OA = int(text[0][0])
  OB = int(text[0][1])
  VA = int(text[0][2])
  VB = int(text[0][3])
  FC = int(text[0][4])
  FV = int(text[0][5])
  # Impose frozen core appriximation (always freeze the same number of
  # alpha and beta orbitals)
  if(FC == 0 and FreezeCore):
    for a in range(len(atoms_list)):
      if(atoms_list[a] > 2 and atoms_list[a] < 11): FC += 1
      elif(atoms_list[a] > 10): FC += 5
    OA -= FC
    OB -= FC
  # Create list of atom names in the molecule/unit cell
  mol_weight, atoms_list_names = mol_mass(atoms_list)
  #
  # Print out geometry and orbital information
  with open(f"{mol}.txt","a") as writer:
    writer.write(f"\nGeometry (Ang):\n") 
    for a in range(len(atoms_list)):
      writer.write(f"{atoms_list_names[a]} {geometry[a,0]:+.8f} {geometry[a,1]:+.8f} {geometry[a,2]:+.8f} \n") 
    writer.write(f"\nOrbitals Information:\n" 
    f"N-Occ. Alpha: {OA}, Beta: {OB} -- N-Vir. Alpha: {VA}, Beta: {VB}\n"
    f"N-frozen core: {FC} -- N-frozen virtuals: {FV}\n")
  if(OA != OB):
    print(f"Not ready for open shell yet")
    exit()
  O = OA
  V = VA
  NB = O + V + FC + FV
  NOrb = O + V
  #
  #SCF Energy
  with open(f"{mol}_txts/scf.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  scfE=float(text[0][0])
  #
  # Read PBC info if available
  ipbc=[]
  k_weights=[]
  if(os.path.exists(f"{mol}_txts/pbc_info.txt")):
    #
    # Read PBC integers
    with open(f"{mol}_txts/pbc_info.txt","r") as reader:
      text=[]
      for line in reader:
        text.append(line.split())
    # Remove parentheses
    for i in range(len(text)):
      for j in range(len(text[i])):
        text[i][j] = text[i][j].replace("[","")
        text[i][j] = text[i][j].replace("]","")
    # Remove empty slots
    for i in range(len(text)):
      text[i][:] = [x for x in text[i] if x]
    for i in range(len(text)):
      for j in range(len(text[i])):
        ipbc.append(int(text[i][j]))
    #
    # Read k-point weigths
    with open(f"{mol}_txts/k_weights.txt","r") as reader:
      text=[]
      for line in reader:
        text.append(line.split())
    # Remove parentheses
    for i in range(len(text)):
      for j in range(len(text[i])):
        text[i][j] = text[i][j].replace("[","")
        text[i][j] = text[i][j].replace("]","")
    # Remove empty slots
    for i in range(len(text)):
      text[i][:] = [x for x in text[i] if x]
    for i in range(len(text)):
      for j in range(len(text[i])):
        k_weights.append(float(text[i][j]))
    k_weights = np.array(k_weights)
  #
  # Read all MO Coefficients
  MOCoef=[[] for _ in range(NB)]
  with open(f"{mol}_txts/mocoef.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  # Remove parentheses and empty spaces
  for i in range(len(text)):
    for j in range(len(text[i])):
      text[i][j] = text[i][j].replace("[","")
      text[i][j] = text[i][j].replace("]","")
  # Remove empty slots
  for i in range(len(text)):
    text[i][:] = [x for x in text[i] if x]
  MOCoef = []
  if(ipbc):
    # PBC
    for i in range(len(text)):
      for j in range(len(text[i])):
        MOCoef.append(complex(text[i][j]))
    kp, l_list = fill_kl(ipbc)
    Nkp = len(kp)
    nrecip = ipbc[9]
    if(nrecip % 2 == 0):
      MOCoef = np.array(MOCoef).reshape((Nkp//2,NB,NB))
      MOCoef = np.append(MOCoef,np.conjugate(MOCoef))
    else:
      nMO = Nkp//2+1
      MOCoef = np.array(MOCoef).reshape((nMO,NB*NB))
      MOCoef = np.append(MOCoef,np.conjugate(MOCoef[1:-1,:]))
    MOCoef = np.array(MOCoef).reshape((Nkp,NB,NB))
  else:    
    # Molecular 
    for i in range(len(text)):
      for j in range(len(text[i])):
        MOCoef.append(float(text[i][j]))
    MOCoef = np.array(MOCoef).reshape((NB,NB))
  return O, V, FC, FV, NB, scfE, MOCoef, ipbc, k_weights, atoms_list

########################################################
# Routine the read the Overlap.
########################################################
def getOvl(mol,O,V,NB,ipbc,basis,dk,MOCoef):
  # basis: "AO" or "MO"
  # dk: = F: regular MO(k) basis, = T: dS/dK in MO(k) basis
  NOrb = O + V
  with open(f"{mol}_txts/overlap.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  # Remove parentheses
  for i in range(len(text)):
    for j in range(len(text[i])):
      text[i][j] = text[i][j].replace("[","")
      text[i][j] = text[i][j].replace("]","")
  # Remove empty slots
  for i in range(len(text)):
    text[i][:] = [x for x in text[i] if x]
  Ovl_r = []
  for i in range(len(text)):
    for j in range(len(text[i])):
      Ovl_r.append(float(text[i][j]))
  if(ipbc):
    # PBC calculation
    nmtpbc = ipbc[1]
    ntt = (NB*(NB+1))//2
    Ovl_r = np.array(Ovl_r).reshape((nmtpbc,ntt))
    if(basis == "AO"):
      Ovl = np.copy(Ovl_r)
      del Ovl_r
    elif(basis == "MO"):
      kp, l_list = fill_kl(ipbc)
      Nkp = len(kp)
      Ovl_k_lt = fourier("Dir",ipbc,Ovl_r,dk)
      OvlA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Ovl_k_lt)
      Ovl = np.zeros((Nkp,NOrb*2,Nkp,NOrb*2),dtype=complex)
      for k in range(Nkp):
        # Fill out the alpha and beta blocks
        # oa-oa
        Ovl[k,:O,k,:O] = OvlA[k,:O,:O]
        # ob-ob
        Ovl[k,O:2*O,k,O:2*O] = OvlA[k,:O,:O]
        # va-va
        Ovl[k,2*O:2*O+V,k,2*O:2*O+V] = OvlA[k,O:,O:]
        # vb-vb
        Ovl[k,2*O+V:,k,2*O+V:] = OvlA[k,O:,O:]
        # oa-va
        Ovl[k,:O,k,2*O:2*O+V] = OvlA[k,:O,O:]
        # va-oa
        Ovl[k,2*O:2*O+V,k,:O] = OvlA[k,O:,:O]
        # ob-vb
        Ovl[k,O:2*O,k,2*O+V:] = OvlA[k,:O,O:]
        # vb-ob
        Ovl[k,2*O+V:,k,O:2*O] = OvlA[k,O:,:O]
      del Ovl_r, Ovl_k_lt, OvlA
      Ovl = Ovl.reshape((Nkp*NOrb*2,Nkp*NOrb*2))
    else:
      print(f"Wrong basis option in getOvl: {basis}")
  else:
    # Molecular calculation
    Ovl_r = np.array(Ovl_r)
    if(basis == "AO"):
      Ovl = np.copy(Ovl_r)
      del Ovl_r
    elif(basis == "MO"):
      # symmetrize and transform to MO basis
      Ovlsq = np.zeros((NB,NB))
      Ovlsq = square_m(NB,True,"Sym",Ovl_r,Ovlsq)
      temp = np.einsum("in,nm->im", MOCoef, Ovlsq, optimize=True)
      Ovlsq = np.einsum("jm,im->ij", MOCoef, temp, optimize=True)
      # Fill out the alpha and beta blocks
      Ovl = np.zeros((2*NOrb,2*NOrb))
      Ovl[:O,:O] = Ovlsq[:O,:O]
      Ovl[O:2*O,O:2*O] = Ovlsq[:O,:O]
      Ovl[2*O:2*O+V,2*O:2*O+V] = Ovlsq[O:,O:]
      Ovl[2*O+V:,2*O+V:] = Ovlsq[O:,O:]
      del Ovl_r, Ovlsq
  return Ovl
  
########################################################
# Routine the read the Fock matrix
########################################################
def getFock(mol,O,V,NB,ipbc,basis,dk,MOCoef):
  # basis: "AO" or "MO"
  # dk: = F: regular MO(k) basis, = T: dS/dK in MO(k) basis
  NOrb = O + V
  with open(f"{mol}_txts/fock.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  # Remove parentheses
  for i in range(len(text)):
    for j in range(len(text[i])):
      text[i][j] = text[i][j].replace("[","")
      text[i][j] = text[i][j].replace("]","")
  # Remove empty slots
  for i in range(len(text)):
    text[i][:] = [x for x in text[i] if x]
  Fock_r = []
  for i in range(len(text)):
    for j in range(len(text[i])):
      Fock_r.append(float(text[i][j]))
  if(ipbc):
    # PBC calculation
    nmtpbc = ipbc[1]
    ntt = (NB*(NB+1))//2
    Fock_r = np.array(Fock_r).reshape((nmtpbc,ntt))
    if(basis == "AO"):
      Fock = np.copy(Fock_r)
      del Fock_r
    elif(basis == "MO"):
      kp, l_list = fill_kl(ipbc)
      Nkp = len(kp)
      Fock_k_lt = fourier("Dir",ipbc,Fock_r,dk)
      FockA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Fock_k_lt)
      Fock = np.zeros((Nkp,NOrb*2,Nkp,NOrb*2),dtype=complex)
      for k in range(Nkp):
        # Fill out the alpha and beta blocks
        # oa-oa
        Fock[k,:O,k,:O] = FockA[k,:O,:O]
        # ob-ob
        Fock[k,O:2*O,k,O:2*O] = FockA[k,:O,:O]
        # va-va
        Fock[k,2*O:2*O+V,k,2*O:2*O+V] = FockA[k,O:,O:]
        # vb-vb
        Fock[k,2*O+V:,k,2*O+V:] = FockA[k,O:,O:]
        # oa-va
        Fock[k,:O,k,2*O:2*O+V] = FockA[k,:O,O:]
        # va-oa
        Fock[k,2*O:2*O+V,k,:O] = FockA[k,O:,:O]
        # ob-vb
        Fock[k,O:2*O,k,2*O+V:] = FockA[k,:O,O:]
        # vb-ob
        Fock[k,2*O+V:,k,O:2*O] = FockA[k,O:,:O]
      del Fock_r, Fock_k_lt, FockA
      Fock = Fock.reshape((Nkp*NOrb*2,Nkp*NOrb*2))
    else:
      print(f"Wrong basis option in getFock: {basis}")
  else:
    # Molecular calculation
    Fock_r = np.array(Fock_r)
    if(basis == "AO"):
      Fock = np.copy(Fock_r)
      del Fock_r
    elif(basis == "MO"):
      # symmetrize and transform to MO basis
      Focksq = np.zeros((NB,NB))
      Focksq = square_m(NB,True,"Sym",Fock_r,Focksq)
      temp = np.einsum("in,nm->im", MOCoef, Focksq, optimize=True)
      Focksq = np.einsum("jm,im->ij", MOCoef, temp, optimize=True)
      # Fill out the alpha and beta blocks
      Fock = np.zeros((2*NOrb,2*NOrb))
      Fock[:O,:O] = Focksq[:O,:O]
      Fock[O:2*O,O:2*O] = Focksq[:O,:O]
      Fock[2*O:2*O+V,2*O:2*O+V] = Focksq[O:,O:]
      Fock[2*O+V:,2*O+V:] = Focksq[O:,O:]
      del Fock_r, Focksq
  return Fock
  
########################################################
# Get 2e integrals
########################################################
#def get2e(NB,ipbc):
def get2e(NB,ipbc,molecule,scratch):
  nmtpbc = 0
  NBX = NB
  mol_int = sys.argv[2]
  if(ipbc):
    nmtpbc = ipbc[1]
    NBX = NB*nmtpbc
    NCMax = (nmtpbc-1)//2
    kp, l_list = fill_kl(ipbc)
  # AOInt=np.zeros((NBX, NBX, NBX, NBX))
  # mol=sys.argv[1]
  # icount = 0
  # with open(f"{mol}_txts/twoeint.txt", "r") as reader:
  #   for line in reader:
  #     text=line.split()
  #     if "I=" and "J=" and "K=" and "L=" in text:
  #       icount += 1
  #       I=int(text[1])-1
  #       J=int(text[3])-1
  #       K=int(text[5])-1
  #       L=int(text[7])-1
  #       integ = float(text[9].replace("D", "E"))
  #       # AOInt[I,J,K,L] = integ
  #       # AOInt[J,I,K,L] = integ
  #       # AOInt[I,J,L,K] = integ
  #       # AOInt[J,I,L,K] = integ
  #       # AOInt[K,L,I,J] = integ
  #       # AOInt[L,K,I,J] = integ
  #       # AOInt[K,L,J,I] = integ
  #       # AOInt[L,K,J,I] = integ
  #       if(ipbc):
  #         # Spread integral over cells
  #         iq = I//NB
  #         jq = J//NB
  #         kq = K//NB
  #         lq = L//NB
  #         # cell number for each function
  #         ic = l_list[iq]
  #         jc = l_list[jq]
  #         kc = l_list[kq]
  #         lc = l_list[lq]
  #         # function number in each cell
  #         ir = I%NB
  #         jr = J%NB
  #         kr = K%NB
  #         lr = L%NB
  #         if(icount == 17501):
  #           print(f"IJKL={I+1},{J+1},{K+1},{L+1} -- {integ}")
  #           print(f"Cells={ic},{jc},{kc},{lc}")
  #           print(f"Functions={ir+1},{jr+1},{kr+1},{lr+1}")
  #         # shift first function to cell 0
  #         iic = 0
  #         jjc = jc - ic
  #         kkc = kc - ic
  #         llc = lc - ic
  #         # make sure we are not shifting out of range
  #         if(max(abs(jjc),abs(kkc),abs(llc)) <= NCMax):
  #           II = ir
  #           JJ = jr + NB*l_list.index(jjc)
  #           KK = kr + NB*l_list.index(kkc)
  #           LL = lr + NB*l_list.index(llc)
  #           if(icount == 17501):
  #             print(f"I0JKL={II+1},{JJ+1},{KK+1},{LL+1}")
  #             print(f"Cells={iic},{jjc},{kkc},{llc}")
  #           AOInt[II,JJ,KK,LL] = integ
  #           AOInt[JJ,II,KK,LL] = integ
  #           AOInt[II,JJ,LL,KK] = integ
  #           AOInt[JJ,II,LL,KK] = integ
  #           AOInt[KK,LL,II,JJ] = integ
  #           AOInt[LL,KK,II,JJ] = integ
  #           AOInt[KK,LL,JJ,II] = integ
  #           AOInt[LL,KK,JJ,II] = integ
  #         # else:
  #         #   print(f"Cells bad={iic},{jjc},{kkc},{llc}")
  #         #   print(f"IJKL={I+1},{J+1},{K+1},{L+1} -- {integ}")
  #         #   print(f"Cells={ic},{jc},{kc},{lc}")
  #         #   print(f"Functions={ir+1},{jr+1},{kr+1},{lr+1}")
  #         #   exit()
  #         # shift second function to cell 0
  #         iic = ic - jc
  #         jjc = 0
  #         kkc = kc - jc
  #         llc = lc - jc
  #         # make sure we are not shifting out of range
  #         if(max(abs(iic),abs(kkc),abs(llc)) <= NCMax):
  #           II = ir + NB*l_list.index(iic)
  #           JJ = jr 
  #           KK = kr + NB*l_list.index(kkc)
  #           LL = lr + NB*l_list.index(llc)
  #           if(icount == 17501):
  #             print(f"IJ0KL={II+1},{JJ+1},{KK+1},{LL+1}")
  #             print(f"Cells={iic},{jjc},{kkc},{llc}")
  #           AOInt[II,JJ,KK,LL] = integ
  #           AOInt[JJ,II,KK,LL] = integ
  #           AOInt[II,JJ,LL,KK] = integ
  #           AOInt[JJ,II,LL,KK] = integ
  #           AOInt[KK,LL,II,JJ] = integ
  #           AOInt[LL,KK,II,JJ] = integ
  #           AOInt[KK,LL,JJ,II] = integ
  #           AOInt[LL,KK,JJ,II] = integ
  #         # else:
  #         #   print(f"Cells bad={iic},{jjc},{kkc},{llc}")
  #         #   print(f"IJKL={I+1},{J+1},{K+1},{L+1} -- {integ}")
  #         #   print(f"Cells={ic},{jc},{kc},{lc}")
  #         #   print(f"Functions={ir+1},{jr+1},{kr+1},{lr+1}")
  #         #   exit()
  #         # shift third function to cell 0
  #         iic = ic - kc
  #         jjc = jc - kc
  #         kkc = 0
  #         llc = lc - kc
  #         # make sure we are not shifting out of range
  #         if(max(abs(iic),abs(jjc),abs(llc)) <= NCMax):
  #           II = ir + NB*l_list.index(iic)
  #           JJ = jr + NB*l_list.index(jjc)
  #           KK = kr 
  #           LL = lr + NB*l_list.index(llc)
  #           if(icount == 17501):
  #             print(f"IJK0L={II+1},{JJ+1},{KK+1},{LL+1}")
  #             print(f"Cells={iic},{jjc},{kkc},{llc}")
  #           AOInt[II,JJ,KK,LL] = integ
  #           AOInt[JJ,II,KK,LL] = integ
  #           AOInt[II,JJ,LL,KK] = integ
  #           AOInt[JJ,II,LL,KK] = integ
  #           AOInt[KK,LL,II,JJ] = integ
  #           AOInt[LL,KK,II,JJ] = integ
  #           AOInt[KK,LL,JJ,II] = integ
  #           AOInt[LL,KK,JJ,II] = integ
  #         # else:
  #         #   print(f"Cells bad={iic},{jjc},{kkc},{llc}")
  #         #   print(f"IJKL={I+1},{J+1},{K+1},{L+1} -- {integ}")
  #         #   print(f"Cells={ic},{jc},{kc},{lc}")
  #         #   print(f"Functions={ir+1},{jr+1},{kr+1},{lr+1}")
  #         #   exit()
  #         # shift fourth function to cell 0
  #         iic = ic - lc
  #         jjc = jc - lc
  #         kkc = kc - lc
  #         llc = 0
  #         # make sure we are not shifting out of range
  #         if(max(abs(iic),abs(jjc),abs(kkc)) <= NCMax):
  #           II = ir + NB*l_list.index(iic)
  #           JJ = jr + NB*l_list.index(jjc)
  #           KK = kr + NB*l_list.index(kkc)
  #           LL = lr 
  #           if(icount == 17501):
  #             print(f"IJKL0={II+1},{JJ+1},{KK+1},{LL+1}")
  #             print(f"Cells={iic},{jjc},{kkc},{llc}")
  #           AOInt[II,JJ,KK,LL] = integ
  #           AOInt[JJ,II,KK,LL] = integ
  #           AOInt[II,JJ,LL,KK] = integ
  #           AOInt[JJ,II,LL,KK] = integ
  #           AOInt[KK,LL,II,JJ] = integ
  #           AOInt[LL,KK,II,JJ] = integ
  #           AOInt[KK,LL,JJ,II] = integ
  #           AOInt[LL,KK,JJ,II] = integ
  #       else:
  #         AOInt[I,J,K,L] = integ
  #         AOInt[J,I,K,L] = integ
  #         AOInt[I,J,L,K] = integ
  #         AOInt[J,I,L,K] = integ
  #         AOInt[K,L,I,J] = integ
  #         AOInt[L,K,I,J] = integ
  #         AOInt[K,L,J,I] = integ
  #         AOInt[L,K,J,I] = integ
  baf = qcb.QCBinAr(file=f"{mol_int}.baf")
  # AOInt0 = baf.matlist["REGULAR 2E INTEGRALS"]
  # lenexp,nshape,ndimens,lennew,shapedef = AOInt0.shapeargs()
  # with open(f"{molecule}.txt","a") as writer:
  #   writer.write(f"Start ERI\n")
  # ERI = np.zeros((NBX,NBX,NBX,NBX))
  # with open(f"{molecule}.txt","a") as writer:
  #   writer.write(f"Define ERI\n")
  # if(ipbc):
  #   ERI = np.zeros((NB,nmtpbc*NB,nmtpbc*NB,nmtpbc*NB))
  # else:
  #   ERI = np.zeros(shapedef)
  # print(f" ERI shape: {ERI.shape}, size: {np.size(ERI)}")
  # np.save(f"{scratch}/{molecule}-ERI-AO",ERI)
  # with open(f"{molecule}.txt","a") as writer:
  #   writer.write(f"Save ERI\n")
  # del ERI
  # with open(f"{molecule}.txt","a") as writer:
  #   writer.write(f"Delete ERI\n")
  # print(f" ERI save file")
  # char = AOInt0.charsplit()
  # print(f"Int2E, type: {type(AOInt0)}, size: {np.size(AOInt0)}  typed= {AOInt0.typed}, typea = {AOInt0.typea}, shape = {np.shape(AOInt0.array)}  \n")
  # print(f"lr: {AOInt0.array.size}, {AOInt0.nelem}, {AOInt0.array.size//AOInt0.nelem} \n")
  # print(f"tuple: {np.shape(AOInt0.array)}, {reversed(np.shape(AOInt0.array))}, {tuple(reversed(np.shape(AOInt0.array)))} {AOInt0.dimens}\n")
  # ERI = np.load(f"{scratch}/{molecule}-ERI-AO.npy",mmap_mode='r+')

  # ERI = np.lib.format.open_memmap(f"{scratch}/{molecule}-ERI-AO.npy",mode='w+',
  #                                 shape=(NBX,NBX,NBX,NBX)) 
  # tot_mem, avlb_mem = mem_check()
  # 
  # eri0 = baf.matlist["REGULAR 2E INTEGRALS"]
  # lenexp,nshape,ndimens,lennew,shapedef = eri0.shapeargs()
  # name,ni,nr,nri,ntot,n1,n2,n3,n4,n5,typea = eri0.labpars
  # print(f"dimens= {eri0.dimens} {eri0.nindices} {eri0.indices} {eri0.nelem} \n")
  # print(f"{lenexp},{nshape},{ndimens},{lennew},{shapedef}\n")
  # print(f"{eri0.dimdef} \n")
  # print(f"{type(eri0)} \n")
  # print(f" labpars: {name},{ni},{nr},{nri},{ntot},{n1},{n2},{n3},{n4},{n5},{typea}")
  # print(f"print2e: n4:{n4}, nr:{nr}, ntot:{ntot} \n")
  # qco.print2e(eri0.name,n4,nr,ntot,eri0.array)
  # 
  # r = eri0.array.reshape([ntot,nr]) 
  # for i in range(n4):
  #   iis = i*n4
  #   for j in range(i+1):
  #     ijs = (j+iis)*n4
  #     for k in range(i+1):
  #       ijks = (k+ijs)*n4
  #       if i == k: llim = j + 1
  #       else: llim = k + 1
  #       for l in range(llim):
  #         ijkl,_ = qcmio.lind4(-n4,-n4,-n4,n4,i+1,j+1,k+1,l+1)
  #         doit = False
  #         for x in range(nr):
  #           doit = doit or (abs(r[ijkl,x]) >= 1.e-12)
  #         if doit:
  #           ERI[i,j,k,l] = r[ijkl,0]
  #           ERI[j,i,k,l] = r[ijkl,0]
  #           ERI[i,j,l,k] = r[ijkl,0]
  #           ERI[j,i,l,k] = r[ijkl,0]
  #           ERI[k,l,i,j] = r[ijkl,0]
  #           ERI[l,k,i,j] = r[ijkl,0]
  #           ERI[k,l,j,i] = r[ijkl,0]
  #           ERI[l,k,j,i] = r[ijkl,0]
  # print(f" ERI \n {ERI}")
  # exit()

  
  ERI = np.lib.format.open_memmap(f"{scratch}/{molecule}-ERI-AO.npy",mode='w+',
                                  shape=(NBX,NBX,NBX,NBX)) 
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"Load ERI {np.size(ERI)} {ERI.shape} \n")
  tot_mem, avlb_mem = mem_check()
  if(avlb_mem*(1024**3) > np.size(ERI)*8):
  # if(False):
    # print(f"enough memory: {avlb_mem} vs {np.size(ERI)*8/(1024**3)}")
    ERI[:,:,:,:] = baf.matlist["REGULAR 2E INTEGRALS"].expand()
  else:
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f" Not enough memory to simply expand 2ERIs:\n")
      writer.write(f" AvlMem: {avlb_mem:.2f}GB vs 2ERI size: {8*np.size(ERI)/(1024**3):.2f}GB\n")
    ERI0 = baf.matlist["REGULAR 2E INTEGRALS"]
    _,_,nr,_,ntot,_,_,_,n4,_,_ = ERI0.labpars
    if(n4 != NBX):
      print(f"ERIs size discrepancy: {n4}!={NBX}")
      exit()
    if(nr != 1):
      print(f"Cannot handle complex AO ERI")
      exit()
    r = ERI0.array.reshape([ntot,nr]) 
    for i in range(n4):
      for j in range(i+1):
        for k in range(i+1):
          if i == k: llim = j + 1
          else: llim = k + 1
          if(ipbc and i>=NB and j>=NB and k>=NB): llim = min(llim,NB)
          for l in range(llim):
            ijkl,_ = qcmio.lind4(-n4,-n4,-n4,n4,i+1,j+1,k+1,l+1)
            doit = abs(r[ijkl,0]) >= 1.e-12
            if(ipbc): doit = doit and (i<NB or j<NB or k<NB or l<NB)
            if doit:
              # print(f"ijkl: {i} {j} {k} {l} {ijkl} {r[ijkl,0]}")
              ERI[i,j,k,l] = r[ijkl,0]
              ERI[j,i,k,l] = r[ijkl,0]
              ERI[i,j,l,k] = r[ijkl,0]
              ERI[j,i,l,k] = r[ijkl,0]
              ERI[k,l,i,j] = r[ijkl,0]
              ERI[l,k,i,j] = r[ijkl,0]
              ERI[k,l,j,i] = r[ijkl,0]
              ERI[l,k,j,i] = r[ijkl,0]
    # for i in np.ndindex(ERI.shape):
    #   print(f" i: {i}")
    #   ERI[i] = baf.matlist["REGULAR 2E INTEGRALS"].get_elemc(*i)
  # print(f" ERI read")
  # exit()
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"Fill ERI\n")
  if(ipbc):
    # ERIpbc = np.zeros((NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB))
    # np.save(f"{scratch}/{molecule}-ERI-AO1",ERIpbc)
    # del ERIpbc
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"Define ERIpbc\n")
    # ERIpbc = np.load(f"{scratch}/{molecule}-ERI-AO1.npy",mmap_mode='r+')
    ERIpbc = np.lib.format.open_memmap(f"{scratch}/{molecule}-ERI-AO1.npy",
                                       mode='w+',shape=(NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB)) 
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"load ERIpbc\n")
    ERI = ERI.reshape((nmtpbc,NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB))
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"reshape ERI\n")
    ERIpbc[:,:,:,:,:,:,:] = ERI[0,:,:,:,:,:,:,:]
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"load ERIpbc\n")
    del ERIpbc
    os.system(f"mv {scratch}/{molecule}-ERI-AO1.npy {scratch}/{molecule}-ERI-AO.npy")
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"overwrite ERI\n")
  del ERI
    
    # print(f" ERI slice")
    # ERI = ERI.reshape((NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB))
    # print(f" ERI reshape 2")
  # for i in np.ndindex(*shapedef):
  #   indx,sign = qcmio.lind4(AOInt0.dimens[0],AOInt0.dimens[1],AOInt0.dimens[2],
  #                           AOInt0.dimens[3],i[3]+1,i[2]+1,i[1]+1,i[0]+1)
  #   if(ipbc):
  #     if(i[0] == 0): ERI[i] = AOInt0.get_elemc(*i)
  #   else:
  #     ERI[i] = AOInt0.get_elemc(*i)
  #   print(f"i : {i}, indx={indx}, sign={sign}, Val = {AOInt0.get_elemc(*i)}")
  # print(f"{lenexp},{nshape},{ndimens},{lennew},{shapedef}\n {AOInt0}")
  # print(f"char, type: {type(char)}, shape: {np.shape(char)}, size: {np.size(char)}\n ")
  # print(f"char:\n {char}")
  # AOInt1 = baf.matlist["REGULAR 2E INTEGRALS"].expand()
  # print(f"Int2E, shape: {AOInt1.shape}, size: {np.size(AOInt1)}\n")
  # exit()
  # AOInt = baf.matlist["REGULAR 2E INTEGRALS"].expand()
  # if(ipbc):
  #   nmtpbc = ipbc[1]
  #   AOInt = AOInt.reshape((nmtpbc,NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB))
  #   AOInt = AOInt[0,:,:,:,:,:,:,:]
  #   AOInt = AOInt.reshape((NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB))
  # return AOInt
  return
  
#########################################################
####### AO -> MO Basis 2e Integral transformation########
#########################################################
#def conMO(molecule, scratch, O, V, NB, ipbc, MOCoef, AOInt):
def conMO(molecule, scratch, O, V, NB, ipbc, MOCoef):
  # AOInt: single-bar 2ERI in AO, Mulliken notation [11|22]
  # MO: double-bar 2ERI in MO, physicist notation <12||12>
  O2 = O*2
  V2 = V*2
  NOrb = O + V
  AOInt = np.load(f"{scratch}/{molecule}-ERI-AO.npy",mmap_mode='r')
  # with open(f"{molecule}.txt","a") as writer:
  #   writer.write(f" AOInt shape: {AOInt.shape}\n {AOInt[:4,:4,:4,:4]}")
  # exit()
  if(ipbc):
    #
    # PBC Fourier transform
    nmtpbc = ipbc[1]
    kp, l_list = fill_kl(ipbc)
    Nkp = len(kp)
    co = np.einsum('k,l->kl',kp,l_list,optimize=True)
    cof = np.cos(co) + 1j*np.sin(co)
    temp = np.lib.format.open_memmap(f"{scratch}/{molecule}-temp.npy",
                                     mode='w+',shape=(Nkp,NB,NB,nmtpbc,NB,nmtpbc,NB),
                                     dtype=complex) 
    temp2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-temp2.npy",
                                      mode='w+',shape=(Nkp,Nkp,NB,NB,NB,nmtpbc,NB),
                                      dtype=complex) 
    AOk = np.lib.format.open_memmap(f"{scratch}/{molecule}-AOk.npy",
                                    mode='w+',shape=(Nkp,Nkp,Nkp,NB,NB,NB,NB),
                                    dtype=complex) 
    temp[:,:,:,:,:,:,:] = np.einsum('hl,albmcnd->habmcnd',cof,AOInt,optimize=True)
    del AOInt
    temp2[:,:,:,:,:,:,:] = np.einsum('km,habmcnd->hkabcnd',np.conjugate(cof),temp,optimize=True)
    AOk[:,:,:,:,:,:,:] = np.einsum('gn,hkabcnd->hkgabcd',cof,temp2,optimize=True)
    del temp, temp2
    os.system(f"rm {scratch}/{molecule}-temp*.npy")
    #
    # AO(k,k')->MO(k,k') transformation
    temp = np.lib.format.open_memmap(f"{scratch}/{molecule}-temp.npy",
                                     mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NB,NB),
                                     dtype=complex) 
    temp2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-temp2.npy",
                                      mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NOrb,NB),
                                      dtype=complex) 
    temp3 = np.lib.format.open_memmap(f"{scratch}/{molecule}-temp3.npy",
                                      mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NOrb,NOrb),
                                      dtype=complex) 
    twoEk = np.lib.format.open_memmap(f"{scratch}/{molecule}-twoEk.npy",
                                      mode='w+',shape=(Nkp,Nkp,Nkp,Nkp,NOrb,NOrb,NOrb,NOrb),
                                      dtype=complex) 
    temp[:,:,:,:,:,:,:] = np.einsum('hbm,hkgamcd->hkgabcd',MOCoef,AOk,optimize=True)
    del AOk
    temp2[:,:,:,:,:,:,:] = np.einsum('kcm,hkgabmd->hkgabcd',np.conjugate(MOCoef),temp,optimize=True)
    del temp
    temp3[:,:,:,:,:,:,:] = np.einsum('gdm,hkgabcm->hkgabcd',MOCoef,temp2,optimize=True)
    del temp2
    twoEk[:,:,:,:,:,:,:,:] = np.einsum('nam,hkgmbcd->nkhgacbd',np.conjugate(MOCoef),temp3,optimize=True)
    del temp3
    os.system(f"rm {scratch}/{molecule}-temp*.npy")
    os.system(f"rm {scratch}/{molecule}-AOk.npy")
    #
    # Form double-bar integrals in physicist notation <12||12>
    MO = np.lib.format.open_memmap(f"{scratch}/{molecule}-MO.npy",mode='w+',
                                   shape=(2*NOrb,2*NOrb,2*NOrb,2*NOrb),dtype=complex) 
    ABCDt = np.lib.format.open_memmap(f"{scratch}/{molecule}-ABCDt.npy",mode='w+',
                                      shape=(Nkp,V2,Nkp,V2,Nkp,V2,Nkp,V2),dtype=complex) 
    IABCt = np.lib.format.open_memmap(f"{scratch}/{molecule}-IABCt.npy",mode='w+',
                                      shape=(Nkp,O2,Nkp,V2,Nkp,V2,Nkp,V2),dtype=complex) 
    IJABt = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJABt.npy",mode='w+',
                                      shape=(Nkp,O2,Nkp,O2,Nkp,V2,Nkp,V2),dtype=complex) 
    IABJt = np.lib.format.open_memmap(f"{scratch}/{molecule}-IABJt.npy",mode='w+',
                                      shape=(Nkp,O2,Nkp,V2,Nkp,V2,Nkp,O2),dtype=complex) 
    IJKLt = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJKLt.npy",mode='w+',
                                      shape=(Nkp,O2,Nkp,O2,Nkp,O2,Nkp,O2),dtype=complex) 
    IJKAt = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJKAt.npy",mode='w+',
                                      shape=(Nkp,O2,Nkp,O2,Nkp,O2,Nkp,V2),dtype=complex)
    # ABCDt = np.lib.format.open_memmap(f"{scratch}/{molecule}-ABCDt.npy",
    #                                   mode='w+',shape=(Nkp,Nkp,Nkp,Nkp,V2,V2,V2,V2),
    #                                   dtype=complex) 
    # IABCt = np.lib.format.open_memmap(f"{scratch}/{molecule}-IABCt.npy",
    #                                   mode='w+',shape=(Nkp,Nkp,Nkp,Nkp,O2,V2,V2,V2),
    #                                   dtype=complex) 
    # IJABt = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJABt.npy",
    #                                   mode='w+',shape=(Nkp,Nkp,Nkp,Nkp,O2,O2,V2,V2),
    #                                   dtype=complex) 
    # IABJt = np.lib.format.open_memmap(f"{scratch}/{molecule}-IABJt.npy",
    #                                   mode='w+',shape=(Nkp,Nkp,Nkp,Nkp,O2,V2,V2,O2),
    #                                   dtype=complex) 
    # IJKLt = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJKLt.npy",
    #                                   mode='w+',shape=(Nkp,Nkp,Nkp,Nkp,O2,O2,O2,O2),
    #                                   dtype=complex) 
    # IJKAt = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJKAt.npy",
    #                                   mode='w+',shape=(Nkp,Nkp,Nkp,Nkp,O2,O2,O2,V2),
    #                                   dtype=complex) 
    # MO = np.zeros((2*NOrb,2*NOrb,2*NOrb,2*NOrb),dtype=complex)
    # IJAB = np.zeros((Nkp,Nkp,Nkp,Nkp,O2,O2,V2,V2),dtype=complex)
    # IJKL = np.zeros((Nkp,Nkp,Nkp,Nkp,O2,O2,O2,O2),dtype=complex)
    # IJKA = np.zeros((Nkp,Nkp,Nkp,Nkp,O2,O2,O2,V2),dtype=complex)
    # IABJ = np.zeros((Nkp,Nkp,Nkp,Nkp,O2,V2,V2,O2),dtype=complex)
    # IABC = np.zeros((Nkp,Nkp,Nkp,Nkp,O2,V2,V2,V2),dtype=complex)
    # ABCD = np.zeros((Nkp,Nkp,Nkp,Nkp,V2,V2,V2,V2),dtype=complex)
    pi2 = round(2*np.pi,10)
    O2k = O2*Nkp
    V2k = V2*Nkp
    Nksum = 0
    for n in range(Nkp):
      for k in range(Nkp):
        for h in range(Nkp):
          for g in range(Nkp):
            kn = kp[n]
            kh = kp[h]
            kk = kp[k]
            kg = kp[g]
            ktot = round(kn-kh+kk-kg,10)
            if(abs(ktot) < 1e-8 or abs(ktot%pi2) < 1e-8): 
              # Form double-bar integrals <12||12>. Spin blocks are stored as follows:
              # aaaa: Coulomb - Exchange
              # bbbb: Coulomb - Exchange
              # baba: Coulomb 
              # abab: Coulomb 
              # baab: - Exchange
              # abba: - Exchange
              #
              Nksum += 1
              MO[:NOrb,:NOrb,:NOrb,:NOrb] = np.copy(twoEk[n,k,h,g,:,:,:,:])
              MO[:NOrb,:NOrb,:NOrb,:NOrb] -= np.transpose(twoEk[n,k,g,h,:,:,:,:],axes=(0,1,3,2))
              MO[NOrb:,NOrb:,NOrb:,NOrb:] = np.copy(MO[:NOrb,:NOrb,:NOrb,:NOrb])
              MO[NOrb:,:NOrb,NOrb:,:NOrb] = np.copy(twoEk[n,k,h,g,:,:,:,:])
              MO[:NOrb,NOrb:,:NOrb,NOrb:] = np.copy(MO[NOrb:,:NOrb,NOrb:,:NOrb])
              MO[NOrb:,:NOrb,:NOrb,NOrb:] = -np.transpose(twoEk[n,k,g,h,:,:,:,:],axes=(0,1,3,2))
              MO[:NOrb,NOrb:,NOrb:,:NOrb] = np.copy(MO[NOrb:,:NOrb,:NOrb,NOrb:])
              #IJAB
              IJABt[n,:O,k,:O,h,:V,g,:V] = np.copy(MO[:O,:O,O:NOrb,O:NOrb])
              IJABt[n,O:,k,O:,h,V:,g,V:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
              IJABt[n,O:,k,:O,h,V:,g,:V] = np.copy(MO[NOrb:O+NOrb,:O,O+NOrb:2*NOrb,O:NOrb])
              IJABt[n,:O,k,O:,h,:V,g,V:] = np.copy(MO[:O,NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb])
              IJABt[n,O:,k,:O,h,:V,g,V:] = np.copy(MO[NOrb:O+NOrb,:O,O:NOrb,O+NOrb:2*NOrb])
              IJABt[n,:O,k,O:,h,V:,g,:V] = np.copy(MO[:O,NOrb:O+NOrb,O+NOrb:2*NOrb,O:NOrb])
              # IJABt[n,k,h,g,:O,:O,:V,:V] = np.copy(MO[:O,:O,O:NOrb,O:NOrb])
              # IJABt[n,k,h,g,O:,O:,V:,V:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
              # IJABt[n,k,h,g,O:,:O,V:,:V] = np.copy(MO[NOrb:O+NOrb,:O,O+NOrb:2*NOrb,O:NOrb])
              # IJABt[n,k,h,g,:O,O:,:V,V:] = np.copy(MO[:O,NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb])
              # IJABt[n,k,h,g,O:,:O,:V,V:] = np.copy(MO[NOrb:O+NOrb,:O,O:NOrb,O+NOrb:2*NOrb])
              # IJABt[n,k,h,g,:O,O:,V:,:V] = np.copy(MO[:O,NOrb:O+NOrb,O+NOrb:2*NOrb,O:NOrb])
              #
              # IJKL
              IJKLt[n,:O,k,:O,h,:O,g,:O] = np.copy(MO[:O,:O,:O,:O])
              IJKLt[n,O:,k,O:,h,O:,g,O:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb])
              IJKLt[n,O:,k,:O,h,O:,g,:O] = np.copy(MO[NOrb:O+NOrb,:O,NOrb:O+NOrb,:O])
              IJKLt[n,:O,k,O:,h,:O,g,O:] = np.copy(MO[:O,NOrb:O+NOrb,:O,NOrb:O+NOrb])
              IJKLt[n,O:,k,:O,h,:O,g,O:] = np.copy(MO[NOrb:O+NOrb,:O,:O,NOrb:O+NOrb])
              IJKLt[n,:O,k,O:,h,O:,g,:O] = np.copy(MO[:O,NOrb:O+NOrb,NOrb:O+NOrb,:O])
              # IJKLt[n,k,h,g,:O,:O,:O,:O] = np.copy(MO[:O,:O,:O,:O])
              # IJKLt[n,k,h,g,O:,O:,O:,O:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb])
              # IJKLt[n,k,h,g,O:,:O,O:,:O] = np.copy(MO[NOrb:O+NOrb,:O,NOrb:O+NOrb,:O])
              # IJKLt[n,k,h,g,:O,O:,:O,O:] = np.copy(MO[:O,NOrb:O+NOrb,:O,NOrb:O+NOrb])
              # IJKLt[n,k,h,g,O:,:O,:O,O:] = np.copy(MO[NOrb:O+NOrb,:O,:O,NOrb:O+NOrb])
              # IJKLt[n,k,h,g,:O,O:,O:,:O] = np.copy(MO[:O,NOrb:O+NOrb,NOrb:O+NOrb,:O])
              #
              # IJKA
              IJKAt[n,:O,k,:O,h,:O,g,:V] = np.copy(MO[:O,:O,:O,O:NOrb])
              IJKAt[n,O:,k,O:,h,O:,g,V:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb,O+NOrb:2*NOrb])
              IJKAt[n,O:,k,:O,h,O:,g,:V] = np.copy(MO[NOrb:O+NOrb,:O,NOrb:O+NOrb,O:NOrb])
              IJKAt[n,:O,k,O:,h,:O,g,V:] = np.copy(MO[:O,NOrb:O+NOrb,:O,O+NOrb:2*NOrb])
              IJKAt[n,O:,k,:O,h,:O,g,V:] = np.copy(MO[NOrb:O+NOrb,:O,:O,O+NOrb:2*NOrb])
              IJKAt[n,:O,k,O:,h,O:,g,:V] = np.copy(MO[:O,NOrb:O+NOrb,NOrb:O+NOrb,O:NOrb])
              # IJKAt[n,k,h,g,:O,:O,:O,:V] = np.copy(MO[:O,:O,:O,O:NOrb])
              # IJKAt[n,k,h,g,O:,O:,O:,V:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb,O+NOrb:2*NOrb])
              # IJKAt[n,k,h,g,O:,:O,O:,:V] = np.copy(MO[NOrb:O+NOrb,:O,NOrb:O+NOrb,O:NOrb])
              # IJKAt[n,k,h,g,:O,O:,:O,V:] = np.copy(MO[:O,NOrb:O+NOrb,:O,O+NOrb:2*NOrb])
              # IJKAt[n,k,h,g,O:,:O,:O,V:] = np.copy(MO[NOrb:O+NOrb,:O,:O,O+NOrb:2*NOrb])
              # IJKAt[n,k,h,g,:O,O:,O:,:V] = np.copy(MO[:O,NOrb:O+NOrb,NOrb:O+NOrb,O:NOrb])
              #
              # IABJ
              IABJt[n,:O,k,:V,h,:V,g,:O] = np.copy(MO[:O,O:NOrb,O:NOrb,:O])
              IABJt[n,O:,k,V:,h,V:,g,O:] = np.copy(MO[NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,NOrb:O+NOrb])
              IABJt[n,O:,k,:V,h,V:,g,:O] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb,:O])
              IABJt[n,:O,k,V:,h,:V,g,O:] = np.copy(MO[:O,O+NOrb:2*NOrb,O:NOrb,NOrb:O+NOrb])
              IABJt[n,O:,k,:V,h,:V,g,O:] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O:NOrb,NOrb:O+NOrb])
              IABJt[n,:O,k,V:,h,V:,g,:O] = np.copy(MO[:O,O+NOrb:2*NOrb,O+NOrb:2*NOrb,:O])
              # IABJt[n,k,h,g,:O,:V,:V,:O] = np.copy(MO[:O,O:NOrb,O:NOrb,:O])
              # IABJt[n,k,h,g,O:,V:,V:,O:] = np.copy(MO[NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,NOrb:O+NOrb])
              # IABJt[n,k,h,g,O:,:V,V:,:O] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb,:O])
              # IABJt[n,k,h,g,:O,V:,:V,O:] = np.copy(MO[:O,O+NOrb:2*NOrb,O:NOrb,NOrb:O+NOrb])
              # IABJt[n,k,h,g,O:,:V,:V,O:] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O:NOrb,NOrb:O+NOrb])
              # IABJt[n,k,h,g,:O,V:,V:,:O] = np.copy(MO[:O,O+NOrb:2*NOrb,O+NOrb:2*NOrb,:O])
              #
              # IABC
              IABCt[n,:O,k,:V,h,:V,g,:V] = np.copy(MO[:O,O:NOrb,O:NOrb,O:NOrb])
              IABCt[n,O:,k,V:,h,V:,g,V:] = np.copy(MO[NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
              IABCt[n,O:,k,:V,h,V:,g,:V] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb,O:NOrb])
              IABCt[n,:O,k,V:,h,:V,g,V:] = np.copy(MO[:O,O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb])
              IABCt[n,O:,k,:V,h,:V,g,V:] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O:NOrb,O+NOrb:2*NOrb])
              IABCt[n,:O,k,V:,h,V:,g,:V] = np.copy(MO[:O,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O:NOrb])
              # IABCt[n,k,h,g,:O,:V,:V,:V] = np.copy(MO[:O,O:NOrb,O:NOrb,O:NOrb])
              # IABCt[n,k,h,g,O:,V:,V:,V:] = np.copy(MO[NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
              # IABCt[n,k,h,g,O:,:V,V:,:V] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb,O:NOrb])
              # IABCt[n,k,h,g,:O,V:,:V,V:] = np.copy(MO[:O,O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb])
              # IABCt[n,k,h,g,O:,:V,:V,V:] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O:NOrb,O+NOrb:2*NOrb])
              # IABCt[n,k,h,g,:O,V:,V:,:V] = np.copy(MO[:O,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O:NOrb])
              #
              # ABCD
              ABCDt[n,:V,k,:V,h,:V,g,:V] = np.copy(MO[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
              ABCDt[n,V:,k,V:,h,V:,g,V:] = np.copy(MO[O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
              ABCDt[n,V:,k,:V,h,V:,g,:V] = np.copy(MO[O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb,O:NOrb])
              ABCDt[n,:V,k,V:,h,:V,g,V:] = np.copy(MO[O:NOrb,O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb])
              ABCDt[n,V:,k,:V,h,:V,g,V:] = np.copy(MO[O+NOrb:2*NOrb,O:NOrb,O:NOrb,O+NOrb:2*NOrb])
              ABCDt[n,:V,k,V:,h,V:,g,:V] = np.copy(MO[O:NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O:NOrb])
              # ABCDt[n,k,h,g,:V,:V,:V,:V] = np.copy(MO[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
              # ABCDt[n,k,h,g,V:,V:,V:,V:] = np.copy(MO[O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
              # ABCDt[n,k,h,g,V:,:V,V:,:V] = np.copy(MO[O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb,O:NOrb])
              # ABCDt[n,k,h,g,:V,V:,:V,V:] = np.copy(MO[O:NOrb,O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb])
              # ABCDt[n,k,h,g,V:,:V,:V,V:] = np.copy(MO[O+NOrb:2*NOrb,O:NOrb,O:NOrb,O+NOrb:2*NOrb])
              # ABCDt[n,k,h,g,:V,V:,V:,:V] = np.copy(MO[O:NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O:NOrb])
    del MO, twoEk
    os.system(f"rm {scratch}/{molecule}-MO.npy")
    os.system(f"rm {scratch}/{molecule}-twoEk.npy")
    # ABCDt2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-ABCDt2.npy",
    #                                    mode='w+',shape=(Nkp,V2,Nkp,V2,Nkp,V2,Nkp,V2),
    #                                    dtype=complex) 
    # ABCDt2[:,:,:,:,:,:,:,:] = np.transpose(ABCDt,axes=(0,4,1,5,2,6,3,7))
    # del ABCDt
    ABCD = np.lib.format.open_memmap(f"{scratch}/{molecule}-ABCD.npy",
                                     mode='w+',shape=(V2k,V2k,V2k,V2k),
                                     dtype=complex) 
    ABCD[:,:,:,:] = ABCDt.reshape((V2k,V2k,V2k,V2k))
    # np.save(f"{scratch}/{molecule}-ABCD",ABCD)
    del ABCD, ABCDt
    os.system(f"rm {scratch}/{molecule}-ABCDt.npy")
    # IJABt2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJABt2.npy",
    #                                    mode='w+',shape=(Nkp,O2,Nkp,O2,Nkp,V2,Nkp,V2),
    #                                    dtype=complex) 
    # IJABt2[:,:,:,:,:,:,:,:] = np.transpose(IJABt,axes=(0,4,1,5,2,6,3,7))
    # del IJABt
    IJAB = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJAB.npy",
                                     mode='w+',shape=(O2k,O2k,V2k,V2k),
                                     dtype=complex) 
    IJAB[:,:,:,:] = IJABt.reshape((O2k,O2k,V2k,V2k))
    del IJAB, IJABt
    os.system(f"rm {scratch}/{molecule}-IJABt.npy")
    # IJAB = np.transpose(IJAB,axes=(0,4,1,5,2,6,3,7))
    # IJAB = IJAB.reshape((O2k,O2k,V2k,V2k))
    # IJKLt2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJKLt2.npy",
    #                                    mode='w+',shape=(Nkp,O2,Nkp,O2,Nkp,O2,Nkp,O2),
    #                                    dtype=complex) 
    # IJKLt2[:,:,:,:,:,:,:,:] = np.transpose(IJKLt,axes=(0,4,1,5,2,6,3,7))
    # del IJKLt
    IJKL = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJKL.npy",
                                     mode='w+',shape=(O2k,O2k,O2k,O2k),
                                     dtype=complex) 
    IJKL[:,:,:,:] = IJKLt.reshape((O2k,O2k,O2k,O2k))
    del IJKL, IJKLt
    os.system(f"rm {scratch}/{molecule}-IJKLt.npy")
    # IJKL = np.transpose(IJKL,axes=(0,4,1,5,2,6,3,7))
    # IJKL = IJKL.reshape((O2k,O2k,O2k,O2k))
    # IJKAt2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJKAt2.npy",
    #                                    mode='w+',shape=(Nkp,O2,Nkp,O2,Nkp,O2,Nkp,V2),
    #                                    dtype=complex) 
    # IJKAt2[:,:,:,:,:,:,:,:] = np.transpose(IJKAt,axes=(0,4,1,5,2,6,3,7))
    # del IJKAt
    IJKA = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJKA.npy",
                                     mode='w+',shape=(O2k,O2k,O2k,V2k),
                                     dtype=complex) 
    IJKA[:,:,:,:] = IJKAt.reshape((O2k,O2k,O2k,V2k))
    del IJKA, IJKAt
    os.system(f"rm {scratch}/{molecule}-IJKAt.npy")
    # IJKA = np.transpose(IJKA,axes=(0,4,1,5,2,6,3,7))
    # IJKA = IJKA.reshape((O2k,O2k,O2k,V2k))
    # IABJt2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-IABJt2.npy",
    #                                    mode='w+',shape=(Nkp,O2,Nkp,V2,Nkp,V2,Nkp,O2),
    #                                    dtype=complex) 
    # IABJt2[:,:,:,:,:,:,:,:] = np.transpose(IABJt,axes=(0,4,1,5,2,6,3,7))
    # del IABJt
    IABJ = np.lib.format.open_memmap(f"{scratch}/{molecule}-IABJ.npy",
                                     mode='w+',shape=(O2k,V2k,V2k,O2k),
                                     dtype=complex) 
    IABJ[:,:,:,:] = IABJt.reshape((O2k,V2k,V2k,O2k))
    del IABJ, IABJt
    os.system(f"rm {scratch}/{molecule}-IABJt.npy")
    # IABJ = np.transpose(IABJ,axes=(0,4,1,5,2,6,3,7))
    # IABJ = IABJ.reshape((O2k,V2k,V2k,O2k))
    # IABCt2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-IABCt2.npy",
    #                                    mode='w+',shape=(Nkp,O2,Nkp,V2,Nkp,V2,Nkp,V2),
    #                                    dtype=complex) 
    # IABCt2[:,:,:,:,:,:,:,:] = np.transpose(IABCt,axes=(0,4,1,5,2,6,3,7))
    # del IABCt
    IABC = np.lib.format.open_memmap(f"{scratch}/{molecule}-IABC.npy",
                                     mode='w+',shape=(O2k,V2k,V2k,V2k),
                                     dtype=complex) 
    IABC[:,:,:,:] = IABCt.reshape((O2k,V2k,V2k,V2k))
    del IABC, IABCt
    os.system(f"rm {scratch}/{molecule}-IABCt.npy")
  else:
    #
    # AO->MO transformation
    temp = np.einsum('im,mjkl->ijkl',MOCoef,AOInt,optimize=True)
    del AOInt
    temp2 = np.einsum('jm,imkl->ijkl',MOCoef,temp,optimize=True)
    del temp
    temp = np.einsum('km,ijml->ijkl',MOCoef,temp2,optimize=True)
    del temp2
    twoE = np.einsum('lm,ijkm->ikjl',MOCoef,temp,optimize=True)
    del temp
    #
    # ABCD
    ABCD = np.lib.format.open_memmap(f"{scratch}/{molecule}-ABCD.npy",
                                     mode='w+',shape=(V2,V2,V2,V2))
    # ABCD = np.zeros((V2,V2,V2,V2))
    ABCD[:V,:V,:V,:V] = np.copy(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
    ABCD[:V,:V,:V,:V] -= np.transpose(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb],axes=(0,1,3,2))
    ABCD[V:,V:,V:,V:] = np.copy(ABCD[:V,:V,:V,:V])
    ABCD[V:,:V,V:,:V] = np.copy(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
    ABCD[:V,V:,:V,V:] = np.copy(ABCD[V:,:V,V:,:V])
    ABCD[V:,:V,:V,V:] = -np.transpose(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb],axes=(0,1,3,2))
    ABCD[:V,V:,V:,:V] = np.copy(ABCD[V:,:V,:V,V:])
    # np.save(f"{scratch}/{molecule}-ABCD",ABCD)
    del ABCD
    #
    # IJAB
    IJAB = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJAB.npy",
                                     mode='w+',shape=(O2,O2,V2,V2))
    # IJAB = np.zeros((O2,O2,V2,V2))
    IJAB[:O,:O,:V,:V] = np.copy(twoE[:O,:O,O:NOrb,O:NOrb])
    IJAB[:O,:O,:V,:V] -= np.transpose(twoE[:O,:O,O:NOrb,O:NOrb],axes=(0,1,3,2))
    IJAB[O:,O:,V:,V:] = np.copy(IJAB[:O,:O,:V,:V])
    IJAB[O:,:O,V:,:V] = np.copy(twoE[:O,:O,O:NOrb,O:NOrb])
    IJAB[:O,O:,:V,V:] = np.copy(IJAB[O:,:O,V:,:V])    
    IJAB[O:,:O,:V,V:] = -np.transpose(twoE[:O,:O,O:NOrb,O:NOrb],axes=(0,1,3,2))
    IJAB[:O,O:,V:,:V] = np.copy(IJAB[O:,:O,:V,V:])
    del IJAB
    #
    # IJKL
    IJKL = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJKL.npy",
                                     mode='w+',shape=(O2,O2,O2,O2))
    # IJKL = np.zeros((O2,O2,O2,O2))
    IJKL[:O,:O,:O,:O] = np.copy(twoE[:O,:O,:O,:O])
    IJKL[:O,:O,:O,:O] -= np.transpose(twoE[:O,:O,:O,:O],axes=(0,1,3,2))
    IJKL[O:,O:,O:,O:] = np.copy(IJKL[:O,:O,:O,:O])
    IJKL[O:,:O,O:,:O] = np.copy(twoE[:O,:O,:O,:O])
    IJKL[:O,O:,:O,O:] = np.copy(IJKL[O:,:O,O:,:O])
    IJKL[O:,:O,:O,O:] = -np.transpose(twoE[:O,:O,:O,:O],axes=(0,1,3,2))
    IJKL[:O,O:,O:,:O] = np.copy(IJKL[O:,:O,:O,O:])
    del IJKL
    #
    # IJKA
    IJKA = np.lib.format.open_memmap(f"{scratch}/{molecule}-IJKA.npy",
                                     mode='w+',shape=(O2,O2,O2,V2))
    # IJKA = np.zeros((O2,O2,O2,V2))
    IJKA[:O,:O,:O,:V] = np.copy(twoE[:O,:O,:O,O:NOrb])
    IJKA[:O,:O,:O,:V] -= np.transpose(twoE[:O,:O,O:NOrb,:O],axes=(0,1,3,2))
    IJKA[O:,O:,O:,V:] = np.copy(IJKA[:O,:O,:O,:V])
    IJKA[O:,:O,O:,:V] = np.copy(twoE[:O,:O,:O,O:NOrb])
    IJKA[:O,O:,:O,V:] = np.copy(IJKA[O:,:O,O:,:V])
    IJKA[O:,:O,:O,V:] = -np.transpose(twoE[:O,:O,O:NOrb,:O],axes=(0,1,3,2))
    IJKA[:O,O:,O:,:V] = np.copy(IJKA[O:,:O,:O,V:])
    del IJKA
    #
    # IABJ
    IABJ = np.lib.format.open_memmap(f"{scratch}/{molecule}-IABJ.npy",
                                     mode='w+',shape=(O2,V2,V2,O2))
    # IABJ = np.zeros((O2,V2,V2,O2))
    IABJ[:O,:V,:V,:O] = np.copy(twoE[:O,O:NOrb,O:NOrb,:O])
    IABJ[:O,:V,:V,:O] -= np.transpose(twoE[:O,O:NOrb,:O,O:NOrb],axes=(0,1,3,2))
    IABJ[O:,V:,V:,O:] = np.copy(IABJ[:O,:V,:V,:O])
    IABJ[O:,:V,V:,:O] = np.copy(twoE[:O,O:NOrb,O:NOrb,:O])
    IABJ[:O,V:,:V,O:] = np.copy(IABJ[O:,:V,V:,:O])
    IABJ[O:,:V,:V,O:] = -np.transpose(twoE[:O,O:NOrb,:O,O:NOrb],axes=(0,1,3,2))
    IABJ[:O,V:,V:,:O] = np.copy(IABJ[O:,:V,:V,O:])
    del IABJ
    #
    # IABC
    IABC = np.lib.format.open_memmap(f"{scratch}/{molecule}-IABC.npy",
                                     mode='w+',shape=(O2,V2,V2,V2))
    # IABC = np.zeros((O2,V2,V2,V2))
    IABC[:O,:V,:V,:V] = np.copy(twoE[:O,O:NOrb,O:NOrb,O:NOrb])
    IABC[:O,:V,:V,:V] -= np.transpose(twoE[:O,O:NOrb,O:NOrb,O:NOrb],axes=(0,1,3,2))
    IABC[O:,V:,V:,V:] = np.copy(IABC[:O,:V,:V,:V])
    IABC[O:,:V,V:,:V] = np.copy(twoE[:O,O:NOrb,O:NOrb,O:NOrb])
    IABC[:O,V:,:V,V:] = np.copy(IABC[O:,:V,V:,:V])
    IABC[O:,:V,:V,V:] = -np.transpose(twoE[:O,O:NOrb,O:NOrb,O:NOrb],axes=(0,1,3,2))
    IABC[:O,V:,V:,:V] = np.copy(IABC[O:,:V,:V,V:])
    del IABC
    # del MO
    del twoE
  return 
  # return IJKL, IABC, IJAB, IJKA, IABJ

#########################################################
# Get perturbation integrals and return them in MO basis
#########################################################
def getPert(O, V, NB, ipbc, MOCoef, Fock, pert_type, mol):
  NBX = NB
  O2 = 2*O
  V2 = 2*V
  NOrb = O + V
  O2k = O2
  V2k = V2
  ntt = NB*(NB+1)//2
  nttx = ntt
  if(ipbc):
    kp, l_list = fill_kl(ipbc)
    Nkp = len(kp)
    nmtpbc = ipbc[1]
    NBX = NB*nmtpbc
    O2k = O2*Nkp
    V2k = V2*Nkp
    nttx = ntt*nmtpbc
  #
  with open(f"{mol}.txt","a") as writer:
    writer.write(f"Reading perturbation {pert_type}\n")
  if(pert_type == "DipE"):
    NP1 = 3
    NP2 = 3
    NP3 = 0
    NP = 3
    AOPert = np.zeros((NP*nttx))
    if(f"{mol}_txts/dipole_r.txt"):
      with open(f"{mol}_txts/dipole_r.txt","r") as reader:
        text=[]
        for line in reader:
          text.append(line.split())
      # Remove parentheses
      for i in range(len(text)):
        for j in range(len(text[i])):
          text[i][j] = text[i][j].replace("[","")
          text[i][j] = text[i][j].replace("]","")
      # Remove empty slots
      for i in range(len(text)):
        text[i][:] = [x for x in text[i] if x]
      ind = 0
      for i in range(len(text)):
        for j in range(len(text[i])):
          AOPert[ind] = float(text[i][j])
          ind += 1
      AOPert = AOPert.reshape(NP,nttx)
    else:
      print(f" No electric dipole integrals found\n")
      exit()
  elif(pert_type == "DipEV"):
    NP1 = 3
    NP2 = 3
    NP3 = 0
    NP = 3
    AOPert = np.zeros((NP*nttx))
    if(f"{mol}_txts/dipole_v.txt"):
      with open(f"{mol}_txts/dipole_v.txt","r") as reader:
        text=[]
        for line in reader:
          text.append(line.split())
      # Remove parentheses
      for i in range(len(text)):
        for j in range(len(text[i])):
          text[i][j] = text[i][j].replace("[","")
          text[i][j] = text[i][j].replace("]","")
      # Remove empty slots
      for i in range(len(text)):
        text[i][:] = [x for x in text[i] if x]
      ind = 0
      for i in range(len(text)):
        for j in range(len(text[i])):
          AOPert[ind] = float(text[i][j])
          ind += 1
      AOPert = AOPert.reshape(NP,nttx)
    else:
      print(f" No velocity electric dipole integrals found\n")
      exit()
  elif(pert_type == "OR_V"):
    NP1 = 3
    NP2 = 3
    NP3 = 0
    NP = 6
    AOPert = np.zeros((NP*nttx))
    if(f"{mol}_txts/dipole_v.txt"):
      with open(f"{mol}_txts/dipole_v.txt","r") as reader:
        text=[]
        for line in reader:
          text.append(line.split())
      # Remove parentheses
      for i in range(len(text)):
        for j in range(len(text[i])):
          text[i][j] = text[i][j].replace("[","")
          text[i][j] = text[i][j].replace("]","")
      # Remove empty slots
      for i in range(len(text)):
        text[i][:] = [x for x in text[i] if x]
      ind = 0
      for i in range(len(text)):
        for j in range(len(text[i])):
          AOPert[ind] = float(text[i][j])
          ind += 1
    else:
      print(f" No velocity electric dipole integrals found\n")
      exit()
    if(f"{mol}_txts/magnetic.txt"):
      with open(f"{mol}_txts/magnetic.txt","r") as reader:
        text=[]
        for line in reader:
          text.append(line.split())
      # Remove parentheses
      for i in range(len(text)):
        for j in range(len(text[i])):
          text[i][j] = text[i][j].replace("[","")
          text[i][j] = text[i][j].replace("]","")
      # Remove empty slots
      for i in range(len(text)):
        text[i][:] = [x for x in text[i] if x]
      for i in range(len(text)):
        for j in range(len(text[i])):
          AOPert[ind] = float(text[i][j])
          ind += 1
    AOPert = AOPert.reshape(NP,nttx)
  elif(pert_type == "FullOR_V"):
    NP1 = 3
    NP2 = 3
    NP3 = 6
    NP = 12
    AOPert = np.zeros((NP*nttx))
    # Read electric dipole V integrals (Del)
    if(f"{mol}_txts/dipole_v.txt"):
      with open(f"{mol}_txts/dipole_v.txt","r") as reader:
        text=[]
        for line in reader:
          text.append(line.split())
      # Remove parentheses
      for i in range(len(text)):
        for j in range(len(text[i])):
          text[i][j] = text[i][j].replace("[","")
          text[i][j] = text[i][j].replace("]","")
      # Remove empty slots
      for i in range(len(text)):
        text[i][:] = [x for x in text[i] if x]
      ind = 0
      for i in range(len(text)):
        for j in range(len(text[i])):
          AOPert[ind] = float(text[i][j])
          ind += 1
    else:
      print(f" No velocity electric dipole integrals found\n")
      exit()
    # Read magnetic dipole integrals (r x Del)
    if(f"{mol}_txts/magnetic.txt"):
      with open(f"{mol}_txts/magnetic.txt","r") as reader:
        text=[]
        for line in reader:
          text.append(line.split())
      # Remove parentheses
      for i in range(len(text)):
        for j in range(len(text[i])):
          text[i][j] = text[i][j].replace("[","")
          text[i][j] = text[i][j].replace("]","")
      # Remove empty slots
      for i in range(len(text)):
        text[i][:] = [x for x in text[i] if x]
      for i in range(len(text)):
        for j in range(len(text[i])):
          AOPert[ind] = float(text[i][j])
          ind += 1
    else:
      print(f" No velocity electric dipole integrals found\n")
      exit()
    # Read electric quadrupole integrals ((r Del + Del r)/2)
    if(f"{mol}_txts/quadrupole_v.txt"):
      with open(f"{mol}_txts/quadrupole_v.txt","r") as reader:
        text=[]
        for line in reader:
          text.append(line.split())
      # Remove parentheses
      for i in range(len(text)):
        for j in range(len(text[i])):
          text[i][j] = text[i][j].replace("[","")
          text[i][j] = text[i][j].replace("]","")
      # Remove empty slots
      for i in range(len(text)):
        text[i][:] = [x for x in text[i] if x]
      for i in range(len(text)):
        for j in range(len(text[i])):
          AOPert[ind] = float(text[i][j])
          ind += 1
    else:
      print(f" No velocity electric dipole integrals found\n")
      exit()
    AOPert = AOPert.reshape(NP,nttx)
  else:
    print(f" Perturbation ",pert_type," is not available")
    exit()
  if(ipbc):
    # PBC case
    #
    if(pert_type == "DipE" or pert_type == "FullOR_V"):
      # For the length gauge electric dipole, for the magnetic dipole
      # and electric quadrupole, we need the translation vector and to
      # form the U matrix
      #
      # Read translation vector
      if(f"{mol}_txts/tv.txt"):
        with open(f"{mol}_txts/tv.txt","r") as reader:
          text=[]
          for line in reader:
            text.append(line.split())
        tv = np.zeros((3))
        tv[0] = float(text[0][0])
        tv[1] = float(text[0][1])
        tv[2] = float(text[0][2])
        # Convert to Bohr
        bohr_radius = physical_constants["Bohr radius"][0]
        tv = np.array(tv)*angstrom / bohr_radius
      else:
        print(f" Translation vector is not available")
        exit()
      #
      # dF/dk = F'
      FockDk = getFock(mol,O,V,NB,ipbc,"MO",True,MOCoef)
      #
      # dS/dk = S'
      OvlDk = getOvl(mol,O,V,NB,ipbc,"MO",True,MOCoef)
      #
      # Form i(U + 1/2S')
      OrbE = np.diag(Fock.real)
      NOrb2k = NOrb*2*Nkp
      if(len(OrbE)!=NOrb2k):
        print(f"Mismatch in the number of orbital energies: {NOrb2k} != {len(OrbE)}")
        exit()
      DE = DEk(1,NOrb2k,OrbE)
      UMat = FockDk - np.einsum('ij,j->ij',OvlDk,OrbE,optimize=True)
      UMat /= -DE
      if(pert_type == "DipE"):
        UMat += 0.5*OvlDk
        # The diagonal of this matrix is 0
        np.fill_diagonal(UMat,0)
      elif(pert_type == "FullOR_V"):
        # The diagonal of this matrix is -S'/2
        np.fill_diagonal(UMat,-np.diag(OvlDk)/2)
      UMat = UMat*1j
      del FockDk, OvlDk, OrbE, DE
    # Now form the perturbation matrices in MO(k) basis
    X_ij = np.zeros((NP,Nkp,Nkp,O2,O2),dtype=complex)
    X_ia = np.zeros((NP,Nkp,Nkp,O2,V2),dtype=complex)
    X_ab = np.zeros((NP,Nkp,Nkp,V2,V2),dtype=complex)
    AOPert = AOPert.reshape((NP,nmtpbc,ntt))
    save_p = []
    if(pert_type == "FullOR_V"):
      save_p = np.zeros((NP1,Nkp*NOrb*2,Nkp*NOrb*2),dtype=complex)
    for n in range (NP):
      Pert_k_lt = fourier("Dir",ipbc,AOPert[n,:,:],False)
      if(pert_type == "DipE"):
        # Electric dipole length gauge
        PertA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Pert_k_lt)
      elif(pert_type == "DipEV" or pert_type == "FullOR_V"):
        # Electric dipole velocity gauge
        PertA = basis_tran("Dir",True,False,"AHer",NB,Nkp,MOCoef,Pert_k_lt)
      Pert = np.zeros((Nkp,NOrb*2,Nkp,NOrb*2),dtype=complex)
      for k in range(Nkp):
        # Fill out the alpha and beta blocks
        # oa-oa
        Pert[k,:O,k,:O] = PertA[k,:O,:O]
        # ob-ob
        Pert[k,O:2*O,k,O:2*O] = PertA[k,:O,:O]
        # va-va
        Pert[k,2*O:2*O+V,k,2*O:2*O+V] = PertA[k,O:,O:]
        # vb-vb
        Pert[k,2*O+V:,k,2*O+V:] = PertA[k,O:,O:]
        # oa-va
        Pert[k,:O,k,2*O:2*O+V] = PertA[k,:O,O:]
        # va-oa
        Pert[k,2*O:2*O+V,k,:O] = PertA[k,O:,:O]
        # ob-vb
        Pert[k,O:2*O,k,2*O+V:] = PertA[k,:O,O:]
        # vb-ob
        Pert[k,2*O+V:,k,O:2*O] = PertA[k,O:,:O]
      if(pert_type == "DipE"):
        # Electric dipole length gauge
        # Add UMat contribution
        Pert = Pert.reshape((Nkp*NOrb*2,Nkp*NOrb*2))
        Pert -= UMat*tv[n]
        Pert = Pert.reshape((Nkp,NOrb*2,Nkp,NOrb*2))
      elif(pert_type == "FullOR_V"):
        # Magnetic dipole and electric quadrupole
        # Add UMat contribution
        Pert = Pert.reshape((Nkp*NOrb*2,Nkp*NOrb*2))
        if(n < NP1):
          # Save electric dipole integrals in temporary array
          save_p[n,:,:] = np.copy(Pert)
        elif(n < NP1+NP2):
          # Correct magnetic dipole
          nn0 = n - NP1
          nn1 = (nn0+1)%3 
          nn2 = (nn0+2)%3
          # with open(f"{mol}.txt","a") as writer:
          #   writer.write(f"n={n}, nn0={nn0}, nn1={nn1}, nn2={nn2} \n")
          temp = -tv[nn2]*np.einsum('ij,jk->ik',save_p[nn1,:,:],UMat,optimize=True)
          temp += tv[nn1]*np.einsum('ij,jk->ik',save_p[nn2,:,:],UMat,optimize=True)
          temp = (temp - np.conjugate(temp).T)/2
          # Pert = np.copy(temp)
          Pert -= temp
          del temp
        else:
          # Correct electric quadrupole
          nn0 = n - (NP1+NP2)
          if(nn0 < 3):
            nn1 = nn0
            nn2 = nn0
            temp = 2*tv[nn2]*np.einsum('ij,jk->ik',save_p[nn1,:,:],UMat,optimize=True)
          else:
            if(nn0 == 3):
              # xy component
              nn1 = 0
              nn2 = 1
            elif(nn0 == 4):
              # xz component
              nn1 = 0
              nn2 = 2
            elif(nn0 == 5):
              # yz component
              nn1 = 1
              nn2 = 2
            temp = tv[nn2]*np.einsum('ij,jk->ik',save_p[nn1,:,:],UMat,optimize=True)
            temp += tv[nn1]*np.einsum('ij,jk->ik',save_p[nn2,:,:],UMat,optimize=True)
          temp = (temp - np.conjugate(temp).T)/2
          Pert -= temp
          del temp
        Pert = Pert.reshape((Nkp,NOrb*2,Nkp,NOrb*2))
      Pert = np.transpose(Pert,axes=(0,2,1,3))
      for k in range(Nkp):
        # prod_pert = np.einsum('ij,ij->',Pert[k,k,:,:],np.conjugate(Pert[k,k,:,:]),optimize=True)
        X_ij[n,k,k,:,:] = Pert[k,k,:O2,:O2]
        X_ia[n,k,k,:,:] = Pert[k,k,:O2,O2:]
        X_ab[n,k,k,:,:] = Pert[k,k,O2:,O2:]
        #prod_pert2 = np.einsum('ij,ij->',X_ij[n,k,k,:,:],np.conjugate(X_ij[n,k,k,:,:]),optimize=True)
        # prod_pert2 = np.einsum('ij,ij->',X_ia[n,k,k,:,:],np.conjugate(X_ia[n,k,k,:,:]),optimize=True)
        #prod_pert2 += np.einsum('ij,ij->',X_ab[n,k,k,:,:],np.conjugate(X_ab[n,k,k,:,:]),optimize=True)
        # with open(f"{mol}.txt","a") as writer:
        #   writer.write(f"Pert: {n+1}, Nk={k+1}, PertTot={prod_pert.real/2}, PertOV={prod_pert2.real/2} \n")
    if(pert_type == "DipE"):
      X_ij = np.transpose(X_ij,axes=(0,1,3,2,4))
    elif(pert_type == "DipEV" or pert_type == "FullOR_V"):
      X_ij = np.transpose(X_ij,axes=(0,2,4,1,3))
    else:
      print(f"getPert is confused about X_ij")
      exit()
    X_ij = X_ij.reshape((NP,O2k,O2k))
    X_ia = np.transpose(X_ia,axes=(0,1,3,2,4))
    X_ia = X_ia.reshape((NP,O2k,V2k))
    if(pert_type == "DipE"):
      X_ab = np.transpose(X_ab,axes=(0,1,3,2,4))
    elif(pert_type == "DipEV" or pert_type == "FullOR_V"):
      X_ab = np.transpose(X_ab,axes=(0,2,4,1,3))
    else:
      print(f"getPert is confused about X_ab")
      exit()
    X_ab = X_ab.reshape((NP,V2k,V2k))
    X_ia = np.conjugate(X_ia)
    if(pert_type == "DipE"):
      X_ij = np.conjugate(X_ij)
      X_ab = np.conjugate(X_ab)
    # elif(pert_type == "DipEV"):
    #   X_ia = np.conjugate(X_ia)
  else:
    # Molecular case
    PertSQ  = np.zeros((NP, NB, NB))
    if(pert_type == "DipE"):
      # Electric dipole length gauge
      for n in range (NP):
        PertSQ[n,:,:] = square_m(NB,True,"Sym",AOPert[n,:],PertSQ[n,:,:])
    elif(pert_type == "DipEV" or pert_type == "OR_V" or pert_type == "FullOR_V"):
      # Electric dipole, magnetic dipole, electric quadrupole velocity gauge
      for n in range (NP):
        PertSQ[n,:,:] = square_m(NB,True,"ASym",AOPert[n,:],PertSQ[n,:,:])
    temp = np.einsum('im,kml,jl->kij',MOCoef,PertSQ,MOCoef,optimize=True)
    X_ij = np.zeros((NP,O2,O2))
    X_ia = np.zeros((NP,O2,V2))
    X_ab = np.zeros((NP,V2,V2))
    for n in range(NP):
      for i in range(O):
        for j in range(O):
          X_ij[n,i,j] = temp[n,j,i]
          X_ij[n,i+O,j+O] = temp[n,j,i]
          # X_ij[n,i,j] = temp[n,i,j]
          # X_ij[n,i+O,j+O] = temp[n,i,j]
      for i in range(O):
        for a in range(V):
          X_ia[n,i,a] = temp[n,i,a+O]
          X_ia[n,i+O,a+V] = temp[n,i,a+O]
      for a in range(V):
        for b in range(V):
          X_ab[n,a,b] = temp[n,b+O,a+O]
          X_ab[n,a+V,b+V] = temp[n,b+O,a+O]
          # X_ab[n,a,b] = temp[n,a+O,b+O]
          # X_ab[n,a+V,b+V] = temp[n,a+O,b+O]
    del temp, AOPert, PertSQ
  return NP, NP1, NP2, NP3, X_ij, X_ia, X_ab

