"""
     CCCCCCCC
   CCCCCCCCCC                           
  CCC          RRRRRR                    PPPPPP 
 CCC   CCCCC   RR   RR                   PP   PP
 CCC  CC       RR    RR                  PP    PP  
 CCC  CC       RR   RR    eeee    sssss  PP   PP  yy   yy
 CCC   CCCCC   RRRRRR    ee  ee  ss      PPPPP     yy yy 
  CCC          RR   RR   eeeeee   ssss   PP         yyy  
   CCCCCCCCCC  RR    RR  ee          ss  PP         yy   
     CCCCCCCC  RR     RR  eeee   sssss   PP        yy    



VERSION 1.2.0
DATE May 12, 2026
This program is licensed under the terms of the GNU General Public
License v3.0 or later

Authors: M. Caricato, T. Parsons, J. Abdoullaeva

This program evaluates linear response funtions at CCSD level for
molecular and 1D periodic systems.

Available LR functions:
DipE : electric dipole-electric dipole polarizability - LG
DipEV : electric dipole-electric dipole polarizability - MVG
OR_L : optical rotation beta tensor - molecules only - LG(OI)
  (electric dipole-magnetic dipole) 
OR_V : optical rotation beta tensor - molecules only - MVG
  (electric dipole-magnetic dipole) 
FullOR_L : optical rotation full tensor - LG(OI)
  (electric dipole-magnetic dipole + electric dipole-electric quadrupole) 
FullOR_V : optical rotation full tensor - MVG
  (electric dipole-magnetic dipole + electric dipole-electric quadrupole) 
"""
#
import numpy as np
import os
import sys
import re
import time
from scipy.constants import angstrom, physical_constants
#
from ccres_read import input_parser, Initialize, getFort, getFock, get2e, conMO, getPert, getPert1k, getOvl, getFock1k, getOvl1k, TwoECO2, TwoECO1, TwoEMO
from ccres_funct import mem_check, denom, AmpIt, tau_tildeEq, tauEq, T_interm, fill_kl, Const_Interm, pert_rhs, Xi, TrDen1, print_tensor, reciprocal, form_map_kp, momentum_cons, denom3k, get_ktable, tau_tildeEq3k, tauEq3k, T_interm3k, Const_Interm3k, pert_rhs3k, Xi3k, TrDen1_1k
#
##########################################################################  
# Start Program
##########################################################################  
#
# Read input file and setup program parameters
start0=time.time()
if len(sys.argv)<2:
  print("Missing input file")
  exit()
input_file = sys.argv[1]
ThrE, ThrA, MaxIt, Wlist, MaxD, RepD, PertType, tv, FreezeCore, memory, scratch, path_gauopen, eri, mol_inp, mol_out, Kstore = input_parser(input_file)
#
# Initialize output file
Initialize(mol_out,memory,ThrE,MaxIt,Wlist)
#
# Retrieve various quantities
O, V, FC, FV, NB, scfE, MOCoef_Tot, ipbc, k_weights, atoms_list = getFort(mol_inp,mol_out,FreezeCore)
#
Ktable = []
if(ipbc):
  print(f"iPBC: {len(ipbc)}\n {ipbc}")
  npdir = ipbc[0]
  nmtpbc = ipbc[1]
  nrecip = ipbc[9]
  ndimk = ipbc[12:15]
  Nkp, _, _ = fill_kl(ipbc)
  #
  # Build table of momentum conserving k points for 2e arrays when
  # using compressed storage
  if(Kstore == "compress"): Ktable = get_ktable(npdir, ndimk)
  # 
  # Nkp = 1
  # for n in range(npdir): Nkp *= ndimk[n]
  # kp, l_list = fill_kl(ipbc)
  # co = np.einsum('k,l', kp, l_list, optimize=True)
  # cof = np.cos(co) + 1j*np.sin(co)
  # print(f"kp: {kp}\n")
  # print(f"l_list: {l_list}\n")
  # print(f"CO: {co}\n")
  # print(f"COF: {cof}\n")
  # exit()
  # Nkp = len(kp)
  sumtv = sum(np.array(tv))
  print(f"sumtv= {sumtv}, max = {max(abs(sumtv))} \n TV: {tv}")
  if(max(abs(sumtv)) == 0.0):
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"The translation vector is empty.\n")
    exit()
  with open(f"{mol_out}.txt","a") as writer:
    writer.write(f"PBC Information: N-cells: {nmtpbc} -- N-k points: {Nkp}\n")
  if nrecip == 1:
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"                 Gamma-point only\n")
  elif nrecip % 2 != 0 and nrecip != 1:
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"                 Edge and Gamma points are included\n")
  if(PertType == "OR_V" or PertType == "OR_L"):
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"The full OR tensor should be computed for periodic systems\n")
    exit()
  # Print out translation vectors
  with open(f"{mol_out}.txt","a") as writer:
    writer.write(f"                 Tranlation vectors (Ang)\n")
  for n in range(npdir):
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"                 a{n+1}: {tv[n][0]:+.8f} {tv[n][1]:+.8f} {tv[n][2]:+.8f} \n")
  # Print out reciprocal vectors
  with open(f"{mol_out}.txt","a") as writer:
    writer.write(f"                 Reciprocal vectors (Ang)\n")
  b_vecs = reciprocal(npdir,tv)
  for n in range(npdir):
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"                 b{n+1}: {b_vecs[n,0]:+.8f} {b_vecs[n,1]:+.8f} {b_vecs[n,2]:+.8f} \n")
  # Convert translation and reciprocal vectors Ang -> Bohr
  bohr_radius = physical_constants["Bohr radius"][0]
  tv = np.array(tv)*angstrom / bohr_radius
  b_vecs = b_vecs*bohr_radius/angstrom 
  # for n in range(npdir):
  #   with open(f"{mol_out}.txt","a") as writer:
  #     writer.write(f"                 a{n+1}: {tv[n][0]:+.8f} {tv[n][1]:+.8f} {tv[n][2]:+.8f} \n")
  # for n in range(npdir):
  #   with open(f"{mol_out}.txt","a") as writer:
  #     writer.write(f"                 b{n+1}: {b_vecs[n,0]:+.8f} {b_vecs[n,1]:+.8f} {b_vecs[n,2]:+.8f} \n")
  # prod = np.einsum('ij,kj->ik',tv,b_vecs,optimize=True)
  # with open(f"{mol_out}.txt","a") as writer:
  #   writer.write(f" product:\n {prod}\n")
#      
# Slice MO coefficient array to remove frozen core orbitals
if(ipbc):
  MOCoef = MOCoef_Tot[:,FC:,:]
else:
  MOCoef = MOCoef_Tot[FC:,:]
#  
# Get Fock matrix in MO basis
if(ipbc and Kstore == "compress"):
  Fock = getFock1k(mol_inp,O,V,NB,ipbc,"MO",False,0,MOCoef)
else:
  Fock = getFock(mol_inp,O,V,NB,ipbc,"MO",False,0,MOCoef)
tot_mem, avlb_mem = mem_check()
with open(f"{mol_out}.txt","a") as writer:
  writer.write(f"\nRead MO Coeff and Fock Matrix, Time: {time.time()-start0:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")
O2 = O*2
V2 = V*2
NOrb2 = O2 + V2
if(ipbc and Kstore == "compress"):
  print(f"Fock: {Fock.shape}, {np.size(Fock)}\n ")
  for kk in range(Nkp):
    FockD = np.diag(Fock[kk,:,:].real)
    print(f"{FockD}")
  Ovl = getOvl1k(mol_inp,O,V,NB,ipbc,"MO",False,0,MOCoef)
  print(f"Ovl: {Ovl.shape}, {np.size(Ovl)}\n ")
  for kk in range(Nkp):
    OvlD = np.diag(Ovl[kk,:,:].real)
    print(f"{OvlD}")
else:
  print(f"Fock: {Fock.shape}, {np.size(Fock)}\n ")
  FockD = np.diag(Fock.real).reshape((Nkp,NOrb2))
  print(f"{FockD}")
  Ovl = getOvl(mol_inp,O,V,NB,ipbc,"MO",False,0,MOCoef)
  print(f"Ovl: {Ovl.shape}, {np.size(Ovl)}\n ")
  OvlD = np.diag(Ovl.real).reshape((Nkp,NOrb2))
  print(f"{OvlD}")
# ndimk = ipbc[12:15]
# map_kp = form_map_kp(npdir,ndimk)
# for n in range(Nkp):
#   for k in range(Nkp):
#     for h in range(Nkp):
#       g = momentum_cons(npdir,ndimk,map_kp,n,k,h)
#
# _, _, _, _, _, _, _, _ = getPert(O,V,NB,ipbc,tv,MOCoef,
#                                  Fock,"DipE",mol_inp,
#                                  mol_out)
# _, _, _, _, _, _, _, _ = getPert(O,V,NB,ipbc,tv,MOCoef,
#                                  Fock,"DipEV",mol_inp,
#                                  mol_out)
# _, _, _, _, _, _, _, _ = getPert(O,V,NB,ipbc,tv,MOCoef,
#                                  Fock,"FullOR_L",mol_inp,
#                                  mol_out)
# _, _, _, _, _, _, _, _ = getPert(O,V,NB,ipbc,tv,MOCoef,
#                                  Fock,"FullOR_V",mol_inp,
#                                  mol_out)
# exit()
##########################################################################  
# Get AO 2e integrals and transform in MO basis
##########################################################################  
start=time.time()
get2e(NB,ipbc,mol_out,eri,scratch,path_gauopen)
with open(f"{mol_out}.txt","a") as writer:
  writer.write(f"Read AO 2ERI, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")
# Transform to molecular spin-orbital basis
start=time.time()
# conMO(mol_out,scratch,O,V,NB,ipbc,MOCoef)
if(ipbc):
  if(Kstore == "compress"):
    TwoECO2(mol_out,scratch,O,V,NB,ipbc,MOCoef)
  elif(Kstore == "collective"):
    TwoECO1(mol_out,scratch,O,V,NB,ipbc,MOCoef)
else:
  TwoEMO(mol_out, scratch, O, V, NB, MOCoef)
tot_mem, avlb_mem = mem_check()
with open(f"{mol_out}.txt","a") as writer:
  writer.write(f"2ERI AO->MO, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")
start=time.time()
#
# PBC Info
nmtpbc = 1
Nkp = 1
kp = []
Ok = O
Vk = V
O2k = O2
V2k = V2
NOrb2k = NOrb2
if(ipbc):
  nmtpbc = ipbc[1]
  Nkp, _, _ = fill_kl(ipbc)
  # Nkp = len(kp)
  if(Kstore == "collective"):
    O2k = O2*Nkp
    V2k = V2*Nkp
    Ok = O*Nkp
    Vk = V*Nkp
    NOrb2k = NOrb2*Nkp
NkpC = Nkp*Nkp*Nkp
#  
##########################################################################  
# CCSD Energy and Amplitudes
##########################################################################
#
# Define denominator arrays
W = 0
# D1, D2 =  denom(1,O2,V2,kp,Fock,W)
if(ipbc and Kstore == "compress"):
  D1, D2 =  denom3k(O2,V2,ipbc,Ktable,Fock,W)
else:
  D1, D2 =  denom(1,O2,V2,ipbc,Fock,W)
tot_mem, avlb_mem = mem_check()
with open(f"{mol_out}.txt","a") as writer:
  writer.write(f"Compute energy denominators, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
start=time.time()
#
# Initialize T1 and T2
IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
t2 = np.conjugate(IJAB)/D2.real
if(ipbc and Kstore == "compress"):
  t1 = np.zeros((Nkp,O2,V2),dtype=Fock.dtype)
  EMP2 = 0.25*np.einsum('nkhijab,nkhijab',IJAB,t2,optimize=True)/NkpC
else:
  t1 = np.zeros((O2k,V2k),dtype=Fock.dtype)
  EMP2 = 0.25*np.einsum('ijab,ijab',IJAB,t2,optimize=True)/NkpC
del IJAB
tot_mem, avlb_mem = mem_check()
with open(f"{mol_out}.txt","a") as writer:
  writer.write(f"T guess, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
#
# Solve amplitude equations
with open(f"{mol_out}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*          SOLVING CCSD T AMPLITUDE EQS.           *\n")
  writer.write("****************************************************\n")
  writer.write(f"E(SCF)= = {scfE.real:.10f}au, DE(MP2) = {EMP2.real:.10f}au"
               f", E(MP2) = {scfE.real+EMP2.real:.10f}au\n")
# print(f"good enough")
# exit()
tau = []
if(ipbc and Kstore == "compress"):
  W_iemn = np.lib.format.open_memmap(f"{scratch}/{mol_out}-Wiemn.npy",
                                     mode='w+',shape=(Nkp,Nkp,Nkp,O2,V2,O2,O2),
                                     dtype=Fock.dtype) 
  W_mbej = np.lib.format.open_memmap(f"{scratch}/{mol_out}-Wmbej.npy",
                                     mode='w+',shape=(Nkp,Nkp,Nkp,O2,V2,V2,O2),
                                     dtype=Fock.dtype) 
  W_mnij = np.lib.format.open_memmap(f"{scratch}/{mol_out}-Wmnij.npy",
                                     mode='w+',shape=(Nkp,Nkp,Nkp,O2,O2,O2,O2),
                                     dtype=Fock.dtype) 
  W_efam = np.lib.format.open_memmap(f"{scratch}/{mol_out}-Wefam.npy",
                                     mode='w+',shape=(Nkp,Nkp,Nkp,V2,V2,V2,O2),
                                     dtype=Fock.dtype)
else:
  W_iemn = np.lib.format.open_memmap(f"{scratch}/{mol_out}-Wiemn.npy",
                                     mode='w+',shape=(O2k,V2k,O2k,O2k),
                                     dtype=Fock.dtype) 
  W_mbej = np.lib.format.open_memmap(f"{scratch}/{mol_out}-Wmbej.npy",
                                     mode='w+',shape=(O2k,V2k,V2k,O2k),
                                     dtype=Fock.dtype) 
  W_mnij = np.lib.format.open_memmap(f"{scratch}/{mol_out}-Wmnij.npy",
                                     mode='w+',shape=(O2k,O2k,O2k,O2k),
                                     dtype=Fock.dtype) 
  W_efam = np.lib.format.open_memmap(f"{scratch}/{mol_out}-Wefam.npy",
                                     mode='w+',shape=(V2k,V2k,V2k,O2k),
                                     dtype=Fock.dtype) 
F_ae = []
F_mi = []
F_me = []
t1, t2 = AmpIt("T",mol_out,scratch,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
               tau,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,t1,t2,t1,t2,ipbc,
               Kstore,Ktable)
#
##########################################################################  
# Compute constant intermediates
##########################################################################  
start=time.time()
if(f"{scratch}/{mol_out}-ABCD.npy"): 
  os.system(f"mv {scratch}/{mol_out}-ABCD.npy {scratch}/{mol_out}-Wabef.npy")
else:
  with open(f"{mol_out}.txt","a") as writer:
    writer.write(f"ABCD integrals file is missing\n")
  exit()
if(ipbc and Kstore == "compress"):
  tau_tilde = tau_tildeEq3k(Nkp,t1,t2)
  tau = tauEq3k(Nkp,t1,t2)
  F_ae,F_mi,F_me = T_interm3k(mol_out,scratch,Ok,Vk,Nkp,Ktable,Fock,t1,t2,
                              tau_tilde,tau)
  F_ae,F_mi = Const_Interm3k(mol_out,scratch,Nkp,Ktable,t1,t2,tau,F_ae,
                             F_mi,F_me)
else:
  tau_tilde = tau_tildeEq(Nkp,t1,t2)
  tau = tauEq(Nkp,t1,t2)
  F_ae,F_mi,F_me = T_interm(mol_out,scratch,Ok,Vk,Nkp,Fock,t1,t2,tau_tilde,tau)
  F_ae,F_mi = Const_Interm(mol_out,scratch,Nkp,t1,t2,tau,F_ae,F_mi,F_me)
tot_mem, avlb_mem = mem_check()
with open(f"{mol_out}.txt","a") as writer:
  writer.write(f"Constant intermediates evaluated, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
#  
##########################################################################  
# CCSD Lambda Amplitudes
##########################################################################
l1 = np.copy(np.conjugate(t1))
l2 = np.copy(np.conjugate(t2))
with open(f"{mol_out}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*        SOLVING CCSD Lambda AMPLITUDE EQS.        *\n")
  writer.write("****************************************************\n")
l1, l2 = AmpIt("L",mol_out,scratch,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
               tau,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,l1,l2,t1,t2,ipbc,
               Kstore,Ktable)
np.save(f"{scratch}/{mol_out}-l1",l1)
np.save(f"{scratch}/{mol_out}-l2",l2)
del l1, l2
# exit()
#
##########################################################################  
# CCSD LR equations
##########################################################################
#
# NP = number of perturbations (3 for dipoles and 6 for quadrupoles)
# W = frequency of perturbation
# if W != 0, there two sets of amplitudes per perturbation Tx(+w) and Tx(-w)
# Use same intermediates as in Lambda equations
start=time.time()
with open(f"{mol_out}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*           COMPUTING CCSD LR FUNCTION             *\n")
  writer.write("****************************************************\n")
if(ipbc and Kstore == 'compress'):
  NP, NP1, NP2, NP3, NP4, X_ij, X_ia, X_ab = getPert1k(O,V,NB,ipbc,tv,MOCoef,
                                                       Fock,PertType,mol_inp,
                                                       mol_out)
else:
  NP, NP1, NP2, NP3, NP4, X_ij, X_ia, X_ab = getPert(O,V,NB,ipbc,tv,MOCoef,
                                                     Fock,PertType,mol_inp,
                                                     mol_out)
tot_mem, avlb_mem = mem_check()
with open(f"{mol_out}.txt","a") as writer:
  writer.write(f"Perturbation integrals read, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
tensor = np.zeros((len(Wlist), NP1, NP2),dtype=Fock.dtype)
tensorDQ = []
alpha_mix = []
if(PertType == "FullOR_V" or PertType == "FullOR_L"):
  tensorDQ = np.zeros((len(Wlist),NP1, NP3),dtype=Fock.dtype)
if(PertType == "OR_L" or PertType == "FullOR_L"):
  alpha_mix = np.zeros((len(Wlist),NP1, NP1),dtype=Fock.dtype)
for iw in range(len(Wlist)):
  # Loop over frequencies    
  W = Wlist[iw]
  with open(f"{mol_out}.txt","a") as writer:
    writer.write("\n****************************************************\n")
    writer.write(f" Start Linear Response Calculation for Frequency {W:f} a.u.\n\n")
  NW = 2
  MaxX = np.zeros((NP))
  if (W==0): NW = 1
  if(ipbc and Kstore == 'compress'):
    tx1 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-tx1.npy",mode='w+',
                                    dtype=Fock.dtype, shape=(NP,2,Nkp,O2,V2)) 
    tx2 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-tx2.npy",mode='w+',
                                    dtype=Fock.dtype, shape=(NP,2,Nkp,Nkp,Nkp,
                                                             O2,O2,V2,V2)) 
  else:
    tx1 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-tx1.npy",mode='w+',
                                    dtype=Fock.dtype, shape=(NP,2,O2k,V2k)) 
    tx2 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-tx2.npy",mode='w+',
                                    dtype=Fock.dtype, shape=(NP,2,O2k,O2k,V2k,V2k)) 
  for ip in range(NP):
    # Loop over number of non-zero pertubations
    MaxIJr = np.max(abs(X_ij[ip,:,:].real))
    MaxIJi = np.max(abs(X_ij[ip,:,:].imag))
    MaxIAr = np.max(abs(X_ia[ip,:,:].real))
    MaxIAi = np.max(abs(X_ia[ip,:,:].imag))
    MaxABr = np.max(abs(X_ab[ip,:,:].real))
    MaxABi = np.max(abs(X_ab[ip,:,:].imag))
    MaxX[ip] = max(MaxIJr,MaxIJi,MaxIAr,MaxIAi,MaxABr,MaxABi)
    if(MaxX[ip] > 1e-15):
      start=time.time()
      PertSymm = "Symm"
      if(PertType == "DipEV" or PertType == "OR_V" or PertType == "FullOR_V"
         or ((PertType == "OR_L" or PertType == "FullOR_L") and ip >= NP1)):
        PertSymm = "ASymm"
      if(ipbc and Kstore == 'compress'):
        rhs1, rhs2 = pert_rhs3k(PertSymm,Nkp,O2,V2,t1,t2,X_ij[ip],X_ia[ip],
                                X_ab[ip],Ktable)
        rhs1prod = np.einsum('Iia,Iia->',rhs1,np.conjugate(rhs1),optimize=True)
        rhs2prod = np.einsum('IJAijab,IJAijab->',rhs2,np.conjugate(rhs2),optimize=True)
      else:
        rhs1, rhs2 = pert_rhs(PertSymm,Nkp,O2k,V2k,t1,t2,X_ij[ip],X_ia[ip],
                              X_ab[ip])
        rhs1prod = np.einsum('ia,ia->',rhs1,np.conjugate(rhs1),optimize=True)
        rhs2prod = np.einsum('ijab,ijab->',rhs2,np.conjugate(rhs2),optimize=True)
      print(f"rhs1: {rhs1prod/Nkp}, rhs2: {rhs2prod/Nkp**3}")
      tot_mem, avlb_mem = mem_check()
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f"Right hand side evaluated, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f"\n Perturbation {PertType}-{ip+1}\n\n")
      for ipmw in range(NW):
        # Loop over +/-omega
        PMW = W
        if (ipmw==1): PMW = -W 
        with open(f"{mol_out}.txt","a") as writer:
          writer.write(f" Frequency {PMW:+f} a.u.\n")
        # Reset denominators including frequency term and initialize amplitudes
        if(ipbc and Kstore == 'compress'):
          D1, D2 =  denom3k(O2,V2,ipbc,Ktable,Fock,PMW)
        else:
          D1, D2 =  denom(1, O2, V2, ipbc, Fock, PMW)
        # D1, D2 =  denom(1, O2, V2, kp, Fock, PMW)
        tx1[ip,ipmw] = 0
        tx2[ip,ipmw] = 0
        ttx1 = tx1[ip,ipmw]
        ttx2 = tx2[ip,ipmw]
        # ttx1 = tx1[ip,ipmw,:,:]
        # ttx2 = tx2[ip,ipmw,:,:,:,:]
        ttx1 -= rhs1/D1.real
        ttx2 -= rhs2/D2.real
        # Amplitudes loop
        if(ipbc and Kstore == 'compress'):
          ttx1[:,:,:], ttx2[:,:,:,:,:,:,:] = AmpIt("Tx",mol_out,scratch,Ok,Vk,
                                                   Nkp,MaxIt,ThrE,ThrA,scfE,
                                                   Fock,tau,F_ae,F_mi,F_me,rhs1,
                                                   rhs2,D1,D2,t1,t2,t1,t2,ttx1,
                                                   ttx2,ipbc,Kstore,Ktable)
          tx1prod = np.einsum('Iia,Iia->',ttx1,np.conjugate(ttx1),optimize=True)/Nkp
          tx2prod = np.einsum('IJAijab,IJAijab->',ttx2,np.conjugate(ttx2),optimize=True)/Nkp**3
        else:
          ttx1[:,:], ttx2[:,:,:,:] = AmpIt("Tx",mol_out,scratch,Ok,Vk,Nkp,
                                           MaxIt,ThrE,ThrA,scfE,Fock,tau,
                                           F_ae,F_mi,F_me,rhs1,rhs2,D1,D2,
                                           t1,t2,t1,t2,ttx1,ttx2,ipbc,Kstore,
                                           Ktable)
          tx1prod = np.einsum('ia,ia->',ttx1,np.conjugate(ttx1),optimize=True)/Nkp
          tx2prod = np.einsum('ijab,ijab->',ttx2,np.conjugate(ttx2),optimize=True)/Nkp**3
        print(f"tx1: {tx1prod}, tx2: {tx2prod}")
        del ttx1, ttx2
      if(NW == 1):
        # This is a static case. Make a copy of the amplitudes for the -W case.
        tx1[ip,1] = np.copy(tx1[ip,0])
        tx2[ip,1] = np.copy(tx2[ip,0])
        # tx1[ip,1,:,:] = np.copy(tx1[ip,0,:,:])
        # tx2[ip,1,:,:,:,:] = np.copy(tx2[ip,0,:,:,:,:])
  del tx1, tx2
  #
  # Now that we have all the Tx amplitudes for this W, we can compute
  # the corresponding Xi amplitudes and contract with all other Tx
  # amplitudes, and the transition 1PDM-like rho1 and contract with
  # the perturbation integrals
  #
  # Reset denominators
  if(ipbc and Kstore == 'compress'):
    D1, D2 =  denom3k(O2,V2,ipbc,Ktable,Fock,0)
  else:
    D1, D2 =  denom(1, O2, V2, ipbc, Fock, 0)
  # D1, D2 =  denom(1, O2, V2, kp, Fock, 0)
  tx1 = np.load(f"{scratch}/{mol_out}-tx1.npy",mmap_mode='r')
  tx2 = np.load(f"{scratch}/{mol_out}-tx2.npy",mmap_mode='r')
  for ip in range(NP):
    if(MaxX[ip] > 1e-15):
      # Evaluate Xi amplitudes 
      start=time.time()
      l1 = np.load(f"{scratch}/{mol_out}-l1.npy",mmap_mode='r')
      l2 = np.load(f"{scratch}/{mol_out}-l2.npy",mmap_mode='r')
      ttx1 = tx1[ip,0]
      ttx2 = tx2[ip,0]
      if(ipbc and Kstore == 'compress'):
        Xi1, Xi2 = Xi3k(mol_out,scratch,Nkp,Ktable,O2,V2,ttx1,ttx2,l1,l2,t1,
                        F_ae,F_mi,F_me,D2)
        Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
        Xi2prod = np.einsum('IJAijab,IJAijab->',Xi2,np.conjugate(Xi2),optimize=True)
      else:
        Xi1, Xi2 = Xi(1,mol_out,scratch,Nkp,O2k,ttx1,ttx2,l1,l2,t1,
                      F_ae,F_mi,F_me,D2)
        Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
        Xi2prod = np.einsum('ijab,ijab->',Xi2,np.conjugate(Xi2),optimize=True)
      print(f"Xi1: {Xi1prod/Nkp}, Xi2: {Xi2prod/Nkp**3}")
      del ttx1, ttx2, l1, l2
      tot_mem, avlb_mem = mem_check()
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f"Xi terms evaluated, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
      for ipa in range(NP2+NP3+NP4):
        # Contract Xi(ip) with Tx(ipa)
        if(PertType == "DipE" or PertType == "DipEV"):
          ttx1 = tx1[ipa,1]
          ttx2 = tx2[ipa,1]
          if(ipbc and Kstore == 'compress'):
            tensor[iw,ip,ipa] -= np.einsum('Iia,Iia->',Xi1,ttx1,
                                           optimize=True)/Nkp 
            tensor[iw,ip,ipa] -= 0.25*np.einsum('IJAijab,IJAijab->',Xi2,
                                                ttx2,optimize=True)/NkpC
          else:
            tensor[iw,ip,ipa] -= np.einsum('ia,ia->',Xi1,ttx1,
                                           optimize=True)/Nkp 
            tensor[iw,ip,ipa] -= 0.25*np.einsum('ijab,ijab->',Xi2,
                                                ttx2,optimize=True)/NkpC
          del ttx1, ttx2
        elif(PertType == "OR_V"):
          if(ip < NP1):
            # mu(+)m(-)
            ip1 = ip
            ipa1 = ipa
            ipa2 = ipa + NP1
          else:
            # mu(-)m(+)
            ip1 = ipa
            ipa1 = ip - NP1 
            ipa2 = ipa
          ttx1 = tx1[ipa2,1]
          ttx2 = tx2[ipa2,1]
          if(ipbc and Kstore == 'compress'):
            tensor[iw,ip1,ipa1] -= np.einsum('Iia,Iia->',Xi1,ttx1,
                                             optimize=True)/Nkp 
            tensor[iw,ip1,ipa1] -= 0.25*np.einsum('IJAijab,IJAijab->',Xi2,
                                                ttx2,optimize=True)/NkpC
          else:
            tensor[iw,ip1,ipa1] -= np.einsum('ia,ia->',Xi1,ttx1,
                                             optimize=True)/Nkp 
            tensor[iw,ip1,ipa1] -= 0.25*np.einsum('ijab,ijab->',Xi2,
                                                ttx2,optimize=True)/NkpC
          del ttx1, ttx2
        elif(PertType == "OR_L"):
          if(ipa < NP2 and ip < NP1+NP2):
            # Beta contribution
            if(ip < NP1):
              # mu(+)m(-)
              ip1 = ip
              ipa1 = ipa
              ipa2 = ipa + NP1
              fact = 1
            elif(ip < NP1+NP2):
              # mu(-)m(+)
              ip1 = ipa
              ipa1 = ip - NP1 
              ipa2 = ipa
              fact = -1
            ttx1 = tx1[ipa2,1]
            ttx2 = tx2[ipa2,1]
            if(ipbc and Kstore == 'compress'):
              tensor[iw,ip1,ipa1] -= fact*np.einsum('Iia,Iia->',Xi1,ttx1,
                                                    optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] -= 0.25*fact*np.einsum('IJAijab,IJAijab->',Xi2,
                                                         ttx2,optimize=True)/NkpC
            else:
              tensor[iw,ip1,ipa1] -= fact*np.einsum('ia,ia->',Xi1,ttx1,
                                                    optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] -= 0.25*fact*np.einsum('ijab,ijab->',Xi2,
                                                         ttx2,optimize=True)/NkpC
            del ttx1, ttx2
          elif((ipa >= NP2 and ip < NP1) or (ip >= NP1+NP2 and ipa < NP2)):
            # alpha(L,V) contribution
            if(ip < NP1):
              # mu_L(+)mu_V(-)
              ip1 = ip
              ipa1 = ipa - NP2
              ipa2 = ipa + NP1 
              fact = 1
            elif(ip >= NP1+NP2):
              # mu_L(-)mu_V(+)
              ip1 = ipa 
              ipa1 = ip - NP1 - NP2 
              ipa2 = ipa 
              fact = -1
            ttx1 = tx1[ipa2,1]
            ttx2 = tx2[ipa2,1]
            if(ipbc and Kstore == 'compress'):
              alpha_mix[iw,ip1,ipa1] -= fact*np.einsum('Iia,Iia->',Xi1,
                                                       ttx1,optimize=True)/Nkp 
              alpha_mix[iw,ip1,ipa1] -= 0.25*fact*np.einsum('IJAijab,IJAijab->',
                                                            Xi2,ttx2,
                                                            optimize=True)/NkpC
            else:
              alpha_mix[iw,ip1,ipa1] -= fact*np.einsum('ia,ia->',Xi1,
                                                       ttx1,optimize=True)/Nkp 
              alpha_mix[iw,ip1,ipa1] -= 0.25*fact*np.einsum('ijab,ijab->',Xi2,
                                                            ttx2,optimize=True)/NkpC
            del ttx1, ttx2
        elif(PertType == "FullOR_V"):
          if(ipa < NP2 and ip < NP1+NP2):
            # Beta contribution
            if(ip < NP1):
              # mu(+)m(-)
              ip1 = ip
              ipa1 = ipa
              ipa2 = ipa + NP1
            elif(ip < NP1+NP2):
              # mu(-)m(+)
              ip1 = ipa
              ipa1 = ip - NP1 
              ipa2 = ipa
            ttx1 = tx1[ipa2,1]
            ttx2 = tx2[ipa2,1]
            if(ipbc and Kstore == 'compress'):
              tensor[iw,ip1,ipa1] -= np.einsum('Iia,Iia->',Xi1,ttx1,optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] -= 0.25*np.einsum('IJAijab,IJAijab->',Xi2,
                                                    ttx2,optimize=True)/NkpC
            else:
              tensor[iw,ip1,ipa1] -= np.einsum('ia,ia->',Xi1,ttx1,optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] -= 0.25*np.einsum('ijab,ijab->',Xi2,
                                                    ttx2,optimize=True)/NkpC
            del ttx1, ttx2
          elif((ipa >= NP2 and ip < NP1) or (ip >= NP1+NP2 and ipa < NP2)):
            # A contribution
            if(ip < NP1):
              # mu(+)Theta(-)
              ip1 = ip
              ipa1 = ipa - NP2
              ipa2 = ipa + NP1 
            elif(ip >= NP1+NP2):
              # mu(-)Theta(+)
              ip1 = ipa 
              ipa1 = ip - NP1 - NP2 
              ipa2 = ipa 
            ttx1 = tx1[ipa2,1]
            ttx2 = tx2[ipa2,1]
            if(ipbc and Kstore == 'compress'):
              tensorDQ[iw,ip1,ipa1] -= np.einsum('Iia,Iia->',Xi1,
                                                 ttx1,optimize=True)/Nkp 
              tensorDQ[iw,ip1,ipa1] -= 0.25*np.einsum('IJAijab,IJAijab->',Xi2,
                                                      ttx2,optimize=True)/NkpC
            else:
              tensorDQ[iw,ip1,ipa1] -= np.einsum('ia,ia->',Xi1,
                                                 ttx1,optimize=True)/Nkp 
              tensorDQ[iw,ip1,ipa1] -= 0.25*np.einsum('ijab,ijab->',Xi2,ttx2,
                                                      optimize=True)/NkpC
            del ttx1, ttx2
        elif(PertType == "FullOR_L"):
          if(ipa < NP2 and ip < NP1+NP2):
            # Beta contribution
            if(ip < NP1):
              # mu(+)m(-)
              ip1 = ip
              ipa1 = ipa
              ipa2 = ipa + NP1
              fact = 1
            elif(ip < NP1+NP2):
              # mu(-)m(+)
              ip1 = ipa
              ipa1 = ip - NP1 
              ipa2 = ipa
              fact = -1
            ttx1 = tx1[ipa2,1]
            ttx2 = tx2[ipa2,1]
            if(ipbc and Kstore == 'compress'):
              tensor[iw,ip1,ipa1] -= fact*np.einsum('Iia,Iia->',Xi1,ttx1,
                                                    optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] -= 0.25*fact*np.einsum('IJAijab,IJAijab->',
                                                         Xi2,ttx2,
                                                         optimize=True)/NkpC
            else:
              tensor[iw,ip1,ipa1] -= fact*np.einsum('ia,ia->',Xi1,ttx1,
                                                    optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] -= 0.25*fact*np.einsum('ijab,ijab->',Xi2,ttx2,
                                                         optimize=True)/NkpC
            del ttx1, ttx2
          elif((ipa >= NP2 and ip < NP1 and ipa<NP2+NP3) or
               (ip >= NP1+NP2 and ipa < NP2 and ip<NP1+NP2+NP3)):
            # A contribution
            if(ip < NP1):
              # mu(+)Theta(-)
              ip1 = ip
              ipa1 = ipa - NP2
              ipa2 = ipa + NP1 
              fact = 1
            elif(ip >= NP1+NP2):
              # mu(-)Theta(+)
              ip1 = ipa 
              ipa1 = ip - NP1 - NP2 
              ipa2 = ipa 
              fact = -1
            ttx1 = tx1[ipa2,1]
            ttx2 = tx2[ipa2,1]
            if(ipbc and Kstore == 'compress'):
              tensorDQ[iw,ip1,ipa1] -= fact*np.einsum('Iia,Iia->',Xi1,
                                                      ttx1,optimize=True)/Nkp 
              tensorDQ[iw,ip1,ipa1] -= 0.25*fact*np.einsum('IJAijab,IJAijab->',
                                                           Xi2,ttx2,
                                                           optimize=True)/NkpC
            else:
              tensorDQ[iw,ip1,ipa1] -= fact*np.einsum('ia,ia->',Xi1,
                                                      ttx1,optimize=True)/Nkp 
              tensorDQ[iw,ip1,ipa1] -= 0.25*fact*np.einsum('ijab,ijab->',
                                                           Xi2,ttx2,
                                                           optimize=True)/NkpC
            del ttx1, ttx2
          elif((ipa >= NP2+NP3 and ip < NP1) or (ip >= NP1+NP2+NP3 and ipa < NP2)):
            # alpha(L,V) contribution
            if(ip < NP1):
              # mu_L(+)mu_V(-)
              ip1 = ip
              ipa1 = ipa - NP2 - NP3
              ipa2 = ipa + NP1 
              fact = 1
            elif(ip >= NP1+NP2+NP3):
              # mu(-)Theta(+)
              ip1 = ipa 
              ipa1 = ip - NP1 - NP2 - NP3 
              ipa2 = ipa 
              fact = -1
            ttx1 = tx1[ipa2,1]
            ttx2 = tx2[ipa2,1]
            if(ipbc and Kstore == 'compress'):
              alpha_mix[iw,ip1,ipa1] -= fact*np.einsum('Iia,Iia->',Xi1,
                                                       ttx1,optimize=True)/Nkp 
              alpha_mix[iw,ip1,ipa1] -= 0.25*fact*np.einsum('IJAijab,IJAijab->',
                                                            Xi2,ttx2,
                                                            optimize=True)/NkpC
            else:
              alpha_mix[iw,ip1,ipa1] -= fact*np.einsum('ia,ia->',Xi1,
                                                       ttx1,optimize=True)/Nkp 
              alpha_mix[iw,ip1,ipa1] -= 0.25*fact*np.einsum('ijab,ijab->',
                                                            Xi2,ttx2,
                                                            optimize=True)/NkpC
            del ttx1, ttx2
      del Xi1, Xi2
      for ipmw in range(NW):
        # Loop over +/-omega
        # Evaluate 1PDM
        start=time.time()
        ttx1 = tx1[ip,ipmw]
        ttx2 = tx2[ip,ipmw]
        l1 = np.load(f"{scratch}/{mol_out}-l1.npy",mmap_mode='r')
        l2 = np.load(f"{scratch}/{mol_out}-l2.npy",mmap_mode='r')
        if(ipbc and Kstore == 'compress'):
          rho1 = TrDen1_1k(O2k,NOrb2k,Nkp,Ktable,ttx1,ttx2,l1,l2,t1,t2)
          rhoprod = np.einsum('Iia,Iia->',rho1,np.conjugate(rho1),optimize=True)
        else:
          rho1 = TrDen1(1,O2k,NOrb2k,Nkp,ttx1,ttx2,l1,l2,t1,t2)
          rhoprod = np.einsum('ia,ia->',rho1,np.conjugate(rho1),optimize=True)
        print(f"rho1: {rhoprod/Nkp}")
        del ttx1, ttx2, l1, l2
        tot_mem, avlb_mem = mem_check()
        with open(f"{mol_out}.txt","a") as writer:
          writer.write(f"Rho evaluated, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
        for ipa in range(NP2+NP3+NP4):
          # Contract 1PDM(ip) with Pert(ipa)
          if(PertType == "DipE"):
            if(ipbc and Kstore == 'compress'):
              tensor[iw,ip,ipa] += np.einsum('Iia,Iia->',np.conjugate(X_ia[ipa]),
                                             rho1[:,:O2k,O2k:],optimize=True)/Nkp
              tensor[iw,ip,ipa] += np.einsum('Iij,Iij->',np.conjugate(X_ij[ipa]),
                                             rho1[:,:O2k,:O2k],optimize=True)/Nkp 
              tensor[iw,ip,ipa] += np.einsum('Aab,Aab->',np.conjugate(X_ab[ipa]),
                                             rho1[:,O2k:,O2k:],optimize=True)/Nkp
            else:
              tensor[iw,ip,ipa] += np.einsum('ia,ia->',np.conjugate(X_ia[ipa]),
                                             rho1[:O2k,O2k:],optimize=True)/Nkp
              tensor[iw,ip,ipa] += np.einsum('ij,ij->',np.conjugate(X_ij[ipa]),
                                             rho1[:O2k,:O2k],optimize=True)/Nkp 
              tensor[iw,ip,ipa] += np.einsum('ab,ab->',np.conjugate(X_ab[ipa]),
                                             rho1[O2k:,O2k:],optimize=True)/Nkp
          elif(PertType == "DipEV"):
            f_static = 1
            if(iw == 0): f_static = 2
            if(ipbc and Kstore == 'compress'):
              tensor[iw,ip,ipa] += f_static*np.einsum('Iia,Iia->',
                                                      np.conjugate(X_ia[ipa]),
                                                      rho1[:,:O2k,O2k:],
                                                      optimize=True)/Nkp   
              tensor[iw,ip,ipa] += f_static*np.einsum('Iji,Iij->',X_ij[ipa],
                                                      rho1[:,:O2k,:O2k],
                                                      optimize=True)/Nkp 
              tensor[iw,ip,ipa] += f_static*np.einsum('Aba,Aab->',X_ab[ipa],
                                                      rho1[:,O2k:,O2k:],
                                                      optimize=True)/Nkp   
            else:
              tensor[iw,ip,ipa] += f_static*np.einsum('ia,ia->',
                                                      np.conjugate(X_ia[ipa]),
                                                      rho1[:O2k,O2k:],
                                                      optimize=True)/Nkp   
              tensor[iw,ip,ipa] += f_static*np.einsum('ji,ij->',X_ij[ipa],
                                                      rho1[:O2k,:O2k],
                                                      optimize=True)/Nkp 
              tensor[iw,ip,ipa] += f_static*np.einsum('ba,ab->',X_ab[ipa],
                                                      rho1[O2k:,O2k:],
                                                      optimize=True)/Nkp   
          elif(PertType == "OR_V"):
            if(ip < NP1):
              # mu(+)m(-)
              ip1 = ip
              ipa1 = ipa
              ipa2 = ipa + NP1
            else:
              # mu(-)m(+)
              ip1 = ipa
              ipa1 = ip - NP1 
              ipa2 = ipa
            f_static = 1
            if(iw == 0): f_static = 2
            if(ipbc and Kstore == 'compress'):
              tensor[iw,ip1,ipa1] += f_static*np.einsum('Iia,Iia->',
                                                        np.conjugate(X_ia[ipa2]),
                                                        rho1[:,:O2k,O2k:],
                                                        optimize=True)/Nkp   
              tensor[iw,ip1,ipa1] += f_static*np.einsum('Iji,Iij->',X_ij[ipa2],
                                                        rho1[:,:O2k,:O2k],
                                                        optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] += f_static*np.einsum('Aba,Aab->',X_ab[ipa2],
                                                        rho1[:,O2k:,O2k:],
                                                        optimize=True)/Nkp   
            else:
              tensor[iw,ip1,ipa1] += f_static*np.einsum('ia,ia->',
                                                        np.conjugate(X_ia[ipa2]),
                                                        rho1[:O2k,O2k:],
                                                        optimize=True)/Nkp   
              tensor[iw,ip1,ipa1] += f_static*np.einsum('ji,ij->',X_ij[ipa2],
                                                        rho1[:O2k,:O2k],
                                                        optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] += f_static*np.einsum('ba,ab->',X_ab[ipa2],
                                                        rho1[O2k:,O2k:],
                                                        optimize=True)/Nkp   
          elif(PertType == "OR_L"):
            if(ipa < NP2 and ip < NP1+NP2):
              # Beta contribution
              if(ip < NP1):
                # mu(+)m(-)
                ip1 = ip
                ipa1 = ipa
                ipa2 = ipa + NP1
                fact = 1
                if(ipmw > 0): fact = -1
              elif(ip < NP1+NP2):
                # mu(-)m(+)
                ip1 = ipa
                ipa1 = ip - NP1 
                ipa2 = ipa
                fact = -1
                if(ipmw > 0): fact = 1
              if(ipbc and Kstore == 'compress'):
                tensor[iw,ip1,ipa1] += fact*np.einsum('Iia,Iia->',
                                                      np.conjugate(X_ia[ipa2]),
                                                      rho1[:,:O2k,O2k:],
                                                      optimize=True)/Nkp   
                tensor[iw,ip1,ipa1] += fact*np.einsum('Iji,Iij->',X_ij[ipa2],
                                                      rho1[:,:O2k,:O2k],
                                                      optimize=True)/Nkp 
                tensor[iw,ip1,ipa1] += fact*np.einsum('Aba,Aab->',X_ab[ipa2],
                                                      rho1[:,O2k:,O2k:],
                                                      optimize=True)/Nkp   
              else:
                tensor[iw,ip1,ipa1] += fact*np.einsum('ia,ia->',
                                                      np.conjugate(X_ia[ipa2]),
                                                      rho1[:O2k,O2k:],
                                                      optimize=True)/Nkp   
                tensor[iw,ip1,ipa1] += fact*np.einsum('ji,ij->',X_ij[ipa2],
                                                      rho1[:O2k,:O2k],
                                                      optimize=True)/Nkp 
                tensor[iw,ip1,ipa1] += fact*np.einsum('ba,ab->',X_ab[ipa2],
                                                      rho1[O2k:,O2k:],
                                                      optimize=True)/Nkp   
            elif((ipa >= NP2 and ip < NP1) or (ip >= NP1+NP2 and ipa < NP2)):
              # alpha(L,V) contribution
              if(ip < NP1):
                # mu_L(+)mu_V(-)
                ip1 = ip
                ipa1 = ipa - NP2
                ipa2 = ipa + NP1
                fact = 1
                if(ipmw > 0): fact = -1
              elif(ip >= NP1+NP2):
                # mu_L(-)mu_V(+)
                ip1 = ipa 
                ipa1 = ip - NP1 - NP2 
                ipa2 = ipa 
                fact = -1
                if(ipmw > 0): fact = 1
              if(ipbc and Kstore == 'compress'):
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('Iia,Iia->',
                                                         np.conjugate(X_ia[ipa2]),
                                                         rho1[:,:O2k,O2k:],
                                                         optimize=True)/Nkp   
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('Iji,Iij->',X_ij[ipa2],
                                                         rho1[:,:O2k,:O2k],
                                                         optimize=True)/Nkp 
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('Aba,Aab->',X_ab[ipa2],
                                                         rho1[:,O2k:,O2k:],
                                                         optimize=True)/Nkp   
              else:
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ia,ia->',
                                                         np.conjugate(X_ia[ipa2]),
                                                         rho1[:O2k,O2k:],
                                                         optimize=True)/Nkp   
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ji,ij->',X_ij[ipa2],
                                                         rho1[:O2k,:O2k],
                                                         optimize=True)/Nkp 
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ba,ab->',X_ab[ipa2],
                                                         rho1[O2k:,O2k:],
                                                         optimize=True)/Nkp   
          elif(PertType == "FullOR_V"):
            if(ipa < NP2 and ip < NP1+NP2):
              # Beta contribution
              if(ip < NP1):
                # mu(+)m(-)
                ip1 = ip
                ipa1 = ipa
                ipa2 = ipa + NP1
              elif(ip < NP1+NP2):
                # mu(-)m(+)
                ip1 = ipa
                ipa1 = ip - NP1 
                ipa2 = ipa
              f_static = 1
              if(iw == 0): f_static = 2
              if(ipbc and Kstore == 'compress'):
                tensor[iw,ip1,ipa1] += f_static*np.einsum('Iia,Iia->',
                                                          np.conjugate(X_ia[ipa2]),
                                                          rho1[:,:O2k,O2k:],
                                                          optimize=True)/Nkp   
                tensor[iw,ip1,ipa1] += f_static*np.einsum('Iji,Iij->',X_ij[ipa2],
                                                          rho1[:,:O2k,:O2k],
                                                          optimize=True)/Nkp 
                tensor[iw,ip1,ipa1] += f_static*np.einsum('Aba,Aab->',X_ab[ipa2],
                                                          rho1[:,O2k:,O2k:],
                                                          optimize=True)/Nkp   
              else:
                tensor[iw,ip1,ipa1] += f_static*np.einsum('ia,ia->',
                                                          np.conjugate(X_ia[ipa2]),
                                                          rho1[:O2k,O2k:],
                                                          optimize=True)/Nkp   
                tensor[iw,ip1,ipa1] += f_static*np.einsum('ji,ij->',X_ij[ipa2],
                                                          rho1[:O2k,:O2k],
                                                          optimize=True)/Nkp 
                tensor[iw,ip1,ipa1] += f_static*np.einsum('ba,ab->',X_ab[ipa2],
                                                          rho1[O2k:,O2k:],
                                                          optimize=True)/Nkp   
            elif((ipa >= NP2 and ip < NP1) or (ip >= NP1+NP2 and ipa < NP2)):
              # A contribution
              if(ip < NP1):
                # mu(+)Theta(-)
                ip1 = ip
                ipa1 = ipa - NP2
                ipa2 = ipa + NP1 
              elif(ip >= NP1+NP2):
                # mu(-)Theta(+)
                ip1 = ipa 
                ipa1 = ip - NP1 - NP2 
                ipa2 = ipa 
              f_static = 1
              if(iw == 0): f_static = 2
              if(ipbc and Kstore == 'compress'):
                tensorDQ[iw,ip1,ipa1] += f_static*np.einsum('Iia,Iia->',
                                                            np.conjugate(X_ia[ipa2]),
                                                            rho1[:,:O2k,O2k:],
                                                            optimize=True)/Nkp   
                tensorDQ[iw,ip1,ipa1] += f_static*np.einsum('Iji,Iij->',X_ij[ipa2],
                                                            rho1[:,:O2k,:O2k],
                                                            optimize=True)/Nkp 
                tensorDQ[iw,ip1,ipa1] += f_static*np.einsum('Aba,Aab->',X_ab[ipa2],
                                                            rho1[:,O2k:,O2k:],
                                                            optimize=True)/Nkp   
              else:
                tensorDQ[iw,ip1,ipa1] += f_static*np.einsum('ia,ia->',
                                                            np.conjugate(X_ia[ipa2]),
                                                            rho1[:O2k,O2k:],
                                                            optimize=True)/Nkp   
                tensorDQ[iw,ip1,ipa1] += f_static*np.einsum('ji,ij->',X_ij[ipa2],
                                                            rho1[:O2k,:O2k],
                                                            optimize=True)/Nkp 
                tensorDQ[iw,ip1,ipa1] += f_static*np.einsum('ba,ab->',X_ab[ipa2],
                                                            rho1[O2k:,O2k:],
                                                            optimize=True)/Nkp   
          elif(PertType == "FullOR_L"):
            if(ipa < NP2 and ip < NP1+NP2):
              # Beta contribution
              if(ip < NP1):
                # mu(+)m(-)
                ip1 = ip
                ipa1 = ipa
                ipa2 = ipa + NP1
                fact = 1
                if(ipmw > 0): fact = -1
              elif(ip < NP1+NP2):
                # mu(-)m(+)
                ip1 = ipa
                ipa1 = ip - NP1 
                ipa2 = ipa
                fact = -1
                if(ipmw > 0): fact = 1
              if(ipbc and Kstore == 'compress'):
                tensor[iw,ip1,ipa1] += fact*np.einsum('Iia,Iia->',
                                                      np.conjugate(X_ia[ipa2]),
                                                      rho1[:,:O2k,O2k:],
                                                      optimize=True)/Nkp   
                tensor[iw,ip1,ipa1] += fact*np.einsum('Iji,Iij->',X_ij[ipa2],
                                                      rho1[:,:O2k,:O2k],
                                                      optimize=True)/Nkp 
                tensor[iw,ip1,ipa1] += fact*np.einsum('Aba,Aab->',X_ab[ipa2],
                                                      rho1[:,O2k:,O2k:],
                                                      optimize=True)/Nkp   
              else:
                tensor[iw,ip1,ipa1] += fact*np.einsum('ia,ia->',
                                                      np.conjugate(X_ia[ipa2]),
                                                      rho1[:O2k,O2k:],
                                                      optimize=True)/Nkp   
                tensor[iw,ip1,ipa1] += fact*np.einsum('ji,ij->',X_ij[ipa2],
                                                      rho1[:O2k,:O2k],
                                                      optimize=True)/Nkp 
                tensor[iw,ip1,ipa1] += fact*np.einsum('ba,ab->',X_ab[ipa2],
                                                      rho1[O2k:,O2k:],
                                                      optimize=True)/Nkp   
            elif((ipa >= NP2 and ip < NP1 and ipa<NP2+NP3)
                 or (ip >= NP1+NP2 and ipa < NP2 and ip < NP1+NP2+NP3)):
              # A contribution
              if(ip < NP1):
                # mu(+)Theta(-)
                ip1 = ip
                ipa1 = ipa - NP2
                ipa2 = ipa + NP1 
                fact = 1
                if(ipmw > 0): fact = -1
              elif(ip >= NP1+NP2):
                # mu(-)Theta(+)
                ip1 = ipa 
                ipa1 = ip - NP1 - NP2 
                ipa2 = ipa 
                fact = -1
                if(ipmw > 0): fact = 1
              if(ipbc and Kstore == 'compress'):
                tensorDQ[iw,ip1,ipa1] += fact*np.einsum('Iia,Iia->',
                                                        np.conjugate(X_ia[ipa2]),
                                                        rho1[:,:O2k,O2k:],
                                                        optimize=True)/Nkp   
                tensorDQ[iw,ip1,ipa1] += fact*np.einsum('Iji,Iij->',X_ij[ipa2],
                                                        rho1[:,:O2k,:O2k],
                                                        optimize=True)/Nkp 
                tensorDQ[iw,ip1,ipa1] += fact*np.einsum('Aba,Aab->',X_ab[ipa2],
                                                        rho1[:,O2k:,O2k:],
                                                        optimize=True)/Nkp   
              else:
                tensorDQ[iw,ip1,ipa1] += fact*np.einsum('ia,ia->',
                                                        np.conjugate(X_ia[ipa2]),
                                                        rho1[:O2k,O2k:],
                                                        optimize=True)/Nkp   
                tensorDQ[iw,ip1,ipa1] += fact*np.einsum('ji,ij->',X_ij[ipa2],
                                                        rho1[:O2k,:O2k],
                                                        optimize=True)/Nkp 
                tensorDQ[iw,ip1,ipa1] += fact*np.einsum('ba,ab->',X_ab[ipa2],
                                                        rho1[O2k:,O2k:],
                                                        optimize=True)/Nkp   
            elif((ipa >= NP2+NP3 and ip < NP1) or (ip >= NP1+NP2+NP3 and ipa < NP2)):
              # alpha(L,V) contribution
              if(ip < NP1):
                # mu_L(+)mu_V(-)
                ip1 = ip
                ipa1 = ipa - NP2 - NP3
                ipa2 = ipa + NP1 
                fact = 1
                if(ipmw > 0): fact = -1
              elif(ip >= NP1+NP2+NP3):
                # mu_L(-)mu_V(+)
                ip1 = ipa 
                ipa1 = ip - NP1 - NP2 - NP3 
                ipa2 = ipa 
                fact = -1
                if(ipmw > 0): fact = 1
              if(ipbc and Kstore == 'compress'):
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('Iia,Iia->',
                                                         np.conjugate(X_ia[ipa2]),
                                                         rho1[:,:O2k,O2k:],
                                                         optimize=True)/Nkp   
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('Iji,Iij->',X_ij[ipa2],
                                                         rho1[:,:O2k,:O2k],
                                                         optimize=True)/Nkp 
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('Aba,Aab->',X_ab[ipa2],
                                                         rho1[:,O2k:,O2k:],
                                                         optimize=True)/Nkp   
              else:
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ia,ia->',
                                                         np.conjugate(X_ia[ipa2]),
                                                         rho1[:O2k,O2k:],
                                                         optimize=True)/Nkp   
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ji,ij->',X_ij[ipa2],
                                                         rho1[:O2k,:O2k],
                                                         optimize=True)/Nkp 
                alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ba,ab->',X_ab[ipa2],
                                                         rho1[O2k:,O2k:],
                                                         optimize=True)/Nkp   
  del tx1, tx2
  # Fix OR tensors sign
  if(PertType == "FullOR_V" or PertType == "FullOR_L"):
    tensor *= -1
    tensorDQ *= -1
  #
  # Print the tensor for frequency W
  print_tensor(mol_out,PertType,iw,W,tensor,tensorDQ,alpha_mix,atoms_list)
with open(f"{mol_out}.txt","a") as writer:
  writer.write(f"Total Calculation Time: {time.time()-start0:.2f}s\n")
# Delete scratch files
os.system(f"rm {scratch}/{mol_out}*.npy")
               
