#      CCCCCCCC
#    CCCCCCCCCC                           
#   CCC          RRRRRR                    PPPPPP 
#  CCC   CCCCC   RR   RR                   PP   PP
#  CCC  CC       RR    RR                  PP    PP  
#  CCC  CC       RR   RR    eeee    sssss  PP   PP  yy   yy
#  CCC   CCCCC   RRRRRR    ee  ee  ss      PPPPP     yy yy 
#   CCC          RR   RR   eeeeee   ssss   PP         yyy  
#    CCCCCCCCCC  RR    RR  ee          ss  PP         yy   
#      CCCCCCCC  RR     RR  eeee   sssss   PP        yy    
#
#
# VERSION 1.0.0
# DATE July 29, 2025
# This program is licensed under the terms of the GNU General Public
# License v3.0 or later
#
# Authors: M. Caricato, T. Parsons, J. Abdoullaeva
#
import numpy as np
import os
import sys
import re
import time
import datetime
import resource, platform, tracemalloc

from ccres_read import getFort, getFock, get2e, conMO, getPert
from ccres_funct import mem_check, denom, AmpIt, tau_tildeEq, tauEq, T_interm, t1Eq, t2Eq, E_CCSD, fill_kl, L_Interm, Const_Interm, l1Eq, l2Eq, pert_rhs, tx1Eq, tx2Eq, Xi, TrDen1, print_tensor

#Define molecule
if len(sys.argv)<3:
  print("MISSING MOLECULE NAME")
  exit()
else:
  molecule=sys.argv[1]
# Scratch path
scratch = "/Users/marco/scratch"
# Clean previous outputs
os.system(f"rm {molecule}.txt")
start0=time.time()
tot_mem, avlb_mem = mem_check()
tracemalloc.start()
current_date = datetime.date.today()
current_time = datetime.datetime.now()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"CCResPy PROGRAM \n")
  writer.write(current_date.strftime("%m/%d/%Y "))  
  writer.write(current_time.strftime("%H:%M:%S \n"))
  writer.write(f"Platform: {platform.system()} -- Python v{platform.python_version()} -- NumPy v{np.version.version}\n")
  writer.write(f"Total Memory: {tot_mem:.2f}GB, Available Memory: {avlb_mem:.2f}GB \n")
if(len(sys.argv)>3 and platform.system() == "Linux"):
  mem_limit = int(sys.argv[3])*(1024**3)                         
  resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit)) 
if(platform.system() == "Linux"):
  soft, hard = resource.getrlimit(resource.RLIMIT_AS)            
  soft /= 1024**3                                                
  hard /= 1024**3                                                
  with open(f"{molecule}.txt","a") as writer:                    
    writer.write(f"Soft Memory Limit: {soft:.2f}GB, Hard Memory Limit: {hard:.2f}GB \n") 
# Convergence thresholds on energy and amplitudes
ThrE = 1e-8
ThrA = ThrE*100
# Maximum number of iterations allowed
MaxIt = 1000
# Frozen core
FreezeCore = True
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"\nEnergy convergence threshold: {ThrE:.1e}au -- Max N Iterations: {MaxIt}\n")

# Retrieve various quantities
O, V, FC, FV, NB, scfE, MOCoef_Tot, ipbc, k_weights, atoms_list = getFort(molecule,FreezeCore)
if(ipbc):
  nmtpbc = ipbc[1]
  nrecip = ipbc[9]
  kp, l_list = fill_kl(ipbc)
  Nkp = len(kp)  
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"PBC Information: N-cells: {nmtpbc} -- N-k points: {Nkp}\n")
  if nrecip == 1:
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"                 Gamma-point only\n")
  elif nrecip % 2 != 0 and nrecip != 1:
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"                 Edge and Gamma points are included\n")

# Tensor choice
#PertType = "DipE"
PertType = "DipEV"
#PertType = "OR_V"
#PertType = "OR_L"
#PertType = "FullOR_V"
#PertType = "FullOR_L"
if(ipbc and (PertType == "OR_V" or PertType == "OR_L")):
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"The full OR tensor should be computed for periodic systems\n")
  exit()
#      
# Slice MO coefficient array to remove frozen core orbitals
# tot_mem, avlb_mem = mem_check()
# with open(f"{molecule}.txt","a") as writer:
#   writer.write(f"\n Before MO slicing AvlMem: {avlb_mem:.2f}GB \n")
if(ipbc):
  MOCoef = MOCoef_Tot[:,FC:,:]
else:
  MOCoef = MOCoef_Tot[FC:,:]
# tot_mem, avlb_mem = mem_check()
# with open(f"{molecule}.txt","a") as writer:
#   writer.write(f"\n After MO slicing AvlMem: {avlb_mem:.2f}GB \n")
#  
# Get Fock matrix in MO basis 
Fock = getFock(molecule,O,V,NB,ipbc,"MO",False,MOCoef)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"\nRead MO Coeff and Fock Matrix, Time: {time.time()-start0:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")
O2 = O*2
V2 = V*2
NOrb2 = O2 + V2

##########################################################################  
# Get AO 2e integrals and transform in MO basis
##########################################################################  
start=time.time()
# AOInt = get2e(NB,ipbc)
get2e(NB,ipbc,molecule,scratch)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Read AO 2ERI, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")
# Transform to molecular spin-orbital basis
start=time.time()
conMO(molecule,scratch,O,V,NB,ipbc,MOCoef)
# IJKL,IABC,IJAB,IJKA,IABJ = conMO(molecule,scratch,O,V,NB,ipbc,MOCoef)
# IJKL,IABC,IJAB,IJKA,IABJ = conMO(molecule,scratch,O,V,NB,ipbc,MOCoef,AOInt)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"2ERI AO->MO, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")
start=time.time()
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
  kp, l_list = fill_kl(ipbc)
  Nkp = len(kp)
  O2k = O2*Nkp
  V2k = V2*Nkp
  Ok = O*Nkp
  Vk = V*Nkp
  NOrb2k = NOrb2*Nkp
NkpC = Nkp*Nkp*Nkp
# Define denominator arrays
W = 0
D1, D2 =  denom(1,O2,V2,kp,Fock,W)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Compute energy denominators, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
  
##########################################################################  
# CCSD Energy and Amplitudes
##########################################################################
start=time.time()
# Initialize T1 and T2
IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
t1 = np.zeros((O2k,V2k),dtype=Fock.dtype)
t2 = np.conjugate(IJAB)/D2.real
EMP2 = 0.25*np.einsum('ijab,ijab',IJAB,t2,optimize=True)/NkpC
del IJAB
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"T guess, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")

# Solve amplitude equations
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*          SOLVING CCSD T AMPLITUDE EQS.           *\n")
  writer.write("****************************************************\n")
  writer.write(f"E(SCF)= = {scfE.real:.10f}au, DE(MP2) = {EMP2.real:.10f}au"
               f", E(MP2) = {scfE.real+EMP2.real:.10f}au\n")
tau = []
# W_efam = []
# W_iemn = []
# W_mbej = []
# W_mnij = []
W_iemn = np.lib.format.open_memmap(f"{scratch}/{molecule}-Wiemn.npy",
                                   mode='w+',shape=(O2k,V2k,O2k,O2k),
                                   dtype=Fock.dtype) 
W_mbej = np.lib.format.open_memmap(f"{scratch}/{molecule}-Wmbej.npy",
                                   mode='w+',shape=(O2k,V2k,V2k,O2k),
                                   dtype=Fock.dtype) 
W_mnij = np.lib.format.open_memmap(f"{scratch}/{molecule}-Wmnij.npy",
                                   mode='w+',shape=(O2k,O2k,O2k,O2k),
                                   dtype=Fock.dtype) 
W_efam = np.lib.format.open_memmap(f"{scratch}/{molecule}-Wefam.npy",
                                   mode='w+',shape=(V2k,V2k,V2k,O2k),
                                   dtype=Fock.dtype) 
F_ae = []
F_mi = []
F_me = []
t1, t2 = AmpIt("T",molecule,scratch,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
               tau,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,t1,t2,t1,t2,ipbc)
# t1, t2 = AmpIt("T",molecule,scratch,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
#                IJKL,IABC,IJAB,IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,
#                W_mnij,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,t1,t2,t1,t2,ipbc)
# exit()

##########################################################################  
# Compute constant intermediates
##########################################################################  
start=time.time()
tau_tilde = tau_tildeEq(1, Nkp, t1, t2)
tau = tauEq(1, Nkp, t1, t2)
F_ae,F_mi,F_me = T_interm(molecule,scratch,Ok,Vk,Nkp,Fock,t1,t2,tau_tilde,tau)
# F_ae,F_mi,F_me,W_mnij,W_mbej = T_interm(1,Ok,Vk,Nkp,Fock,t1,t2,IJKL,IABC,
#                                         IJAB,IABJ,IJKA,tau_tilde,tau)
if(f"{scratch}/{molecule}-ABCD.npy"): 
  os.system(f"mv {scratch}/{molecule}-ABCD.npy {scratch}/{molecule}-Wabef.npy")
else:
  print(f"ABCD integrals file is missing\n")
  exit()
F_ae,F_mi = Const_Interm(1,molecule,scratch,Nkp,t1,t2,tau,F_ae,F_mi,F_me)
# F_ae,F_mi,W_mbej,W_efam,W_iemn = Const_Interm(1,molecule,scratch,Nkp,t1,t2,
#                                               tau,IJAB,IABJ,IJKA,IABC,F_ae,
#                                               F_mi,F_me,W_mnij,W_mbej)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Constant intermediates evaluated, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
  
##########################################################################  
# CCSD Lambda Amplitudes
##########################################################################
l1 = np.copy(np.conjugate(t1))
l2 = np.copy(np.conjugate(t2))
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*        SOLVING CCSD Lambda AMPLITUDE EQS.        *\n")
  writer.write("****************************************************\n")
l1, l2 = AmpIt("L",molecule,scratch,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
               tau,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,l1,l2,t1,t2,ipbc)
np.save(f"{scratch}/{molecule}-l1",l1)
np.save(f"{scratch}/{molecule}-l2",l2)
del l1, l2
# l1, l2 = AmpIt("L",molecule,scratch,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
#                IJKL,IABC,IJAB,IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,
#                W_mnij,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,l1,l2,t1,t2,ipbc)

##########################################################################  
# CCSD LR equations
##########################################################################
#
# NP = number of perturbations (3 for dipoles and 6 for quadrupoles)
# W = frequency of perturbation
# if W != 0, there two sets of amplitudes per perturbation Tx(+w) and Tx(-w)
# Use same intermediates as in Lambda equations
start=time.time()
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*           COMPUTING CCSD LR FUNCTION             *\n")
  writer.write("****************************************************\n")
NP, NP1, NP2, NP3, NP4, X_ij, X_ia, X_ab = getPert(O,V,NB,ipbc,MOCoef,
                                                   Fock,PertType,molecule)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Perturbation integrals read, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
#exit()
# For now, hardwire frequency of 300 nm or 500nm
Wlist = []
if(PertType == "DipEV" or PertType == "OR_V" or PertType == "FullOR_V"):
  Wlist.append(0.0)
#Wlist.append(0.045563352535238417) # 1000nm
#Wlist.append(0.065090503621769158) # 700nm
#Wlist.append(0.075938920892064027) # 600nm
Wlist.append(0.091126705070476835) # 500nm
#Wlist.append(0.15187784178412805) # 300nm
tensor = np.zeros((len(Wlist), NP1, NP2),dtype=Fock.dtype)
tensorDQ = []
alpha_mix = []
if(PertType == "FullOR_V" or PertType == "FullOR_L"):
  tensorDQ = np.zeros((len(Wlist),NP1, NP3),dtype=Fock.dtype)
if(PertType == "OR_L" or PertType == "FullOR_L"):
  alpha_mix = np.zeros((len(Wlist),NP1, NP1),dtype=Fock.dtype)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Alpha_mix: {np.shape(alpha_mix)}; NP: {NP1},{NP2},{NP3},{NP}\n")
for iw in range(len(Wlist)):
  # Loop over frequencies    
  W = Wlist[iw]
  with open(f"{molecule}.txt","a") as writer:
    writer.write("\n****************************************************\n")
    writer.write(f" Start Linear Response Calculation for Frequency {W:f}\n\n")
  NW = 2
  # tx1 = np.zeros((NP,2,O2k,V2k),dtype=Fock.dtype)
  # tx2 = np.zeros((NP,2,O2k,O2k,V2k,V2k),dtype=Fock.dtype)
  # np.save(f"{scratch}/{molecule}-tx1",tx1)
  # np.save(f"{scratch}/{molecule}-tx2",tx2)
  # del tx1, tx2
  MaxX = np.zeros((NP))
  if (W==0): NW = 1
  tx1 = np.lib.format.open_memmap(f"{scratch}/{molecule}-tx1.npy",mode='w+',
                                  dtype=Fock.dtype, shape=(NP,2,O2k,V2k)) 
  tx2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-tx2.npy",mode='w+',
                                  dtype=Fock.dtype, shape=(NP,2,O2k,O2k,V2k,V2k)) 
  # tx1 = np.load(f"{scratch}/{molecule}-tx1.npy",mmap_mode='r+')
  # tx2 = np.load(f"{scratch}/{molecule}-tx2.npy",mmap_mode='r+')
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
      rhs1, rhs2 = pert_rhs(1, PertSymm, Nkp, O2k, V2k, t1, t2, X_ij[ip,:,:], X_ia[ip,:,:], X_ab[ip,:,:])
      tot_mem, avlb_mem = mem_check()
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"Right hand side evaluated, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"\n Perturbation {PertType}-{ip+1}\n\n")
      for ipmw in range(NW):
        # Loop over +/-omega
        PMW = W
        if (ipmw==1): PMW = -W 
        with open(f"{molecule}.txt","a") as writer:
          writer.write(f" Frequency {PMW:+f}\n")
        # Reset denominators including frequency term and initialize amplitudes
        D1, D2 =  denom(1, O2, V2, kp, Fock, PMW)
        # tx1 = np.load(f"{scratch}/{molecule}-tx1.npy",mmap_mode='r+')
        # tx2 = np.load(f"{scratch}/{molecule}-tx2.npy",mmap_mode='r+')
        ttx1 = tx1[ip,ipmw,:,:]
        ttx2 = tx2[ip,ipmw,:,:,:,:]
        ttx1 -= rhs1/D1.real
        ttx2 -= rhs2/D2.real
        # Amplitudes loop
        ttx1[:,:], ttx2[:,:,:,:] = AmpIt("Tx",molecule,scratch,Ok,Vk,Nkp,
                                         MaxIt,ThrE,ThrA,scfE,Fock,tau,
                                         F_ae,F_mi,F_me,rhs1,rhs2,D1,D2,
                                         t1,t2,t1,t2,ttx1,ttx2,ipbc)
        # ttx1[:,:], ttx2[:,:,:,:] = AmpIt("Tx",molecule,scratch,Ok,Vk,Nkp,
        #                                  MaxIt,ThrE,ThrA,scfE,Fock,IJKL,
        #                                  IABC,IJAB,IABJ,IJKA,tau,W_efam,
        #                                  W_iemn,W_mbej,W_mnij,F_ae,F_mi,
        #                                  F_me,rhs1,rhs2,D1,D2,t1,t2,l1,l2,
        #                                  ttx1,ttx2,ipbc)
        del ttx1, ttx2
      #   tx1[ip,ipmw,:,:] -= rhs1/D1.real
      #   tx2[ip,ipmw,:,:,:,:] -= rhs2/D2.real
      #   # Amplitudes loop
      #   tx1[ip,ipmw,:,:], tx2[ip,ipmw,:,:,:,:] = AmpIt("Tx",molecule,scratch,Ok,Vk,Nkp,
      #                                                  MaxIt,ThrE,ThrA,scfE,Fock,IJKL,
      #                                                  IABC,IJAB,IABJ,IJKA,tau,W_efam,
      #                                                  W_iemn,W_mbej,W_mnij,F_ae,F_mi,
      #                                                  F_me,rhs1,rhs2,D1,D2,t1,t2,l1,l2,
      #                                                  tx1[ip,ipmw,:,:],
      #                                                  tx2[ip,ipmw,:,:,:,:],ipbc)
      #   del tx1, tx2
      if(NW == 1):
        # This is a static case. Make a copy of the amplitudes for the -W case.
        tx1[ip,1,:,:] = np.copy(tx1[ip,0,:,:])
        tx2[ip,1,:,:,:,:] = np.copy(tx2[ip,0,:,:,:,:])
  del tx1, tx2
  #
  # Now that we have all the Tx amplitudes for this W, we can compute
  # the corresponding Xi amplitudes and contract with all other Tx
  # amplitudes, and the transition 1PDM-like rho1 and contract with
  # the perturbation integrals
  #
  # Reset denominators
  D1, D2 =  denom(1, O2, V2, kp, Fock, 0)
  tx1 = np.load(f"{scratch}/{molecule}-tx1.npy",mmap_mode='r')
  tx2 = np.load(f"{scratch}/{molecule}-tx2.npy",mmap_mode='r')
  for ip in range(NP):
    if(MaxX[ip] > 1e-15):
      # Evaluate Xi amplitudes 
      start=time.time()
      l1 = np.load(f"{scratch}/{molecule}-l1.npy",mmap_mode='r')
      l2 = np.load(f"{scratch}/{molecule}-l2.npy",mmap_mode='r')
      ttx1 = tx1[ip,0,:,:]
      ttx2 = tx2[ip,0,:,:,:,:]
      Xi1, Xi2 = Xi(1,molecule,scratch,Nkp,O2k,ttx1,ttx2,l1,l2,t1,
                    F_ae,F_mi,F_me,D2)
      # Xi1, Xi2 = Xi(1,Nkp,ttx1,ttx2,l1,l2,t1,IABC,IJAB,IJKA,F_ae,F_mi,F_me,
      #               W_mbej,D2)
      del ttx1, ttx2, l1, l2
      # Xi1, Xi2 = Xi(1,Nkp,tx1[ip,0,:,:],tx2[ip,0,:,:,:,:],l1,l2,t1,IABC,IJAB,
      #               IJKA,F_ae,F_mi,F_me,W_mbej,D2)
      # del tx1, tx2
      tot_mem, avlb_mem = mem_check()
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"Xi terms evaluated, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
      for ipa in range(NP2+NP3+NP4):
        # Contract Xi(ip) with Tx(ipa)
        if(PertType == "DipE" or PertType == "DipEV"):
          ttx1 = tx1[ipa,1,:,:]
          ttx2 = tx2[ipa,1,:,:,:,:]
          tensor[iw,ip,ipa] -= np.einsum('ia,ia->',Xi1,ttx1,optimize=True)/Nkp 
          tensor[iw,ip,ipa] -= 0.25*np.einsum('ijab,ijab->',Xi2,
                                              ttx2,optimize=True)/NkpC
          del ttx1, ttx2
          # tx1 = np.load(f"{scratch}/{molecule}-tx1.npy",mmap_mode='r')
          # tx2 = np.load(f"{scratch}/{molecule}-tx2.npy",mmap_mode='r')
          # tensor[iw,ip,ipa] -= np.einsum('ia,ia->',Xi1,tx1[ipa,1,:,:],
          #                                optimize=True)/Nkp 
          # tensor[iw,ip,ipa] -= 0.25*np.einsum('ijab,ijab->',Xi2,
          #                                     tx2[ipa,1,:,:,:,:],
          #                                     optimize=True)/NkpC
          # del tx1, tx2
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
          ttx1 = tx1[ipa2,1,:,:]
          ttx2 = tx2[ipa2,1,:,:,:,:]
          tensor[iw,ip1,ipa1] -= np.einsum('ia,ia->',Xi1,ttx1,optimize=True)/Nkp 
          tensor[iw,ip1,ipa1] -= 0.25*np.einsum('ijab,ijab->',Xi2,
                                                ttx2,optimize=True)/NkpC
          del ttx1, ttx2
          # tx1 = np.load(f"{scratch}/{molecule}-tx1.npy",mmap_mode='r')
          # tx2 = np.load(f"{scratch}/{molecule}-tx2.npy",mmap_mode='r')
          # tensor[iw,ip1,ipa1] -= np.einsum('ia,ia->',Xi1,
          #                                  tx1[ipa2,1,:,:],optimize=True)/Nkp 
          # tensor[iw,ip1,ipa1] -= 0.25*np.einsum('ijab,ijab->',Xi2,
          #                                       tx2[ipa2,1,:,:,:,:],
          #                                       optimize=True)/NkpC
          # del tx1, tx2
          # if((ip == 2 or ip == 5) and ipa ==2):
          #   xx1 = np.einsum('ia,ia->',Xi1,Xi1,optimize=True)/Nkp
          #   tt1 = np.einsum('ia,ia->',tx1[ipa2,1,:,:],tx1[ipa2,1,:,:],optimize=True)/Nkp
          #   xx2 = np.einsum('ijab,ijab->',Xi2,Xi2,optimize=True)/Nkp
          #   tt2 = np.einsum('ijab,ijab->',tx2[ipa2,1,:,:,:,:],tx2[ipa2,1,:,:,:,:],optimize=True)/Nkp
          #   with open(f"{molecule}.txt","a") as writer:
          #     writer.write(f"ip,ipa,ip1,ipa1,ipa2: {ip},{ipa},{ip1},{ipa1},{ipa2}\n")
          #     writer.write(f"Xi1 = {xx1:+.6f}, Xi2 = {xx2:+.6f}, Tx1 = {tt1:+.6f}, Tx2 = {tt2:+.6f}, Tensor: {tensor[iw,ip1,ipa1]/4}\n")
          # del tx1, tx2
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
            ttx1 = tx1[ipa2,1,:,:]
            ttx2 = tx2[ipa2,1,:,:,:,:]
            tensor[iw,ip1,ipa1] -= fact*np.einsum('ia,ia->',Xi1,ttx1,
                                                  optimize=True)/Nkp 
            tensor[iw,ip1,ipa1] -= 0.25*fact*np.einsum('ijab,ijab->',Xi2,
                                                       ttx2,optimize=True)/NkpC
            del ttx1, ttx2
          elif((ipa >= NP2 and ip < NP1) or (ip >= NP1+NP2 and ipa < NP2)):
            # alpha(l,V) contribution
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
            ttx1 = tx1[ipa2,1,:,:]
            ttx2 = tx2[ipa2,1,:,:,:,:]
            alpha_mix[iw,ip1,ipa1] -= fact*np.einsum('ia,ia->',Xi1,
                                                     ttx1,optimize=True)/Nkp 
            alpha_mix[iw,ip1,ipa1] -= 0.25*fact*np.einsum('ijab,ijab->',Xi2,
                                                          ttx2,optimize=True)/NkpC
            # with open(f"{molecule}.txt","a") as writer:
            #   writer.write(f"Alpha_mix 1: {iw},{ip1},{ipa1},{ip},{ipa} \n {alpha_mix[iw,ip1,ipa1]}\n")
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
            # tot_mem, avlb_mem = mem_check()
            # with open(f"{molecule}.txt","a") as writer:
            #   writer.write(f"Tensor0, AvlMem: {avlb_mem:.2f}GB\n")
            ttx1 = tx1[ipa2,1,:,:]
            ttx2 = tx2[ipa2,1,:,:,:,:]
            # tx1 = np.load(f"{scratch}/{molecule}-tx1.npy",mmap_mode='r')
            # tx2 = np.load(f"{scratch}/{molecule}-tx2.npy",mmap_mode='r')
            # tot_mem, avlb_mem = mem_check()
            # with open(f"{molecule}.txt","a") as writer:
            #   writer.write(f"Tensor1, AvlMem: {avlb_mem:.2f}GB\n")
            tensor[iw,ip1,ipa1] -= np.einsum('ia,ia->',Xi1,ttx1,optimize=True)/Nkp 
            tensor[iw,ip1,ipa1] -= 0.25*np.einsum('ijab,ijab->',Xi2,
                                                  ttx2,optimize=True)/NkpC
            # tensor[iw,ip1,ipa1] -= np.einsum('ia,ia->',Xi1,
            #                                  tx1[ipa2,1,:,:],optimize=True)/Nkp 
            # tensor[iw,ip1,ipa1] -= 0.25*np.einsum('ijab,ijab->',Xi2,
            #                                       tx2[ipa2,1,:,:,:,:],
            #                                       optimize=True)/NkpC
            # tot_mem, avlb_mem = mem_check()
            # with open(f"{molecule}.txt","a") as writer:
            #   writer.write(f"Tensor2, AvlMem: {avlb_mem:.2f}GB\n")
            # if((ip == 2 or ip == 5) and ipa ==2):
            #   xx1 = np.einsum('ia,ia->',Xi1,Xi1,optimize=True)/Nkp
            #   tt1 = np.einsum('ia,ia->',tx1[ipa2,1,:,:],tx1[ipa2,1,:,:],optimize=True)/Nkp
            #   xx2 = np.einsum('ijab,ijab->',Xi2,Xi2,optimize=True)/Nkp
            #   tt2 = np.einsum('ijab,ijab->',tx2[ipa2,1,:,:,:,:],tx2[ipa2,1,:,:,:,:],optimize=True)/Nkp
            #   with open(f"{molecule}.txt","a") as writer:
            #     writer.write(f"ip,ipa,ip1,ipa1,ipa2: {ip},{ipa},{ip1},{ipa1},{ipa2}\n")
            #     writer.write(f"Xi1 = {xx1:+.6f}, Xi2 = {xx2:+.6f}, Tx1 = {tt1:+.6f}, Tx2 = {tt2:+.6f}, Tensor: {tensor[iw,ip1,ipa1]/4}\n")
            # del tx1, tx2
            del ttx1, ttx2
            # tot_mem, avlb_mem = mem_check()
            # with open(f"{molecule}.txt","a") as writer:
            #   writer.write(f"Tensor3, AvlMem: {avlb_mem:.2f}GB\n")
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
            # tot_mem, avlb_mem = mem_check()
            # with open(f"{molecule}.txt","a") as writer:
            #   writer.write(f"TensorDQ0, AvlMem: {avlb_mem:.2f}GB, t2: {np.size(t2)/1024**3:.2f}GB\n")
            # tx1 = np.load(f"{scratch}/{molecule}-tx1.npy",mmap_mode='r')
            # tx2 = np.load(f"{scratch}/{molecule}-tx2.npy",mmap_mode='r')
            ttx1 = tx1[ipa2,1,:,:]
            ttx2 = tx2[ipa2,1,:,:,:,:]
            # tot_mem, avlb_mem = mem_check()
            # with open(f"{molecule}.txt","a") as writer:
            #   writer.write(f"TensorDQ1, AvlMem: {avlb_mem:.2f}GB, tx2: {np.size(ttx2)/1024**3:.2f}GB\n")
            tensorDQ[iw,ip1,ipa1] -= np.einsum('ia,ia->',Xi1,
                                               ttx1,optimize=True)/Nkp 
            tensorDQ[iw,ip1,ipa1] -= 0.25*np.einsum('ijab,ijab->',Xi2,
                                                    ttx2,optimize=True)/NkpC
            # tensorDQ[iw,ip1,ipa1] -= np.einsum('ia,ia->',Xi1,
            #                                    tx1[ipa2,1,:,:],optimize=True)/Nkp 
            # tensorDQ[iw,ip1,ipa1] -= 0.25*np.einsum('ijab,ijab->',Xi2,
            #                                         tx2[ipa2,1,:,:,:,:],
            #                                         optimize=True)/NkpC
            # tot_mem, avlb_mem = mem_check()
            # with open(f"{molecule}.txt","a") as writer:
            #   writer.write(f"TensorDQ2, AvlMem: {avlb_mem:.2f}GB\n")
            # with open(f"{molecule}.txt","a") as writer:
            #   writer.write(f"ip,ipa,ip1,ipa1,ipa2: {ip},{ipa},{ip1},{ipa1},{ipa2}, tensorDQ: {tensorDQ[iw,ip1,ipa1]/4}, Tensor[3,3]: {tensor[iw,2,2]/4}\n")
            # del ttx1, ttx2, tx1, tx2
            del ttx1, ttx2
            # tot_mem, avlb_mem = mem_check()
            # with open(f"{molecule}.txt","a") as writer:
            #   writer.write(f"TensorDQ3, AvlMem: {avlb_mem:.2f}GB\n")
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
            ttx1 = tx1[ipa2,1,:,:]
            ttx2 = tx2[ipa2,1,:,:,:,:]
            tensor[iw,ip1,ipa1] -= fact*np.einsum('ia,ia->',Xi1,ttx1,optimize=True)/Nkp 
            tensor[iw,ip1,ipa1] -= 0.25*fact*np.einsum('ijab,ijab->',Xi2,
                                                       ttx2,optimize=True)/NkpC
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
            ttx1 = tx1[ipa2,1,:,:]
            ttx2 = tx2[ipa2,1,:,:,:,:]
            tensorDQ[iw,ip1,ipa1] -= fact*np.einsum('ia,ia->',Xi1,
                                                    ttx1,optimize=True)/Nkp 
            tensorDQ[iw,ip1,ipa1] -= 0.25*fact*np.einsum('ijab,ijab->',Xi2,
                                                         ttx2,optimize=True)/NkpC
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
            ttx1 = tx1[ipa2,1,:,:]
            ttx2 = tx2[ipa2,1,:,:,:,:]
            alpha_mix[iw,ip1,ipa1] -= fact*np.einsum('ia,ia->',Xi1,
                                                     ttx1,optimize=True)/Nkp 
            alpha_mix[iw,ip1,ipa1] -= 0.25*fact*np.einsum('ijab,ijab->',Xi2,
                                                          ttx2,optimize=True)/NkpC
            del ttx1, ttx2
      del Xi1, Xi2
      # with open(f"{molecule}.txt","a") as writer:
      #   writer.write(f"Tensor[3,3]: {tensor[iw,2,2]/4}\n")
      for ipmw in range(NW):
        # Loop over +/-omega
        # Evaluate 1PDM
        start=time.time()
        ttx1 = tx1[ip,ipmw,:,:]
        ttx2 = tx2[ip,ipmw,:,:,:,:]
        l1 = np.load(f"{scratch}/{molecule}-l1.npy",mmap_mode='r')
        l2 = np.load(f"{scratch}/{molecule}-l2.npy",mmap_mode='r')
        rho1 = TrDen1(1,O2k,NOrb2k,Nkp,ttx1,ttx2,l1,l2,t1,t2)
        del ttx1, ttx2, l1, l2
        # tx1 = np.load(f"{scratch}/{molecule}-tx1.npy",mmap_mode='r')
        # tx2 = np.load(f"{scratch}/{molecule}-tx2.npy",mmap_mode='r')
        # rho1 = TrDen1(1,O2k,NOrb2k,Nkp,tx1[ip,ipmw,:,:],
        #               tx2[ip,ipmw,:,:,:,:],l1,l2,t1,t2)
        # del tx1, tx2
        tot_mem, avlb_mem = mem_check()
        with open(f"{molecule}.txt","a") as writer:
          writer.write(f"Rho evaluated, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
        # with open(f"{molecule}.txt","a") as writer:
        #   writer.write(f"Tensor[3,3]: {tensor[iw,2,2]/4}\n")
        for ipa in range(NP2+NP3+NP4):
          # Contract 1PDM(ip) with Pert(ipa)
          if(PertType == "DipE"):
            tensor[iw,ip,ipa] += np.einsum('ia,ia->',np.conjugate(X_ia[ipa,:,:]),
                                           rho1[:O2k,O2k:],optimize=True)/Nkp   
            tensor[iw,ip,ipa] += np.einsum('ij,ij->',np.conjugate(X_ij[ipa,:,:]),
                                           rho1[:O2k,:O2k],optimize=True)/Nkp 
            tensor[iw,ip,ipa] += np.einsum('ab,ab->',np.conjugate(X_ab[ipa,:,:]),
                                           rho1[O2k:,O2k:],optimize=True)/Nkp   
          elif(PertType == "DipEV"):
            f_static = 1
            if(iw == 0): f_static = 2
            tensor[iw,ip,ipa] += f_static*np.einsum('ia,ia->',np.conjugate(X_ia[ipa,:,:]),
                                                    rho1[:O2k,O2k:],
                                                    optimize=True)/Nkp   
            tensor[iw,ip,ipa] += f_static*np.einsum('ji,ij->',X_ij[ipa,:,:],
                                                    rho1[:O2k,:O2k],
                                                    optimize=True)/Nkp 
            tensor[iw,ip,ipa] += f_static*np.einsum('ba,ab->',X_ab[ipa,:,:],
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
            tensor[iw,ip1,ipa1] += f_static*np.einsum('ia,ia->',np.conjugate(X_ia[ipa2,:,:]),
                                                      rho1[:O2k,O2k:],
                                                      optimize=True)/Nkp   
            tensor[iw,ip1,ipa1] += f_static*np.einsum('ji,ij->',X_ij[ipa2,:,:],
                                                      rho1[:O2k,:O2k],
                                                      optimize=True)/Nkp 
            tensor[iw,ip1,ipa1] += f_static*np.einsum('ba,ab->',X_ab[ipa2,:,:],
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
              tensor[iw,ip1,ipa1] += fact*np.einsum('ia,ia->',np.conjugate(X_ia[ipa2,:,:]),
                                                    rho1[:O2k,O2k:],
                                                    optimize=True)/Nkp   
              tensor[iw,ip1,ipa1] += fact*np.einsum('ji,ij->',X_ij[ipa2,:,:],
                                                    rho1[:O2k,:O2k],
                                                    optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] += fact*np.einsum('ba,ab->',X_ab[ipa2,:,:],
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
              alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ia,ia->',np.conjugate(X_ia[ipa2,:,:]),
                                                       rho1[:O2k,O2k:],
                                                       optimize=True)/Nkp   
              alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ji,ij->',X_ij[ipa2,:,:],
                                                       rho1[:O2k,:O2k],
                                                       optimize=True)/Nkp 
              alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ba,ab->',X_ab[ipa2,:,:],
                                                       rho1[O2k:,O2k:],
                                                       optimize=True)/Nkp   
              # with open(f"{molecule}.txt","a") as writer:
              #   writer.write(f"Alpha_mix 2: {iw},{ip1},{ipa1},{ip},{ipa} \n {alpha_mix[iw,ip1,ipa1]}\n")
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
              tensor[iw,ip1,ipa1] += f_static*np.einsum('ia,ia->',np.conjugate(X_ia[ipa2,:,:]),
                                                        rho1[:O2k,O2k:],
                                                        optimize=True)/Nkp   
              tensor[iw,ip1,ipa1] += f_static*np.einsum('ji,ij->',X_ij[ipa2,:,:],
                                                        rho1[:O2k,:O2k],
                                                        optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] += f_static*np.einsum('ba,ab->',X_ab[ipa2,:,:],
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
              tensorDQ[iw,ip1,ipa1] += f_static*np.einsum('ia,ia->',np.conjugate(X_ia[ipa2,:,:]),
                                                          rho1[:O2k,O2k:],
                                                          optimize=True)/Nkp   
              tensorDQ[iw,ip1,ipa1] += f_static*np.einsum('ji,ij->',X_ij[ipa2,:,:],
                                                          rho1[:O2k,:O2k],
                                                          optimize=True)/Nkp 
              tensorDQ[iw,ip1,ipa1] += f_static*np.einsum('ba,ab->',X_ab[ipa2,:,:],
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
              tensor[iw,ip1,ipa1] += fact*np.einsum('ia,ia->',np.conjugate(X_ia[ipa2,:,:]),
                                                    rho1[:O2k,O2k:],
                                                    optimize=True)/Nkp   
              tensor[iw,ip1,ipa1] += fact*np.einsum('ji,ij->',X_ij[ipa2,:,:],
                                                    rho1[:O2k,:O2k],
                                                    optimize=True)/Nkp 
              tensor[iw,ip1,ipa1] += fact*np.einsum('ba,ab->',X_ab[ipa2,:,:],
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
              tensorDQ[iw,ip1,ipa1] += fact*np.einsum('ia,ia->',np.conjugate(X_ia[ipa2,:,:]),
                                                      rho1[:O2k,O2k:],
                                                      optimize=True)/Nkp   
              tensorDQ[iw,ip1,ipa1] += fact*np.einsum('ji,ij->',X_ij[ipa2,:,:],
                                                      rho1[:O2k,:O2k],
                                                      optimize=True)/Nkp 
              tensorDQ[iw,ip1,ipa1] += fact*np.einsum('ba,ab->',X_ab[ipa2,:,:],
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
              alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ia,ia->',np.conjugate(X_ia[ipa2,:,:]),
                                                       rho1[:O2k,O2k:],
                                                       optimize=True)/Nkp   
              alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ji,ij->',X_ij[ipa2,:,:],
                                                       rho1[:O2k,:O2k],
                                                       optimize=True)/Nkp 
              alpha_mix[iw,ip1,ipa1] += fact*np.einsum('ba,ab->',X_ab[ipa2,:,:],
                                                       rho1[O2k:,O2k:],
                                                       optimize=True)/Nkp   
          # with open(f"{molecule}.txt","a") as writer:
          #   writer.write(f"ipmw: {ipmw}, ipa: {ipa}, Tensor[3,3]: {tensor[iw,2,2]/4}\n")
          # else:
          #   ip1 = ip
          #   ipa1 = ipa
          #   ipa2 = ipa
          # f_static = 1
          # if((PertType == "DipEV" or PertType == "OR_V") and iw == 0): f_static = 2
          # tensor[iw,ip1,ipa1] += f_static*np.einsum('ia,ia->',np.conjugate(X_ia[ipa2,:,:]),
          #                                           rho1[:O2k,O2k:],optimize=True)/Nkp   
          # if(PertType == "DipE"):
          #   tensor[iw,ip1,ipa1] += f_static*np.einsum('ij,ij->',np.conjugate(X_ij[ipa2,:,:]),
          #                                             rho1[:O2k,:O2k],optimize=True)/Nkp 
          #   tensor[iw,ip1,ipa1] += f_static*np.einsum('ab,ab->',np.conjugate(X_ab[ipa2,:,:]),
          #                                             rho1[O2k:,O2k:],optimize=True)/Nkp   
          # elif(PertType == "DipEV" or PertType == "OR_V"):
          #   tensor[iw,ip1,ipa1] += f_static*np.einsum('ji,ij->',X_ij[ipa2,:,:],
          #                                             rho1[:O2k,:O2k],optimize=True)/Nkp 
          #   tensor[iw,ip1,ipa1] += f_static*np.einsum('ba,ab->',X_ab[ipa2,:,:],
          #                                             rho1[O2k:,O2k:],optimize=True)/Nkp   
  del tx1, tx2
  #
  # Print the tensor for frequency W
  # with open(f"{molecule}.txt","a") as writer:
  #   writer.write(f"Tensor[3,3]: {tensor[iw,2,2]/4}\n")
  print_tensor(molecule,PertType,iw,W,tensor,tensorDQ,alpha_mix)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Total Calculation Time: {time.time()-start0:.2f}s\n")
# Delete scratch files
os.system(f"rm {scratch}/{molecule}*.npy")
               
