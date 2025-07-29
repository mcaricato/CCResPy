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
from ccres_funct import mem_check, denom, AmpIt, tau_tildeEq, tauEq, T_interm, t1Eq, t2Eq, E_CCSD, fill_kl, L_Interm, Const_Interm, l1Eq, l2Eq, pert_rhs, tx1Eq, tx2Eq, Xi, TrDen1

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
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"\nEnergy convergence threshold: {ThrE:.1e}au -- Max N Iterations: {MaxIt}\n")

# Retrieve various quantities
O, V, NB, scfE, MOCoef, ipbc, k_weights = getFort(molecule)
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
Fock = getFock(molecule,O,V,NB,ipbc,"MO",False,MOCoef)
#O, V, NB, scfE, Fock, MOCoef, ipbc, k_weights, Core=getFort(molecule)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"\nRead MO Coeff and Fock Matrix, Time: {time.time()-start0:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")
O2 = O*2
V2 = V*2
NB2 = NB*2

##########################################################################  
# Get AO 2e integrals and transform in MO basis
##########################################################################  
start=time.time()
AOInt = get2e(NB,ipbc)
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Read AO 2ERI, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB \n")
#Change to spin orbital form
start=time.time()
IJKL,IABC,IJAB,IJKA,IABJ = conMO(molecule,scratch,O,V,NB,ipbc,MOCoef,AOInt)
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
NBk = NB
NB2k = NB2
if(ipbc):
  nmtpbc = ipbc[1]
  kp, l_list = fill_kl(ipbc)
  Nkp = len(kp)
  O2k = O2*Nkp
  V2k = V2*Nkp
  Ok = O*Nkp
  Vk = V*Nkp
  NBk = NB*Nkp
  NB2k = NB2*Nkp
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
t1 = np.zeros((O2k,V2k),dtype=Fock.dtype)
t2 = np.conjugate(IJAB)/D2.real
EMP2 = 0.25*np.einsum('ijab,ijab',IJAB,t2,optimize=True)/NkpC
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
W_efam = []
W_iemn = []
W_mbej = []
W_mnij = []
F_ae = []
F_mi = []
F_me = []
t1, t2 = AmpIt("T",molecule,scratch,Ok,Vk,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
               IJKL,IABC,IJAB,IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,
               W_mnij,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,t1,t2,t1,t2,ipbc)

##########################################################################  
# Compute constant intermediates
##########################################################################  
start=time.time()
tau_tilde = tau_tildeEq(1, Nkp, t1, t2)
tau = tauEq(1, Nkp, t1, t2)
F_ae,F_mi,F_me,W_mnij,W_mbej = T_interm(1,Ok,Vk,Nkp,Fock,t1,t2,IJKL,IABC,
                                        IJAB,IABJ,IJKA,tau_tilde,tau)
if(f"{scratch}/{molecule}-ABCD.npy"): 
  os.system(f"mv {scratch}/{molecule}-ABCD.npy {scratch}/{molecule}-Wabef.npy")
else:
  print(f"ABCD integrals file is missing\n")
  exit()
F_ae,F_mi,W_mbej,W_efam,W_iemn = Const_Interm(1,molecule,scratch,Nkp,t1,t2,
                                              tau,IJAB,IABJ,IJKA,IABC,F_ae,
                                              F_mi,F_me,W_mnij,W_mbej)
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
               IJKL,IABC,IJAB,IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,
               W_mnij,F_ae,F_mi,F_me,D1,D2,D1,D2,t1,t2,l1,l2,t1,t2,ipbc)

##########################################################################  
# CCSD LR equations
##########################################################################
#
# NPert = number of perturbations (3 for dipoles and 6 for quadrupoles)
# WPert = frequency of perturbation
# if WPErt != 0, there two sets of amplitudes per perturbation Tx(+w) and Tx(-w)
# Use same intermediates as in Lambda equations
start=time.time()
with open(f"{molecule}.txt","a") as writer:
  writer.write("****************************************************\n")
  writer.write("*           COMPUTING CCSD LR FUNCTION             *\n")
  writer.write("****************************************************\n")
PertType = "DipE"
NP, X_ij, X_ia, X_ab = getPert(O,V,NB,ipbc,MOCoef,Fock,PertType,molecule)
tot_mem, avlb_mem = mem_check()
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Perturbation integrals read, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
# For now, hardwire frequency of 300 nm or 500nm
Wlist = []
#Wlist.append(0.0)
Wlist.append(0.045563352535238417) # 1000nm
Wlist.append(0.065090503621769158) # 700nm
Wlist.append(0.075938920892064027) # 600nm
Wlist.append(0.091126705070476835) # 500nm
Wlist.append(0.15187784178412805) # 300nm
tensor = np.zeros((len(Wlist), NP, NP),dtype=Fock.dtype)
for iw in range(len(Wlist)):
  # Loop over frequencies    
  W = Wlist[iw]
  with open(f"{molecule}.txt","a") as writer:
    writer.write("\n****************************************************\n")
    writer.write(f" Start Linear Response Calculation for Frequency {W:f}\n\n")
  NW = 2
  tx1 = np.zeros((NP,2,O2k,V2k),dtype=Fock.dtype)
  tx2 = np.zeros((NP,2,O2k,O2k,V2k,V2k),dtype=Fock.dtype)
  MaxX = np.zeros((NP))
  if (W==0): NW = 1 
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
      rhs1, rhs2 = pert_rhs(1, Nkp, O2k, V2k, t1, t2, X_ij[ip,:,:], X_ia[ip,:,:], X_ab[ip,:,:])
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
        tx1[ip,ipmw,:,:] -= rhs1/D1.real
        tx2[ip,ipmw,:,:,:,:] -= rhs2/D2.real
        # Amplitudes loop
        tx1[ip,ipmw,:,:], tx2[ip,ipmw,:,:,:,:] = AmpIt("Tx",molecule,scratch,Ok,Vk,Nkp,
                                                       MaxIt,ThrE,ThrA,scfE,Fock,IJKL,
                                                       IABC,IJAB,IABJ,IJKA,tau,W_efam,
                                                       W_iemn,W_mbej,W_mnij,F_ae,F_mi,
                                                       F_me,rhs1,rhs2,D1,D2,t1,t2,l1,l2,
                                                       tx1[ip,ipmw,:,:],
                                                       tx2[ip,ipmw,:,:,:,:],ipbc)
  #
  # Now that we have all the Tx amplitudes for this W, we can compute
  # the corresponding Xi amplitudes and contract with all other Tx
  # amplitudes, and the transition 1PDM-like rho1 and contract with
  # the perturbation integrals
  #
  # Reset denominators
  D1, D2 =  denom(1, O2, V2, kp, Fock, 0)
  for ip in range(NP):
    if(MaxX[ip] > 1e-15):
      # Evaluate Xi amplitudes 
      start=time.time()
      Xi1, Xi2 = Xi(1,Nkp,tx1[ip,0,:,:],tx2[ip,0,:,:,:,:],l1,l2,t1,IABC,IJAB,IJKA,F_ae,F_mi,
                    F_me,W_mbej,D2)
      tot_mem, avlb_mem = mem_check()
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"Xi terms evaluated, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
      for ipa in range(NP):
        # Contract Xi(ip) with Tx(ipa)
        tensor[iw,ip,ipa] -= np.einsum('ia,ia->',Xi1,tx1[ipa,1,:,:],optimize=True)/Nkp 
        tensor[iw,ip,ipa] -= 0.25*np.einsum('ijab,ijab->',Xi2,tx2[ipa,1,:,:,:,:],optimize=True)/NkpC
      del Xi1, Xi2
      for ipmw in range(NW):
        # Loop over +/-omega
        # Evaluate 1PDM
        start=time.time()
        rho1 = TrDen1(1,O2k,NB2k,Nkp,tx1[ip,ipmw,:,:],tx2[ip,ipmw,:,:,:,:],l1,l2,t1,t2)
        tot_mem, avlb_mem = mem_check()
        with open(f"{molecule}.txt","a") as writer:
          writer.write(f"Rho evaluated, Time: {time.time()-start:.2f}s, AvlMem: {avlb_mem:.2f}GB\n")
        for ipa in range(NP):
          # Contract 1PDM(ip) with Pert(ipa)
          tensor[iw,ip,ipa] += np.einsum('ij,ij->',np.conjugate(X_ij[ipa,:,:]),rho1[:O2k,:O2k],optimize=True)/Nkp 
          tensor[iw,ip,ipa] += np.einsum('ia,ia->',np.conjugate(X_ia[ipa,:,:]),rho1[:O2k,O2k:],optimize=True)/Nkp   
          tensor[iw,ip,ipa] += np.einsum('ab,ab->',np.conjugate(X_ab[ipa,:,:]),rho1[O2k:,O2k:],optimize=True)/Nkp   
  # Print the tensor for frequency W
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"\n DipE(LG)-DipE(LG) Polarizability in a.u. for W = {W:.6f} a.u.\n")
  for ip in range(NP):
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
with open(f"{molecule}.txt","a") as writer:
  writer.write(f"Total Calculation Time: {time.time()-start0:.2f}s\n")
# Delete scratch files
os.system(f"rm {scratch}/{molecule}*.npy")
               
