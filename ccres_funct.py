############################################################################
#
# This file contains most of the functions used by the main CCResPy program
# v1.0.0
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
import resource, platform, psutil, tracemalloc
from scipy.constants import hbar, c, m_e, N_A
#
##########################################################################
# Function to return total and available memory in GB
##########################################################################
def mem_check():
  memory = psutil.virtual_memory()
  tot_mem = memory.total/(1024**3)
  avlb_mem = memory.available/(1024**3)
  used, peak = tracemalloc.get_traced_memory()
  used /= 1024**3
  peak /= 1024**3
  if(platform.system() == "Linux"):            
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)            
    soft /= 1024**3                                                
    hard /= 1024**3                                                
    if(hard > 0):
      avlb_mem = hard - used
      tot_mem = hard
  return tot_mem, avlb_mem

##########################################################################
# Compute energy denominmator over all orbitals
##########################################################################
def DEk(NOrb2k, OrbE):
  # This is used to compute the U matrix in dC/dk = UC
  # Orbital energies are assumed to be real and stored as NOrb*2*Nkp
  DE = np.ones((NOrb2k,NOrb2k))
  for p in range(NOrb2k):
    for q in range(NOrb2k):
      DE[p,q]=OrbE[p]-OrbE[q]
      # Set small values to a large number to quelch the UMat value to 0
      if(abs(DE[p,q]) < 1.e-7): DE[p,q] = 1.e20
  return DE

##########################################################################
# Compute energy denominators
##########################################################################
# def denom(T, O2, V2, kp, Fock, W):
def denom(T, O2, V2, ipbc, Fock, W):
  if T==1:
    NB2 = O2+V2
    # if(kp):
    if(ipbc):
      # PBC
      npdir = ipbc[0]
      ndimk = ipbc[12:15]
      Nkp, _, _ = fill_kl(ipbc)
      map_kp = form_map_kp(npdir,ndimk)
      # Nkp = len(kp)
      O2k = O2*Nkp
      V2k = V2*Nkp
      # kp2 = kp
    else:
      # Molecular
      Nkp = 1
      O2k = O2
      V2k = V2
      # kp2 = np.zeros((1))
    # pi2 = round(2*np.pi,10)
    D1 = np.ones((Nkp,O2,Nkp,V2),dtype=Fock.dtype)
    D2 = np.ones((Nkp,O2,Nkp,O2,Nkp,V2,Nkp,V2),dtype=Fock.dtype)
    Fock = Fock.reshape((Nkp,NB2,Nkp,NB2))
    # D1 denominator
    NksumS = 0
    ik = np.arange(Nkp)
    ii = np.arange(O2)
    ia = np.arange(V2) + O2
    occ_diag = Fock[ik[:,None],ii[None,:],ik[:,None],ii[None,:]]  # (Nkp, O2)
    vir_diag = Fock[ik[:,None],ia[None,:],ik[:,None],ia[None,:]]  # (Nkp, V2)
    D1[ik,:,ik,:] = occ_diag[:,:,None] - vir_diag[:,None,:] - W    
    # for n in range(Nkp):
    #   for a in range(V2):
    #     for i in range(O2):
    #       D1[n,i,n,a]=Fock[n,i,n,i]-Fock[n,a+O2,n,a+O2] - W
    D1 = D1.reshape((O2k,V2k))
    # if(NksumS != Nkp):
    #   print(f"Issue with k-point count for singles denominator: {NksumS} != {Nkp} ")
    #   exit()
    # D2 denominator
    # NksumD = 0
    for n in range(Nkp):
      for k in range(Nkp):
        for h in range(Nkp):
          g = momentum_cons(npdir,ndimk,map_kp,3,n,k,h,0)
          for i in range(O2):
            deni = Fock[n,i,n,i] 
            for j in range(O2):
              denj = deni + Fock[k,j,k,j]
              for a in range(V2):
                dena = denj - Fock[h,a+O2,h,a+O2]
                for b in range(V2):
                  D2[n,i,k,j,h,a,g,b] = dena - Fock[g,b+O2,g,b+O2] - W
                      # denval = D2[n,i,k,j,h,a,g,b]
                      # if(abs(denval) < 1e-8):
                      #   print(f"DenVal={denval},{n},{i},{k},{j},{h},{a},{g},{b}\n {Fock[n,i,n,i]},{Fock[k,j,k,j]},{Fock[h,a+O2,h,a+O2]},{Fock[g,b+O2,g,b+O2]}")
    # if(NksumD != Nkp*Nkp*Nkp):
    #   print(f"Issue with k-point count for singles denominator: {NksumD} != {Nkp*Nkp*Nkp} ")
    #   exit()
    D2 = D2.reshape((O2k,O2k,V2k,V2k))
    Fock = Fock.reshape((Nkp*NB2,Nkp*NB2))
    return D1, D2

##########################################################################
# Compute energy denominators with 3Nkp storage
##########################################################################
def denom3k(O2,V2,ipbc,Ktable,Fock,W):
  NB2 = O2+V2
  if(ipbc):
    # PBC
    npdir = ipbc[0]
    ndimk = ipbc[12:15]
    Nkp, _, _ = fill_kl(ipbc)
    map_kp = form_map_kp(npdir,ndimk)
  else:
    # Molecular
    Nkp = 1
  D1 = np.ones((Nkp,O2,V2),dtype=Fock.dtype)
  D2 = np.ones((Nkp,Nkp,Nkp,O2,O2,V2,V2),dtype=Fock.dtype)
  idx_p = np.arange(NB2)
  eps = Fock[:, idx_p, idx_p]     # (Nkp, O2+V2) -- Fock[k,p,p] for every k,p
  eps_occ = eps[:, :O2]           # (Nkp, O2)
  eps_virt = eps[:, O2:]          # (Nkp, V2)
  # D1 denominator
  # for k in range(Nkp):
  #   for i in range(O2):
  #     for a in range(V2):
  #       D1[k,i,a]=Fock[k,i,i]-Fock[k,a+O2,a+O2] - W
  D1 = (eps_occ[:,:,None] - eps_virt[:,None,:] - W)
  # D2 denominatot
  # for ki in range(Nkp):
  #   for kj in range(Nkp):
  #     for ka in range(Nkp):
  #       kb = momentum_cons(npdir,ndimk,map_kp,3,ki,kj,ka,0)
  #       for i in range(O2):
  #         deni = Fock[ki,i,i] 
  #         for j in range(O2):
  #           denj = deni + Fock[kj,j,j]
  #           for a in range(V2):
  #             dena = denj - Fock[ka,a+O2,a+O2]
  #             for b in range(V2):
  #               D2[ki,kj,ka,i,j,a,b] = dena - Fock[kb,b+O2,b+O2] - W
  EPS_G = eps_virt[Ktable]        # (Nkp,Nkp,Nkp,V2) -- gathered at kb=Ktable[ki,kj,ka]
  D2 = (eps_occ[:, None, None, :, None, None, None]      # ki,i
        + eps_occ[None, :, None, None, :, None, None]    # kj,j
        - eps_virt[None, None, :, None, None, :, None]   # ka,a
        - EPS_G[:, :, :, None, None, None, :]            # ki,kj,ka -> b
        - W)
  # D2 axes: (ki, kj, ka, i, j, a, b), shape (Nkp,Nkp,Nkp,O2,O2,V2,V2)
  return D1, D2

##########################################################################
# Wrapper routine for iterative solution of CCSD amplitude equations
##########################################################################
def AmpIt(AmpType,mol_out,scratch,O,V,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
          tau,F_ae,F_mi,F_me,rhs1,rhs2,D1,D2,t1,t2,l1,l2,tx1,tx2,ipbc,
          Kstore,Ktable,Method):
# def AmpIt(AmpType,mol_out,scratch,O,V,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,IJKL,
#           IABC,IJAB,IABJ,IJKA,tau,W_efam,W_iemn,W_mbej,W_mnij,
#           F_ae,F_mi,F_me,rhs1,rhs2,D1,D2,t1,t2,l1,l2,tx1,tx2,ipbc):
  E_Corr2 = 0
  N = 0
  not_conver = True
  # Setup DIIS arrays
  MaxD = 6
  RepD = 5
  DoDIIS = "F"
  B_mat = np.zeros((MaxD,MaxD),dtype=Fock.dtype)
  e_DIIS = []
  st1 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-DIISa1.npy",
                                  mode='w+',shape=(MaxD,*t1.shape),
                                  dtype=Fock.dtype) 
  st2 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-DIISa2.npy",
                                  mode='w+',shape=(MaxD,*t2.shape),
                                  dtype=Fock.dtype) 
  # Start loop
  start0=time.time()
  with open(f"{mol_out}.txt","a") as writer:
    writer.write(f"Iter.  DIIS     DE-{AmpType}(au)    Delta-DE(au)    Time(s)\n")
  while not_conver and N< MaxIt:
    start = time.time()
    N +=1
    E_Corr1 = E_Corr2
    if(AmpType == "T"):
      # Ground state T amplitudes
      if(N==1):
        # Initialize DIIS amplitudes with guess
        st1[0] = np.copy(t1)
        st2[0] = np.copy(t2)
        del st1, st2
      # Calculate intermediates and perform amplitude iterations
      if(ipbc and Kstore == "compress"):
        tau_tilde = tau_tildeEq3k(Nkp,t1,t2)
        tau = tauEq3k(Nkp,t1,t2,Method)
        F_ae,F_mi,F_me,F_ae2,F_mi2,F_me2 = T_interm3k(mol_out,scratch,O,V,Nkp,Ktable,
                                                      Fock,t1,t2,tau_tilde,tau,Method)
        t1_f = t1Eq3k(mol_out,scratch,O,V,Nkp,Fock,t1,t2,F_ae,F_mi,F_me,D1)
        if(Method == "CCSD"):
          Fae = F_ae
          Fmi = F_mi
          Fme = F_me
        elif(Method == "CC2"):
          Fae = F_ae2
          Fmi = F_mi2
          Fme = F_me2
        t2_f = t2Eq3k(mol_out,scratch,Nkp,Ktable,t1,t2,tau,Fae,Fmi,Fme,D2,Method)
        prod = np.einsum('lae,lae',t1_f,t1_f,optimize=True)
        print(f"t1 = {prod}")
        prod = np.einsum('ijklmnq,ijklmnq',t2_f,t2_f,optimize=True)
        print(f"t2 = {prod}")
      else:
        tau_tilde = tau_tildeEq(Nkp,t1,t2)
        tau = tauEq(Nkp,t1,t2,Method)
        F_ae,F_mi,F_me,F_ae2,F_mi2,F_me2 = T_interm(mol_out,scratch,O,V,Nkp,Fock,t1,t2,
                                                    tau_tilde,tau,Method)
        t1_f = t1Eq(mol_out,scratch,O,V,Nkp,Fock,t1,t2,F_ae,F_mi,F_me,D1)
        if(Method == "CCSD"):
          Fae = F_ae
          Fmi = F_mi
          Fme = F_me
        elif(Method == "CC2"):
          Fae = F_ae2
          Fmi = F_mi2
          Fme = F_me2
        t2_f = t2Eq(mol_out,scratch,Nkp,t1,t2,tau,Fae,Fmi,Fme,D2,Method)
        prod = np.einsum('ae,ae',t1_f,t1_f,optimize=True)
        print(f"t1 = {prod}")
        prod = np.einsum('ijkl,ijkl',t2_f,t2_f,optimize=True)
        print(f"t2 = {prod}")
      del F_ae,F_mi,F_me
      # Check for convergence
      if(ipbc and Kstore == "compress"):
        tau = tauEq3k(Nkp,t1_f,t2_f,"CCSD")
      else:
        tau = tauEq(Nkp, t1_f, t2_f,"CCSD")
      IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
      not_conver,E_Corr2,t1,t2 = AmpConv(AmpType,O,Nkp,t1,t2,t1_f,t2_f,tau,
                                         Fock,D1,IJAB,ThrE,ThrA,E_Corr1,
                                         Kstore)
      del t1_f, t2_f, IJAB
      # DIIS extrapolation
      t1, t2, DoDIIS = DIIS(scratch,mol_out,O,V,N,MaxD,ThrA,RepD,t1,t2,Kstore)
      a1 = t1
      a2 = t2
    elif (AmpType == "L"):
      # Ground state Lambda (or Z) amplitudes
      if(N==1):
        # Initialize DIIS amplitudes with guess
        st1[0] = np.copy(l1)
        st2[0] = np.copy(l2)
        del st1, st2
      if(ipbc and Kstore == "compress"):
        # Calculate intermediates
        G_ae, G_mi = L_Interm3k(Nkp,t2,l2)
        # Amplitude iteration
        l1_f = l1Eq3k(mol_out,scratch,Nkp,Ktable,t1,l1,l2,F_ae,F_mi,F_me,
                      G_ae,G_mi,D1)
        l2_f = l2Eq3k(mol_out,scratch,Nkp,Ktable,t1,l1,l2,F_ae,F_mi,F_me,
                      G_ae,G_mi,D2)
        tau_tilde = tauEq3k(Nkp,l1_f,l2_f)
      else:
        # Calculate intermediates
        G_ae, G_mi = L_Interm(Nkp,t2,l2)
        # Amplitude iteration
        l1_f = l1Eq(mol_out,scratch,Nkp,t1,l1,l2,F_ae,F_mi,F_me,G_ae,G_mi,D1)
        l2_f = l2Eq(mol_out,scratch,Nkp,t1,l1,l2,F_ae,F_mi,F_me,G_ae,G_mi,D2)
        tau_tilde = tauEq(Nkp, l1_f, l2_f)
      # Check for convergence
      IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
      not_conver, E_Corr2, l1, l2 = AmpConv(AmpType,O,Nkp,l1,l2,l1_f,l2_f,
                                            tau_tilde,Fock,D1,IJAB,ThrE,ThrA,
                                            E_Corr1,Kstore)
      del l1_f, l2_f, G_ae, G_mi, IJAB 
      # DIIS extrapolation
      l1, l2, DoDIIS = DIIS(scratch,mol_out,O,V,N,MaxD,ThrA,RepD,l1,l2,Kstore)
      a1 = l1
      a2 = l2
    elif (AmpType == "Tx"):
      # Perturbed T amplitudes
      if(N==1):
        # Initialize DIIS amplitudes with guess
        st1[0] = np.copy(tx1)
        st2[0] = np.copy(tx2)
        del st1, st2
      # Calculate intermediates
      IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
      if(ipbc and Kstore == "compress"):
        G_ae, G_mi = L_Interm3k(Nkp,IJAB,tx2)
        del IJAB
        # Amplitude iteration
        tx1_f = tx1Eq3k(mol_out,scratch,Nkp,Ktable,O,V,tx1,tx2,t1,F_ae,F_mi,
                        F_me,G_ae,G_mi,D1)
        tx2_f = tx2Eq3k(mol_out,scratch,Nkp,Ktable,O,V,tx1,tx2,t1,t2,F_ae,
                        F_mi,F_me,G_ae,G_mi,D2)
        tx1prod = np.einsum('Iia,Iia->',tx1,np.conjugate(tx1),optimize=True)/Nkp
        tx2prod = np.einsum('IJAijab,IJAijab->',tx2,np.conjugate(tx2),optimize=True)/Nkp**3
      else:
        G_ae, G_mi = L_Interm(Nkp,IJAB,tx2)
        del IJAB
        # Amplitude iteration
        tx1_f = tx1Eq(mol_out,scratch,Nkp,O,V,tx1,tx2,t1,F_ae,F_mi,F_me,G_ae,G_mi,D1)
        tx2_f = tx2Eq(mol_out,scratch,Nkp,O,V,tx1,tx2,t1,t2,F_ae,F_mi,F_me,G_ae,G_mi,D2)
        tx1prod = np.einsum('ia,ia->',tx1,np.conjugate(tx1),optimize=True)/Nkp
        tx2prod = np.einsum('ijab,ijab->',tx2,np.conjugate(tx2),optimize=True)/Nkp**3
      print(f"tx1: {tx1prod}, tx2: {tx2prod}")
      tx1_f -= rhs1/D1.real
      tx2_f -= rhs2/D2.real
      # Check for convergence
      not_conver, E_Corr2, tx1, tx2 = AmpConv(AmpType,O,Nkp,tx1,tx2,tx1_f,tx2_f,tau,
                                              Fock,rhs1,rhs2,ThrE,ThrA,E_Corr1,Kstore)
      del tx1_f, tx2_f, G_ae, G_mi 
      # DIIS extrapolation
      tx1, tx2, DoDIIS = DIIS(scratch,mol_out,O,V,N,MaxD,ThrA,RepD,tx1,tx2,Kstore)
      a1 = tx1
      a2 = tx2
    else :
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f"Amplitude type {AmpType} is not implemented. ")
      exit()
    textA = f"{N:4}     {DoDIIS}   {E_Corr2:+.10f}     {E_Corr2-E_Corr1:+.2e}       {time.time()-start:.2f}"
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"{textA}\n")
  if(not_conver):
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"{AmpType} amplitude equations convergence failure\n")
    exit()
  else:
    tot_mem, avlb_mem = mem_check()
    if(AmpType == "T"):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f"E({Method}) = {scfE+E_Corr2:+.10f} au \n")      
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"{AmpType} amplitude equations converged in {time.time()-start0:.2f}s, AvlMem: {avlb_mem:.2f} GB\n\n")
  # Delete DIIS files
  os.system(f"rm {scratch}/{mol_out}-DIISa1.npy")
  os.system(f"rm {scratch}/{mol_out}-DIISa2.npy")
  return a1, a2

##########################################################################
# Evaluate convergence criteria and update amplitudes for amplitude
# iterations
##########################################################################
def AmpConv(AmpType,O,Nkp,a1,a2,a1_f,a2_f,tau,Fock,I1Int,I2Int,ThrE,ThrA,
            E_Corr1,Kstore):
  DiffA1 = abs(np.max(abs(a1_f)-abs(a1)))
  DiffA2 = abs(np.max(abs(a2_f)-abs(a2)))
  a1RMSE = np.sqrt(np.sum((abs(a1_f)-abs(a1))**(2))/np.size(a1))
  a2RMSE = np.sqrt(np.sum((abs(a2_f)-abs(a2))**(2))/np.size(a2))
  a1 = np.copy(a1_f)
  a2 = np.copy(a2_f)
  NkpC = Nkp*Nkp*Nkp
  if(AmpType == "T"):
    # Here I2int should be the IJAB integrals
    E_Corr2 = E_CCSD(O,Nkp,Fock,a1,I2Int,tau,Kstore)
  elif(AmpType == "L"):
    # Here I2int should be the IJAB integrals
    E_Corr2 = E_CCSD(O,Nkp,np.conjugate(Fock),a1,np.conjugate(I2Int),tau,Kstore)
  elif (AmpType == "Tx"):
    # Here I1/2int should be the right hand side perturbations
    if(Kstore == "compress"):
      E_Corr2 = -0.25*np.einsum('IJAijab,IJAijab->',np.conjugate(I2Int),a2,
                                optimize=True)/NkpC
      E_Corr2 -= np.einsum('Iia,Iia->',np.conjugate(I1Int),a1,optimize=True)/Nkp
    else:
      E_Corr2 = -0.25*np.einsum('ijab,ijab->',np.conjugate(I2Int),a2,
                                optimize=True)/NkpC
      E_Corr2 -= np.einsum('ia,ia->',np.conjugate(I1Int),a1,optimize=True)/Nkp
  E_Corr2 = E_Corr2.real
  DiffE = abs(E_Corr2-E_Corr1)
  not_conver = (DiffE> ThrE or DiffA1> ThrA*10 or DiffA2> ThrA*10 or a1RMSE> ThrA or a2RMSE> ThrA)
  # If the amplitudes start oscillating, do a second check on the
  # energy with a tighter criterion
  if(not_conver == True and DiffE < ThrE/100): not_conver = False
  E_Corr1 = E_Corr2
  return not_conver, E_Corr2, a1, a2

##########################################################################
# DIIS Extrapolation
##########################################################################
def DIIS(scratch,mol_out,O,V,Iter,MaxD,Thr,RepD,amp1,amp2,Kstore):
  # Iter: current iteration
  # MaxD: size of the extrapolation space + 1 (for the constraint)
  # Thr: threshold on error to activate DIIS step
  # RepD: perform extrapolation every RepD iterations
  # amp1/2: amplitudes to extrapolate
  # st1/2: saved amplitudes from previous iterations
  # e_DIIS: save errors between iterations
  # B: DIIS matrix
  amp_type = amp1.dtype
  if(amp_type != amp2.dtype):
    print(f"Amplitude type mismatch: a1={amp_type} vs a2={amp2.dtype}")
    exit()
  ThrD = Thr/100
  st1 = np.load(f"{scratch}/{mol_out}-DIISa1.npy",mmap_mode='r+')
  st2 = np.load(f"{scratch}/{mol_out}-DIISa2.npy",mmap_mode='r+')
  if(Iter < MaxD):
    st1[Iter] = np.copy(amp1)
    st2[Iter] = np.copy(amp2)
  else:
    # Shift amplitudes down by 1
    # (The explicit loop is as fast as np.roll but it uses less memory)
    for n in range(MaxD-1):
      st1[n] = np.copy(st1[n+1])
      st2[n] = np.copy(st2[n+1])
    st1[MaxD-1] = np.copy(amp1)
    st2[MaxD-1] = np.copy(amp2)
  del st1, st2
  DoDIIS = "F"
  if Iter%RepD==0:
#  if len1==MaxD and (Iter%RepD==0):
    B = np.zeros((MaxD,MaxD),dtype=amp_type)
    ev1 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ev1.npy",
                                    mode='w+',shape=(MaxD-1,*amp1.shape),
                                    dtype=amp_type) 
    st1 = np.load(f"{scratch}/{mol_out}-DIISa1.npy",mmap_mode='r')
    ev2 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ev2.npy",
                                    mode='w+',shape=(MaxD-1,*amp2.shape),
                                    dtype=amp_type) 
    st2 = np.load(f"{scratch}/{mol_out}-DIISa2.npy",mmap_mode='r')
    if(Kstore == "compress"):
      for l in range(MaxD-1):
        ev1[l] = st1[l+1] - st1[l]
      del st1
      B[:MaxD-1,:MaxD-1] += np.einsum('iklm,jklm->ij',np.conjugate(ev1),ev1,
                                    optimize=True)
      del ev1
      for l in range(MaxD-1):
        ev2[l] = st2[l+1] - st2[l]
      del st2
      B[:MaxD-1,:MaxD-1] += np.einsum('iklmnopq,jklmnopq->ij',np.conjugate(ev2),
                                      ev2,optimize=True)
    else:
      for l in range(MaxD-1):
        ev1[l] = st1[l+1] - st1[l]
      del st1
      B[:MaxD-1,:MaxD-1] += np.einsum('ikl,jkl->ij',np.conjugate(ev1),ev1,optimize=True)
      del ev1
      for l in range(MaxD-1):
        ev2[l] = st2[l+1] - st2[l]
      del st2
      B[:MaxD-1,:MaxD-1] += np.einsum('iklmn,jklmn->ij',np.conjugate(ev2),ev2,optimize=True)
    del ev2
    os.system(f"rm {scratch}/{mol_out}-ev?.npy")
    B[MaxD-1,:] = 1
    B[:,MaxD-1] = 1
    B[MaxD-1,MaxD-1] = 0
    rhs = np.zeros(MaxD)
    rhs[MaxD-1] = 1
    ETest = np.max(abs(B[:MaxD-1,:MaxD-1]))
    csol = np.linalg.solve(B,rhs)
    csum = np.sum(csol[:MaxD-1])
    if(abs(csum-1)>ThrD):
      print(f"Issue with coefficients in DIIS: sum_C = {csum}\n")
      exit()
    st1 = np.load(f"{scratch}/{mol_out}-DIISa1.npy",mmap_mode='r')
    st2 = np.load(f"{scratch}/{mol_out}-DIISa2.npy",mmap_mode='r')
    amp1 = 0
    amp2 = 0
    for p in range(MaxD-1):
      amp1 += st1[p+1] * csol[p]
      amp2 += st2[p+1] * csol[p]
    del st1, st2
    DoDIIS = "T"
  return amp1, amp2, DoDIIS

##########################################################################
# tau_tilde intermediate for CCSD T equations
##########################################################################
def tau_tildeEq(Nkp,t1,t2):
  tau_tilde = np.copy(t2)
  tau_tilde += 0.5*np.einsum('ia,jb->ijab',t1,t1,optimize=True)*Nkp 
  tau_tilde -= 0.5*np.einsum('ib,ja->ijab',t1,t1,optimize=True)*Nkp
  return tau_tilde

##########################################################################
# tau intermediate for CCSD T equations
##########################################################################
def tauEq(Nkp,t1,t2,Method):
  tau = np.einsum('ia,jb->ijab',t1,t1,optimize=True)*Nkp
  tau -= np.einsum('ib,ja->ijab',t1,t1,optimize=True)*Nkp
  if(Method == "CCSD"): tau += np.copy(t2)
  return tau

##########################################################################
# tau_tilde intermediate for CCSD T equations with explicit loops over
# k points
##########################################################################
def tau_tildeEq3k(Nkp,t1,t2):
  tau_tilde = np.copy(t2)
  for I in range(Nkp):
    t1I = t1[I]
    tau_tilde[I,:,I] += 0.5*np.einsum('ia,Jjb->Jijab',t1I,t1,
                                      optimize=True)*Nkp
    tau_tilde[:,I,I] -= 0.5*np.einsum('Jib,ja->Jijab',t1,t1I,
                                      optimize=True)*Nkp
  return tau_tilde

##########################################################################
# tau intermediate for CCSD T equations with explicit loops over k points
##########################################################################
def tauEq3k(Nkp, t1, t2, Method):
  tau = np.zeros(t2.shape,dtype=t2.dtype)
  for I in range(Nkp):      #k_i
    t1I = t1[I]
    tau[I,:,I] += np.einsum('ia,Jjb->Jijab',t1I,t1,optimize=True)*Nkp
    tau[:,I,I] -= np.einsum('Jib,ja->Jijab',t1,t1I,optimize=True)*Nkp
  if(Method == "CCSD"): tau += np.copy(t2)
  return tau

##########################################################################
# F and W intermediates for CCSD T equations
##########################################################################
# def T_interm(T,O,V,Nkp,Fock,t1,t2,IJKL,IABC,IJAB,IABJ,IJKA,tau_tilde,tau):
def T_interm(mol_out,scratch,O,V,Nkp,Fock,t1,t2,tau_tilde,tau,Method):
  # O,V are assumed to be multiplied by Nkp in a PBC calculation
  O2=2*O
  V2=2*V
  NkpS = Nkp*Nkp
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{mol_out}-IABJ.npy",mmap_mode='r')
  IJKL = np.load(f"{scratch}/{mol_out}-IJKL.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{mol_out}-Wmnij.npy",mmap_mode='r+')
  if(Method == "CCSD"): W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r+')
  # if T==1:
  # F_ae
  st_time = time.time()
  F_ae = np.zeros((V2, V2),dtype=Fock.dtype)
  F_ae += (1 - np.eye(V2)) * Fock[O2:, O2:] 
  F_ae -= 0.5 * np.einsum('me,ma->ae', Fock[:O2, O2:], t1, optimize=True)
  F_ae += np.einsum('mf,mafe->ae', t1, IABC, optimize=True)/Nkp
  F_ae -= 0.5 * np.einsum('mnaf,mnef->ae',tau_tilde,IJAB,optimize=True)/NkpS
  prod = np.einsum('ae,ae',F_ae,F_ae,optimize=True)/Nkp
  print(f"Fae = {prod}")
  # F_mi
  F_mi = np.zeros((O2, O2),dtype=Fock.dtype)
  F_mi += (1 - np.eye(O2)) * Fock[:O2, :O2]
  F_mi += 0.5 * np.einsum('ie,me->mi', t1, Fock[:O2, O2:], optimize=True)
  F_mi -= np.einsum('ne,nmie->mi', t1, IJKA, optimize=True)/Nkp
  F_mi += 0.5 * np.einsum('inef,mnef->mi', tau_tilde, IJAB, optimize=True)/NkpS
  prod = np.einsum('ae,ae',F_mi,F_mi,optimize=True)/Nkp
  print(f"Fmi = {prod}")
  # F_me
  F_me = np.zeros((O2, V2),dtype=Fock.dtype)
  F_me = np.copy(Fock[:O2, O2:])
  F_me += np.einsum('nf,mnef->me', t1, IJAB, optimize=True)/Nkp
  prod = np.einsum('ae,ae',F_me,F_me,optimize=True)/Nkp
  print(f"Fme = {prod}")
  # W_mnij
  W_mnij[:,:,:,:] = np.copy(IJKL)
  W_mnij += np.einsum('je,mnie->mnij', t1, IJKA, optimize=True)
  W_mnij -= np.einsum('ie,mnje->mnij', t1, IJKA, optimize=True)
  # F Intermediates for T2 that are only computed during CC2
  F_ae2 = 0 
  F_mi2 = 0
  F_me2 = 0
  if(Method == "CCSD"):
    tot_mem, avlb_mem = mem_check()
    o4gb = np.size(W_mnij)*8/(1024**3)
    if(W_mnij.dtype == complex): o4gb *= 2
    if(avlb_mem < 2*o4gb):
      lenm = W_mnij.shape[0]
      for m in range(lenm):
        W_mnij[m,:,:,:] += 0.5*np.einsum('nef,ijef->nij',IJAB[m,:,:,:],
                                         tau,optimize=True)/Nkp 
    else:
      W_mnij += 0.5 * np.einsum('mnef,ijef->mnij', IJAB, tau, optimize=True)/Nkp
    # W_mbej
    W_mbej[:,:,:,:] = np.copy(IABJ)
    W_mbej += np.einsum('jf,mbef->mbej', t1, IABC, optimize=True)
    W_mbej += np.einsum('nb,mnje->mbej', t1, IJKA, optimize=True)
    W_mbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, IJAB, optimize=True)/Nkp
    W_mbej -= np.einsum('jf,nb,mnef->mbej', t1, t1, IJAB, optimize=True)/Nkp
    prod = np.einsum('ijkl,ijkl',W_mbej,W_mbej,optimize=True)/Nkp**3
    print(f"Wmbej = {prod}")
    del W_mbej
  elif(Method == "CC2"):
    X1 = np.einsum('jf,mnef->mnej',t1,IJAB,optimize=True)
    W_mnij += 0.5 * np.einsum('ie,mnej->mnij',t1,X1,optimize=True)/Nkp
    X1 = np.einsum('je,mnef->mnjf',t1,IJAB,optimize=True)
    W_mnij -= 0.5 * np.einsum('if,mnjf->mnij',t1,X1,optimize=True)/Nkp
    del X1
    # F_ae (t2)
    F_ae2 = np.zeros((V2,V2),dtype=Fock.dtype)
    F_ae2 += (1 - np.eye(V2)) * Fock[O2:,O2:]
    F_ae2 -= 0.5 * np.einsum('me,ma->ae', Fock[:O2,O2:],t1,optimize=True)
    # F_mi (t2)
    F_mi2 = np.zeros((O2,O2),dtype=Fock.dtype)
    F_mi2 += (1 - np.eye(O2)) * Fock[:O2,:O2]
    F_mi2 += 0.5 * np.einsum('ie,me->mi',t1,Fock[:O2,O2:],optimize=True)
    # F_me (t2)
    F_me2 = np.zeros((O2,V2),dtype=Fock.dtype)
    F_me2 = np.copy(Fock[:O2,O2:])
  prod = np.einsum('ijkl,ijkl',W_mnij,W_mnij,optimize=True)/Nkp**3
  print(f"Wmnij = {prod}")
  del IABC, IJAB, IABJ, IJKL, IJKA, W_mnij
  return F_ae, F_mi, F_me, F_ae2, F_mi2, F_me2
  # return F_ae, F_mi, F_me, W_mnij, W_mbej

##########################################################################
# F and W intermediates for CCSD T equations with explicit loops over
# k points
##########################################################################
def T_interm3k(mol_out,scratch,O,V,Nkp,Ktable,Fock,t1,t2,tau_tilde,tau,Method):
  O2=2*O
  V2=2*V
  NkpS = Nkp*Nkp
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{mol_out}-IABJ.npy",mmap_mode='r')
  IJKL = np.load(f"{scratch}/{mol_out}-IJKL.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{mol_out}-Wmnij.npy",mmap_mode='r+')
  if(Method == "CCSD"): W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r+')
  # if T==1:
  # F_ae
  st_time = time.time()
  F_ae = np.zeros((Nkp,V2, V2),dtype=Fock.dtype)
  F_ae += (1 - np.eye(V2)) * Fock[:,O2:, O2:] 
  F_ae -= 0.5 * np.einsum('kme,kma->kae', Fock[:,:O2, O2:],t1,optimize=True)
  F_ae += np.einsum('kmf,knkmafe->nae', t1, IABC, optimize=True)/Nkp
  F_ae -= 0.5 * np.einsum('khlmnaf,khlmnef->lae',tau_tilde,IJAB,optimize=True)/NkpS
  prod = np.einsum('lae,lae',F_ae,F_ae,optimize=True)/Nkp
  print(f"Fae = {prod}")
  # F_mi
  F_mi = np.zeros((Nkp,O2, O2),dtype=Fock.dtype)
  F_mi += (1 - np.eye(O2)) * Fock[:,:O2, :O2]
  F_mi += 0.5 * np.einsum('kie,kme->kmi', t1, Fock[:,:O2, O2:], optimize=True)
  F_mi -= np.einsum('kne,kllnmie->lmi', t1, IJKA, optimize=True)/Nkp
  F_mi += 0.5 * np.einsum('khlinef,khlmnef->kmi', tau_tilde, IJAB, optimize=True)/NkpS
  prod = np.einsum('lae,lae',F_mi,F_mi,optimize=True)/Nkp
  print(f"Fmi = {prod}")
  # F_me
  F_me = np.zeros((Nkp,O2,V2),dtype=Fock.dtype)
  F_me = np.copy(Fock[:,:O2,O2:])
  F_me += np.einsum('knf,lklmnef->lme',t1,IJAB, optimize=True)/Nkp
  prod = np.einsum('lae,lae',F_me,F_me,optimize=True)/Nkp
  print(f"Fme = {prod}")
  # F Intermediates for T2 that are only computed during CC2
  F_ae2 = 0 
  F_mi2 = 0
  F_me2 = 0
  if(Method == "CCSD"):
    W_mnij[:,:,:,:,:,:,:] = np.copy(IJKL)
    W_mbej[:,:,:,:,:,:,:] = np.copy(IABJ)
    #
    X1 = t1[Ktable]
    W_mnij += np.einsum('MNIje,MNImnie->MNImnij',X1,IJKA,optimize=True)
    X2 = np.einsum('MNJie,MNJmnje->MNJmnji',X1,IJKA,optimize=True)
    W_mnij -= transpose_l(X2,Ktable,'k')
    del X2
    #
    W_mbej += np.einsum('hkljf,hklmbef->hklmbej',X1,IABC,optimize=True)
    X2 = transpose_l(IJKA,Ktable,'k')
    W_mbej += np.einsum('knb,hklmnej->hklmbej',t1,X2,optimize=True)
    del X2
    X2 = np.einsum('hkljf,hklmnef->hklmnej',X1,IJAB,optimize=True)
    W_mbej -= np.einsum('knb,hklmnej->hklmbej',t1,X2,optimize=True)/Nkp
    del X1,X2
    #
    ar = np.arange(Nkp)
    idx_ki = ar.reshape(1, Nkp)
    KJJ = Ktable
    for km in range(Nkp):
      KJ = Ktable[km]
      IJABM = IJAB[km]
      tauM = tau[idx_ki,KJ]
      W_mnij[km] += 0.5*np.einsum('hkmnef,hlkijef->hlmnij',IJABM,tauM,
                                  optimize=True)/Nkp
      del IJABM, tauM, KJ
      #
      kn = km
      KF = Ktable[:,kn,:]
      t2NF = t2[:,kn,:][KJJ,KF[:,None,:]]
      IJABN = IJAB[:,kn,:]
      W_mbej -= 0.5*np.einsum('hkljnfb,hlmnef->hklmbej',t2NF,IJABN,
                              optimize=True)/Nkp
      del t2NF,KF,IJABN
    prod = np.einsum('ijklmnq,ijklmnq',W_mbej,W_mbej,optimize=True)/Nkp**3
    print(f"Wmbej = {prod}")
    del W_mbej
  elif(Method == "CC2"):
    W_mnij[:,:,:,:,:,:,:] = np.copy(IJKL)
    X1 = t1[Ktable]
    W_mnij += np.einsum('MNIje,MNImnie->MNImnij',X1,IJKA,optimize=True)
    X2 = np.einsum('MNJie,MNJmnje->MNJmnji',X1,IJKA,optimize=True)
    W_mnij -= transpose_l(X2,Ktable,'k')
    del X2
    X2 = np.einsum('MNIjf,MNImnef->MNImnej',X1,IJAB,optimize=True)
    W_mnij += 0.5 * np.einsum('Iie,MNImnej->MNImnij',t1,X2,optimize=True)/Nkp
    X2 = np.einsum('Jje,MNJmnef->MNJmnjf',t1,IJAB,optimize=True)
    X3 = 0.5 * np.einsum('MNJif,MNJmnjf->MNJmnji',X1,X2,optimize=True)/Nkp 
    W_mnij -= transpose_l(X3,Ktable,'k')
    del X1, X2, X3
    F_ae2 = np.zeros((Nkp,V2,V2),dtype=Fock.dtype)
    F_ae2 += (1 - np.eye(V2)) * Fock[:,O2:, O2:] 
    F_ae2 -= 0.5 * np.einsum('Ame,Ama->Aae', Fock[:,:O2, O2:],t1,optimize=True)
    # F_mi
    F_mi2 = np.zeros((Nkp,O2,O2),dtype=Fock.dtype)
    F_mi2 += (1 - np.eye(O2)) * Fock[:,:O2, :O2]
    F_mi2 += 0.5 * np.einsum('Mie,Mme->Mmi',t1,Fock[:,:O2,O2:],optimize=True)
    # F_me
    F_me2 = np.zeros((Nkp,O2,V2),dtype=Fock.dtype)
    F_me2 = np.copy(Fock[:,:O2,O2:])
  prod = np.einsum('ijklmnq,ijklmnq',W_mnij,W_mnij,optimize=True)/Nkp**3
  print(f"Wmnij = {prod}")
  del IABC, IJAB, IABJ, IJKL, IJKA, W_mnij
  return F_ae, F_mi, F_me, F_ae2, F_mi2, F_me2
  # return F_ae, F_mi, F_me, W_mnij, W_mbej

#########################################################################
# CCSD T1 amplitude equation
#########################################################################
def t1Eq(mol_out,scratch,O,V,Nkp,Fock,t1,t2,F_ae,F_mi,F_me,D1):
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{mol_out}-IABJ.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  NkpS = Nkp*Nkp
  O2=2*O
  t1_f = np.copy(Fock[:O2, O2:])  
  t1_f += np.einsum('ie,ae->ia', t1, F_ae, optimize=True)
  t1_f -= np.einsum('ma,mi->ia', t1, F_mi, optimize=True)
  t1_f += np.einsum('imae,me->ia', t2, F_me, optimize=True)/Nkp
  t1_f -= 0.5 * np.einsum('imef,maef->ia',t2,IABC,optimize=True)/NkpS
  t1_f += 0.5 * np.einsum('mnae,nmie->ia',t2,IJKA,optimize=True)/NkpS
  t1_f += np.einsum('nf,nafi->ia', t1, IABJ,optimize=True)/Nkp
  t1_f /= D1
  del IABC, IABJ, IJKA
  return t1_f

#########################################################################
# CCSD T1 amplitude equation with explicit loops over k points
#########################################################################
def t1Eq3k(mol_out,scratch,O,V,Nkp,Fock,t1,t2,F_ae,F_mi,F_me,D1):
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{mol_out}-IABJ.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  NkpS = Nkp*Nkp
  O2=2*O
  t1_f = np.copy(Fock[:,:O2, O2:])
  t1_f += np.einsum('Iie,Iae->Iia', t1, F_ae, optimize=True)
  t1_f -= np.einsum('Ima,Imi->Iia', t1, F_mi, optimize=True)
  t1_f += np.einsum('IMIimae,Mme->Iia',t2,F_me,optimize=True)/Nkp
  t1_f -= 0.5 * np.einsum('IMEimef,MIEmaef->Iia',t2,IABC,optimize=True)/NkpS
  t1_f += np.einsum('Nnf,NINnafi->Iia',t1,IABJ,optimize=True)/Nkp
  t1_f += 0.5 * np.einsum('MNImnae,NMInmie->Iia',t2,IJKA,optimize=True)/NkpS
  t1_f /= D1
  return t1_f

#########################################################################
# CCSD T2 amplitude equation
#########################################################################
def t2Eq(mol_out,scratch,Nkp,t1,t2,tau,F_ae,F_mi,F_me,D2,Method):
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{mol_out}-IABJ.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  ABCD = np.load(f"{scratch}/{mol_out}-ABCD.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{mol_out}-Wmnij.npy",mmap_mode='r')
  if Method == "CCSD":
    W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  NkpS = Nkp*Nkp
  # Constant term
  t2_f = np.copy(np.conjugate(IJAB))
  del IJAB
  # P(ab) terms
  X1 = F_ae - 0.5*np.einsum('mb,me->be',t1,F_me,optimize=True)
  X2 = np.einsum('ijae,be->ijab',t2,X1,optimize=True)
  X2 -= np.einsum('ma,ijmb->ijab',t1,np.conjugate(IJKA),optimize=True)
  t2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
  del X1, X2, IJKA
  # P(ij) terms
  X1 = F_mi + 0.5*np.einsum('je,me->mj',t1,F_me,optimize=True)
  X2 = -np.einsum('imab,mj->ijab',t2,X1,optimize=True)
  X2 -= np.einsum('ie,jeab->ijab',t1,np.conjugate(IABC),optimize=True)
  t2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
  del X1, X2
  # P(ij,ab) terms
  X1 = -np.einsum('ie,mbej->mbij',t1,IABJ,optimize=True)
  del IABJ
  X2 = np.einsum('ma,mbij->ijab',t1,X1,optimize=True)
  if Method == "CCSD":
    X2 += np.einsum('imae,mbej->ijab',t2,W_mbej,optimize=True)/Nkp
    del W_mbej
  t2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
  t2_f -= np.transpose(X2,axes=(0,1,3,2))
  t2_f += np.transpose(X2,axes=(1,0,3,2))
  del X1, X2
  # tau terms
  if Method == "CCSD":
    t2_f += 0.5*np.einsum('ijef,abef->ijab',tau,ABCD,optimize=True)/Nkp
  elif Method == "CC2":
    X1 = np.einsum('jf,abef->abej',t1,ABCD,optimize=True)
    t2_f += 0.5 * np.einsum('ie,abej->ijab',t1,X1,optimize=True)/Nkp
    X1 = np.einsum('je,abef->abjf',t1,ABCD,optimize=True)
    t2_f -= 0.5 * np.einsum('if,abjf->ijab',t1,X1,optimize=True)/Nkp
    del X1
  del ABCD
  if Method == "CCSD":
    t2_f += 0.5*np.einsum('mnab,mnij->ijab',tau,W_mnij,optimize=True)/Nkp
  elif Method == "CC2":
    X1 = np.einsum('nb,mnij->mbij',t1,W_mnij,optimize=True)
    t2_f += 0.5 * np.einsum('ma,mbij->ijab',t1,X1,optimize=True)/Nkp
    X1 = np.einsum('na,mnij->maij',t1,W_mnij,optimize=True)
    t2_f -= 0.5 * np.einsum('mb,maij->ijab',t1,X1,optimize=True)/Nkp
    del X1
  del W_mnij
  # Add o3v3 work to avoid storing v4 intermediate (it also saves on
  # permutation work)
  if Method == "CCSD":
    X1 = np.einsum('ijef,mbef->ijmb',tau,IABC,optimize=True)/Nkp
  elif Method == "CC2":
    X0 = np.einsum('jf,mbef->mbej',t1,IABC,optimize=True)
    X1 = np.einsum('ie,mbej->ijmb',t1,X0,optimize=True)/Nkp
    X0 = np.einsum('je,mbef->mbjf',t1,IABC,optimize=True)
    X1 -= np.einsum('if,mbjf->ijmb',t1,X0,optimize=True)/Nkp
    del X0
  X2 = -0.5*np.einsum('ma,ijmb->ijab',t1,X1,optimize=True)
  t2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
  del X1, X2, IABC
  t2_f /= D2    
  return t2_f

#########################################################################
# CCSD T2 amplitude equation with explicit loops over k points
#########################################################################
def t2Eq3k(mol_out,scratch,Nkp,Ktable,t1,t2,tau,F_ae,F_mi,F_me,D2,Method):
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{mol_out}-IABJ.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  ABCD = np.load(f"{scratch}/{mol_out}-ABCD.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{mol_out}-Wmnij.npy",mmap_mode='r')
  if Method == "CCSD":
    W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  NkpS = Nkp*Nkp
  # Constant term
  t2_f = np.copy(np.conjugate(IJAB))
  del IJAB
  # P(ij) terms
  X1 = F_mi + 0.5*np.einsum('Jje,Jme->Jmj',t1,F_me,optimize=True)
  X2 = -np.einsum('IJAimab,Jmj->IJAijab',t2,X1,optimize=True)
  X2 -= np.einsum('Iie,JIAjeab->IJAijab',t1,np.conjugate(IABC),
                  optimize=True)
  t2_f += X2 - np.transpose(X2,axes=(1,0,2,4,3,5,6))
  del X1, X2
  # P(ab) terms
  X1 = F_ae - 0.5*np.einsum('Amb,Ame->Abe',t1,F_me,optimize=True)
  X1k = X1[Ktable]
  X2 = np.einsum('IJAijae,IJAbe->IJAijab',t2,X1k,optimize=True)
  del X1, X1k
  X2 -= np.einsum('Ama,IJAijmb->IJAijab',t1,np.conjugate(IJKA),
                  optimize=True)
  t2_f += X2 - transpose_l(X2,Ktable,'k')
  del X2
  #
  # P(ij,ab) terms
  IABJt = transpose_l(IABJ,Ktable,'j')
  X1 = -np.einsum('Iie,AJImjeb->IJAijmb',t1,IABJt,optimize=True)
  X3 = np.einsum('Ama,IJAijmb->IJAijab',t1,X1,optimize=True)
  del X1, IABJt
  #
  if Method == "CCSD":
    # Unavoidable loop over k point
    ar = np.arange(Nkp)
    idx_ki = ar.reshape(Nkp, 1)
    idx_ka = ar.reshape(1, Nkp)
    X2 = np.zeros(t2_f.shape, dtype=t2_f.dtype)
    KB = Ktable
    for M in range(Nkp):
      # W_mnij
      N = Ktable[:,:,M] # N[I,J]
      tauN = tau[M][N]
      WN = W_mnij[M][N,idx_ki]
      # print(f"M:{M}, {N.shape}, {tauN.shape}, {WN.shape}, {t2_f.shape}\n {N}\n")
      t2_f += 0.5*np.einsum('IJmnij,IJAmnab->IJAijab',WN,tauN,optimize=True)/Nkp
      del N,tauN,WN
      # ABCD
      I = M 
      B = Ktable[I]
      tauI = tau[I]
      ABCDI = ABCD[idx_ka,B]  
      t2_f[I] += 0.5*np.einsum('JEijef,JAEabef->JAijab',tauI,ABCDI,
                               optimize=True)/Nkp
      del ABCDI
      # IABC
      IABCI = IABC[idx_ka,B]  
      X1 = np.einsum('JEijef,JAEmbef->JAijmb',tauI,IABCI,
                     optimize=True)/Nkp
      X2[I] -= 0.5*np.einsum('Ama,JAijmb->JAijab',t1,X1,optimize=True)
      del X1,B,tauI,IABCI
      # W_mbej
      E = Ktable[:,M,:]
      IDX_E = E[:,None,:]
      WE = W_mbej[M][KB,IDX_E]
      t2E = t2[:,M,:]
      X3 += np.einsum('IAimae,IJAmbej->IJAijab',t2E,WE,
                      optimize=True)/Nkp
    t2_f += X2 
    t2_f -= transpose_l(X2,Ktable,'k')
    del ar,idx_ki,idx_ka,X2,KB
  elif Method == "CC2":
    # ABCD
    t1k = t1[Ktable]
    X1 = np.einsum('ABIjf,ABIabef->ABIabej',t1k,ABCD,optimize=True)
    X2 = 0.5 * np.einsum('Iie,ABIabej->ABIabij',t1,X1,
                            optimize=True)/Nkp
    X1 = np.einsum('Jje,ABJabef->ABJabjf',t1,ABCD,optimize=True)
    X0 = -0.5 * np.einsum('ABJif,ABJabjf->ABJabji',t1k,X1,
                            optimize=True)/Nkp
    X2 += transpose_l(X0,Ktable,'k')
    del X0, X1
    t2_f += np.transpose(transpose_l(X2,Ktable,'j'),axes=(2,1,0,5,4,3,6))
    del X2
    # W_mnij
    X1 = np.einsum('Bnb,MBImnij->MBImbij',t1,W_mnij,optimize=True)
    X2 = 0.5 * np.einsum('Ama,ABImbij->ABIabij',t1,X1,optimize=True)/Nkp
    X1 = np.einsum('Ana,MAImnij->MAImaij',t1,W_mnij,optimize=True)
    X0 = -0.5 * np.einsum('Bmb,BAImaij->BAIbaij',t1,X1,optimize=True)/Nkp
    X2 += np.transpose(X0,axes=(1,0,2,4,3,5,6))
    del X0, X1
    t2_f += np.transpose(transpose_l(X2,Ktable,'j'),axes=(2,1,0,5,4,3,6))
    del X2
    # IABC
    X0 = np.einsum('MBEjf,MBEmbef->MBEmbej',t1k,IABC,optimize=True)
    X1 = np.einsum('Iie,MBImbej->MBImbij',t1,X0,optimize=True)/Nkp
    X0 = np.einsum('Jje,MBJmbef->MBJmbjf',t1,IABC,optimize=True)
    X2 = -np.einsum('MBJif,MBJmbjf->MBJmbji',t1k,X0,optimize=True)/Nkp
    X1 += transpose_l(X2,Ktable,'k')
    del X0, X2
    X2 = -0.5*np.einsum('Ama,ABImbij->ABIabij',t1,X1,optimize=True)
    X1 = np.transpose(transpose_l(X2,Ktable,'j'),axes=(2,1,0,5,4,3,6))
    t2_f += X1 - transpose_l(X1,Ktable,'k')
    del t1k, X1, X2
  del IABC, ABCD, W_mnij
  #  
  # Final P(ij,ab) permutation
  t2_f += X3 
  t2_f -= np.transpose(X3,axes=(1,0,2,4,3,5,6))
  t2_f -= transpose_l(X3,Ktable,'k')
  t2_f += np.transpose(transpose_l(X3,Ktable,'k'),axes=(1,0,2,4,3,5,6))
  del X3
  t2_f /= D2    
  return t2_f

#########################################################################
# CCSD energy
#########################################################################
def E_CCSD(O,Nkp,Fock,t1,Int2,tau,Kstore):
  O2 = 2*O
  NkpC = Nkp*Nkp*Nkp
  if(Kstore == "compress"):
    E_Corr2_1 = np.einsum('Iia,Iia->',t1,np.conjugate(Fock[:,:O2,O2:]),
                          optimize=True)/Nkp
    E_Corr2_2 = 0.25 * np.einsum('IJAijab,IJAijab->',tau,Int2,
                                 optimize=True)/NkpC
  else:
    E_Corr2_1 = np.einsum('ia,ia->', t1, np.conjugate(Fock[:O2, O2:]),optimize=True)/Nkp
    E_Corr2_2 = 0.25 * np.einsum('ijab,ijab->', tau,Int2,optimize=True)/NkpC
  E_Corr2 = E_Corr2_1.real + E_Corr2_2.real
  return E_Corr2

#########################################################################
# Define constant intermediates for CCSD Lambda and response equations
#########################################################################
def Const_Interm(mol_out,scratch,Nkp,t1,t2,tau,F_ae,F_mi,F_me):
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{mol_out}-IABJ.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{mol_out}-Wmnij.npy",mmap_mode='r')
  W_abef = np.load(f"{scratch}/{mol_out}-Wabef.npy",mmap_mode='r+')
  W_efam = np.load(f"{scratch}/{mol_out}-Wefam.npy",mmap_mode='r+')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r+')
  W_iemn = np.load(f"{scratch}/{mol_out}-Wiemn.npy",mmap_mode='r+')
  # Remember that the contraction for Lambda is over the opposite
  # one or two indices (same for W_mnij)
  F_ae -= 0.5*np.einsum('ma,me->ae',t1,F_me,optimize=True)    
  # The sign of this terms is wrong in Gauss' paper
  F_mi += 0.5*np.einsum('me,ie->mi',F_me,t1,optimize=True)
  fprod1 = np.einsum('ij,ij',F_ae,F_ae,optimize=True)/Nkp
  fprod2 = np.einsum('ij,ij',F_mi,F_mi,optimize=True)/Nkp
  print(f"Fae: {fprod1}, Fmi: {fprod2}")
  # Here we are forming the tilde-W_abef intermediate as in the
  # paper, at the cost of doing a o2v4 contraction once. The
  # tilde-W_nmij is already as in the paper, as we already doubled
  # the IJAB contribution for the t2 equations.
  W_abef -= np.einsum('ma,mbef->abef',t1,IABC,optimize=True)
  X1 = np.transpose(IABC,axes=(1,0,2,3))
  W_abef += np.einsum('mb,amef->abef',t1,X1,optimize=True)
  del X1
  X1 = np.transpose(tau,axes=(2,3,0,1))
  X2 = np.transpose(IJAB,axes=(2,3,0,1))
  tot_mem, avlb_mem = mem_check()
  v4gb = np.size(W_abef)*8/(1024**3)
  if(W_abef.dtype == complex): v4gb *= 2
  if(avlb_mem < 2*v4gb):
    lena = W_abef.shape[0]
    for a in range(lena):
      W_abef[a,:,:,:] +=  0.5*np.einsum('bmn,efmn->bef',X1[a,:,:,:],X2,optimize=True)/Nkp
  else:
    W_abef += 0.5*np.einsum('abmn,efmn->abef',X1,X2,optimize=True)/Nkp
  del X1, X2
  W_mbej += 0.5*np.einsum('nmfe,jnbf->mbej',IJAB,t2,optimize=True)/Nkp
  wprod1 = np.einsum('abcd,abcd',W_abef,W_abef,optimize=True)/Nkp**3
  wprod2 = np.einsum('abcd,abcd',W_mbej,W_mbej,optimize=True)/Nkp**3
  print(f"Wabef: {wprod1}, Wmbej: {wprod2}")
  del W_mbej
  # These intermediates are new
  W_efam[:,:,:,:] = np.einsum('mnef,na->efam',t2,F_me,optimize=True)
  W_efam -= np.transpose(np.conjugate(IABC),axes=(2,3,1,0)) 
  W_efam += np.einsum('efag,mg->efam',W_abef,t1,optimize=True)
  del W_abef
  # This is the opposite of what's in Gauss' paper
  W_efam -= 0.5*np.einsum('noef,noma->efam',tau,IJKA,optimize=True)/Nkp
  W_iemn[:,:,:,:] = -np.einsum('mnef,if->iemn',t2,F_me,optimize=True)
  W_iemn += np.transpose(np.conjugate(IJKA),axes=(2,3,0,1)) 
  W_iemn -= np.einsum('iomn,oe->iemn',W_mnij,t1,optimize=True)
  del W_mnij
  W_iemn += 0.5*np.einsum('iefg,mnfg->iemn',IABC,tau,optimize=True)/Nkp
  # Create a temp intermediate
  WW_mbej = -np.einsum('mnef,njbf->mbej',IJAB,t2,optimize=True)/Nkp
  WW_mbej += IABJ
  wprod1 = np.einsum('abcd,abcd',W_efam,W_efam,optimize=True)/Nkp**3
  wprod2 = np.einsum('abcd,abcd',W_iemn,W_iemn,optimize=True)/Nkp**3
  wprod3 = np.einsum('abcd,abcd',WW_mbej,WW_mbej,optimize=True)/Nkp**3
  print(f"Wefam1: {wprod1}, Wiemn1: {wprod2}, WWmbej: {wprod3}")
  X1 = - np.einsum('ne,nfam->efam',t1,WW_mbej,optimize=True)
  X1 += np.einsum('nega,mnfg->efam',IABC,t2,optimize=True)/Nkp
  X2 = X1 - np.transpose(X1,axes=(1,0,2,3))
  W_efam += X2
  wprod1 = np.einsum('abcd,abcd',W_efam,W_efam,optimize=True)/Nkp**3
  print(f"Wefam: {wprod1}")
  del X1,X2,W_efam
  X1 = np.einsum('mf,iefn->iemn',t1,WW_mbej,optimize=True)
  X1 += np.einsum('iomf,noef->iemn',IJKA,t2,optimize=True)/Nkp
  X2 = X1 - np.transpose(X1,axes=(0,1,3,2))
  W_iemn += X2
  wprod2 = np.einsum('abcd,abcd',W_iemn,W_iemn,optimize=True)/Nkp**3
  print(f"Wiemn: {wprod2}")
  del X1, X2, WW_mbej, W_iemn
  del IABC, IJAB, IABJ, IJKA
  return F_ae, F_mi

#########################################################################
# Define constant intermediates for CCSD Lambda and response equations
# with explicit loops over k points
#########################################################################
def Const_Interm3k(mol_out,scratch,Nkp,Ktable,t1,t2,tau,F_ae,F_mi,F_me):
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{mol_out}-IABJ.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{mol_out}-Wmnij.npy",mmap_mode='r')
  W_abef = np.load(f"{scratch}/{mol_out}-Wabef.npy",mmap_mode='r+')
  W_efam = np.load(f"{scratch}/{mol_out}-Wefam.npy",mmap_mode='r+')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r+')
  W_iemn = np.load(f"{scratch}/{mol_out}-Wiemn.npy",mmap_mode='r+')
  # Remember that the contraction for Lambda is over the opposite
  # one or two indices (same for W_mnij)
  F_ae -= 0.5*np.einsum('Ama,Ame->Aae',t1,F_me,optimize=True)    
  # The sign of this terms is wrong in Gauss' paper
  F_mi += 0.5*np.einsum('Mme,Mie->Mmi',F_me,t1,optimize=True)
  fprod1 = np.einsum('ijl,ijl',F_ae,F_ae,optimize=True)/Nkp
  fprod2 = np.einsum('ijl,ijl',F_mi,F_mi,optimize=True)/Nkp
  print(f"Fae: {fprod1}, Fmi: {fprod2}")
  # Here we are forming the tilde-W_abef intermediate as in the
  # paper, at the cost of doing a o2v4 contraction once. The
  # tilde-W_nmij is already as in the paper, as we already doubled
  # the IJAB contribution for the t2 equations.
  W_abef -= np.einsum('Ama,ABEmbef->ABEabef',t1,IABC,optimize=True)
  #### MC I'm not sure about the transposition here
  X1 = np.transpose(IABC,axes=(1,0,2,4,3,5,6))
  W_abef += np.einsum('Bmb,ABEamef->ABEabef',t1,X1,optimize=True)
  del X1
  # W_abef += 0.5*np.einsum('mnab,mnef->abef',tau,IJAB,optimize=True)/Nkp
  idx_A = np.arange(Nkp).reshape(Nkp, 1)
  J = Ktable
  IDX_B = np.arange(Nkp).reshape(1, Nkp, 1)   # broadcasts along B
  IDX_M = np.arange(Nkp).reshape(Nkp, 1, 1)   # broadcasts along M
  for M in range(Nkp):
    # W_abef
    N = Ktable[:,:,M]
    tauN = tau[M][N,idx_A]
    IJABN = IJAB[M][N]
    W_abef += 0.5*np.einsum('ABmnab,ABEmnef->ABEabef',tauN,IJABN,
                            optimize=True)/Nkp
    del N,tauN,IJABN
    #
    # W_mbej
    N = M
    F = Ktable[J,N,IDX_B]
    t2NF = t2[:,N,:][J,IDX_B]          # (M,B,E,j,n,b,f)
    IJABNF = IJAB[N][IDX_M,F]          # (M,B,E,n,m,f,e)
    W_mbej += 0.5*np.einsum('MBEnmfe,MBEjnbf->MBEmbej',IJABNF,t2NF,
                            optimize=True)/Nkp
    del F, t2NF, IJABNF
  wprod1 = np.einsum('ijlabcd,ijlabcd',W_abef,W_abef,optimize=True)/Nkp**3
  wprod2 = np.einsum('ijlabcd,ijlabcd',W_mbej,W_mbej,optimize=True)/Nkp**3
  print(f"Wabef: {wprod1}, Wmbej: {wprod2}")
  del W_mbej
  #
  # These intermediates are new
  X1 = transpose_l(t2,Ktable,'i')
  W_efam[:,:,:,:,:,:,:] = np.einsum('FAEfnem,Ana->EFAefam',X1,F_me,
                                    optimize=True)
  del X1
  # W_efam -= np.transpose(np.conjugate(IABC),axes=(2,3,1,0)) 
  W_efam -= np.transpose(transpose_l(np.conjugate(IABC),Ktable,'i'),
                         axes=(2,0,1,5,3,4,6)) 
  X1 = t1[Ktable]
  W_efam += np.einsum('EFAefag,EFAmg->EFAefam',W_abef,X1,optimize=True)
  del W_abef, X1
  X1 = transpose_l(t2,Ktable,'j')
  W_iemn[:,:,:,:,:,:,:] = -np.einsum('MIEmfen,Iif->IEMiemn',X1,F_me,
                                     optimize=True)
  del X1
  W_iemn += np.transpose(transpose_l(np.conjugate(IJKA),Ktable,'j'),
                         axes=(2,1,0,5,4,3,6)) 
  W_iemn -= np.einsum('IEMiomn,Eoe->IEMiemn',W_mnij,t1,optimize=True)
  del W_mnij
  #
  # We need a separate loop over k points because W_efam depends on
  # W_abef, which is computed above with its own k loop. This is not a
  # big deal because these intermediates are only computed once.
  #
  # for W_efam
  M = Ktable
  IDX_E = np.arange(Nkp).reshape(Nkp, 1) # broadcasts along the E axis
  # for W_iemn
  KN = Ktable
  IDX_M = np.arange(Nkp).reshape(1, 1, Nkp) # broadcasts along the M axis
  # for WW_mbej
  # Create a temp intermediate
  WW_mbej = np.copy(IABJ)
  del IABJ
  KJ = Ktable
  IDX_B = np.arange(Nkp).reshape(1, Nkp, 1)    # broadcasts along the B axis
  for N in range(Nkp):
    # W_efam
    O = Ktable[:, :, N] # O_N[E,F] = Ktable[E,F,N] -- depends on N
    tauO = tau[N][O,IDX_E]          # (E,F,n,o,e,f)
    IJKAO = IJKA[N][O[:,:,None],M]  # (E,F,A,n,o,m,a)
    # This is the opposite of what's in Gauss' paper
    W_efam -= 0.5*np.einsum('EFnoef,EFAnoma->EFAefam',tauO,IJKAO,
                            optimize=True)/Nkp
    del O, tauO, IJKAO
    # W_iemn
    F = N
    IABCF = IABC[:, :, F]  # (I,E,i,e,f,g) -- plain slice, no gather
    tauF = tau[:, :, F][IDX_M, KN]              # (I,E,M,m,n,f,g)
    W_iemn += 0.5*np.einsum('IEiefg,IEMmnfg->IEMiemn',IABCF,tauF,
                            optimize=True)/Nkp
    del IABCF, tauF
    # WW_mbej
    IJABN = IJAB[:,N,:]   # (M,E,m,n,e,f) -- plain slice, no gather
    t2N = t2[N][KJ, IDX_B]  # (M,B,E,n,j,b,f)
    WW_mbej -= np.einsum('MEmnef,MBEnjbf->MBEmbej',IJABN,t2N,
                         optimize=True)/Nkp
    del IJABN, t2N
  del IJAB
  wprod1 = np.einsum('ijlabcd,ijlabcd',W_efam,W_efam,optimize=True)/Nkp**3
  wprod2 = np.einsum('ijlabcd,ijlabcd',W_iemn,W_iemn,optimize=True)/Nkp**3
  wprod3 = np.einsum('ijlabcd,ijlabcd',WW_mbej,WW_mbej,optimize=True)/Nkp**3
  print(f"Wefam1: {wprod1}, Wiemn1: {wprod2}, WWmbej: {wprod3}")
  #
  # W_efam last piece
  X1 = - np.einsum('Ene,EFAnfam->EFAefam',t1,WW_mbej,optimize=True)
  M = Ktable 
  IDX_F = np.arange(Nkp).reshape(1, Nkp, 1) # broadcasts along F
  IDX_E = np.arange(Nkp).reshape(Nkp, 1, 1) # broadcasts along E
  for N in range(Nkp):
    G = Ktable[M, N, IDX_F]    # (E,F,A) -- depends on N
    t2N = t2[:, N, :][M,IDX_F] # (E,F,A,m,n,f,g)
    IABCN = IABC[N][IDX_E,G]   # (E,F,A,n,e,g,a)
    X1 += np.einsum('EFAnega,EFAmnfg->EFAefam',IABCN,t2N,optimize=True)/Nkp
    del G, t2N, IABCN
  W_efam += X1 - np.transpose(X1,axes=(1,0,2,4,3,5,6))
  wprod1 = np.einsum('ijlabcd,ijlabcd',W_efam,W_efam,optimize=True)/Nkp**3
  print(f"Wefam: {wprod1}")
  del X1, W_efam, IABC
  #
  # W_iemn last piece
  X1 = np.einsum('Mmf,IEMiefn->IEMiemn',t1,WW_mbej,optimize=True)
  del WW_mbej
  N = Ktable
  IDX_E = np.arange(Nkp).reshape(1, Nkp, 1)   # broadcasts along E
  for O in range(Nkp):
    IJKAO = IJKA[:, O, :]
    t2O = t2[:, O, :][N, IDX_E]  # (I,E,M,n,o,e,f)
    X1 += np.einsum('IMiomf,IEMnoef->IEMiemn',IJKAO,t2O,optimize=True)/ Nkp
  W_iemn += X1 - transpose_l(X1,Ktable,'k')
  wprod2 = np.einsum('ijlabcd,ijlabcd',W_iemn,W_iemn,optimize=True)/Nkp**3
  print(f"Wiemn: {wprod2}")
  del X1, W_iemn, IJKA
  return F_ae, F_mi

#########################################################################
# Define changing intermediates for CCSD Lambda equations
#########################################################################
def L_Interm(Nkp, t2, l2):
  NkpS = Nkp*Nkp
  G_ae = -0.5*np.einsum('mnaf,mnef->ae',l2,t2,optimize=True)/NkpS
  G_mi = 0.5*np.einsum('mnef,inef->mi',t2,l2,optimize=True)/NkpS
  return G_ae, G_mi

#########################################################################
# Define changing intermediates for CCSD Lambda equations with
# explicit loops over k points
#########################################################################
def L_Interm3k(Nkp, t2, l2):
  NkpS = Nkp*Nkp
  G_ae = -0.5*np.einsum('MNAmnaf,MNAmnef->Aae',l2,t2,optimize=True)/NkpS
  G_mi = 0.5*np.einsum('MNEmnef,MNEinef->Mmi',t2,l2,optimize=True)/NkpS
  return G_ae, G_mi

#########################################################################
# CCSD Lambda1 amplitude equation
#########################################################################
def l1Eq(mol_out,scratch,Nkp,t1,l1,l2,F_ae,F_mi,F_me,G_ae,G_mi,D1):
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_efam = np.load(f"{scratch}/{mol_out}-Wefam.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  W_iemn = np.load(f"{scratch}/{mol_out}-Wiemn.npy",mmap_mode='r')
  NkpS = Nkp*Nkp
  l1_f = np.copy(F_me)  
  l1_f += np.einsum('ie,ea->ia',l1,F_ae,optimize=True)
  l1_f -= np.einsum('im,ma->ia',F_mi,l1,optimize=True)
  l1_f += np.einsum('me,ieam->ia',l1,W_mbej,optimize=True)/Nkp
  l1_f += 0.5*np.einsum('imef,efam->ia',l2,W_efam,optimize=True)/NkpS
  del W_efam, W_mbej
  l1_f -= 0.5*np.einsum('iemn,mnae->ia',W_iemn,l2,optimize=True)/NkpS
  del W_iemn
  l1_f += np.einsum('ef,iefa->ia',G_ae,IABC,optimize=True)/Nkp
  l1_f += np.einsum('mn,imna->ia',G_mi,IJKA,optimize=True)/Nkp
  X1 = np.einsum('mf,fe->me',t1,G_ae,optimize=True)
  X1 -= np.einsum('mn,ne->me',G_mi,t1,optimize=True)
  l1_f += np.einsum('me,imae->ia',X1,IJAB,optimize=True)/Nkp
  del X1
  l1_f /= D1
  del IABC, IJAB, IJKA
  return l1_f

#########################################################################
# CCSD Lambda1 amplitude equation with explicit loops over k points
#########################################################################
def l1Eq3k(mol_out,scratch,Nkp,Ktable,t1,l1,l2,F_ae,F_mi,F_me,G_ae,G_mi,D1):
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_efam = np.load(f"{scratch}/{mol_out}-Wefam.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  W_iemn = np.load(f"{scratch}/{mol_out}-Wiemn.npy",mmap_mode='r')
  NkpS = Nkp*Nkp
  l1_f = np.copy(F_me)  
  l1_f += np.einsum('Iie,Iea->Iia',l1,F_ae,optimize=True)
  l1_f -= np.einsum('Iim,Ima->Iia',F_mi,l1,optimize=True)
  l1_f += np.einsum('Mme,IMIieam->Iia',l1,W_mbej,optimize=True)/Nkp
  X1 = transpose_l(l2,Ktable,'j')
  l1_f += 0.5*np.einsum('IFEifem,EFIefam->Iia',X1,W_efam,optimize=True)/NkpS
  del W_efam, W_mbej
  l1_f -= 0.5*np.einsum('IEMiemn,MEImean->Iia',W_iemn,X1,optimize=True)/NkpS
  del W_iemn, X1
  l1_f += np.einsum('Eef,IEEiefa->Iia',G_ae,IABC,optimize=True)/Nkp
  l1_f += np.einsum('Mmn,IMMimna->Iia',G_mi,IJKA,optimize=True)/Nkp
  X1 = np.einsum('Mmf,Mfe->Mme',t1,G_ae,optimize=True)
  X1 -= np.einsum('Mmn,Mne->Mme',G_mi,t1,optimize=True)
  l1_f += np.einsum('Mme,IMIimae->Iia',X1,IJAB,optimize=True)/Nkp
  del X1
  l1_f /= D1
  del IABC, IJAB, IJKA
  return l1_f

#########################################################################
# CCSD Lambda2 amplitude equation
#########################################################################
def l2Eq(mol_out,scratch,Nkp,t1,l1,l2,F_ae,F_mi,F_me,G_ae,G_mi,D2):
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{mol_out}-Wmnij.npy",mmap_mode='r')
  W_abef = np.load(f"{scratch}/{mol_out}-Wabef.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  l2_f = np.copy(IJAB)
  l2_f += 0.5*np.einsum('ijef,efab->ijab',l2,W_abef,optimize=True)/Nkp
  l2_f += 0.5*np.einsum('ijmn,mnab->ijab',W_mnij,l2,optimize=True)/Nkp
  del W_abef, W_mnij
  # P(ab) terms
  X1 = G_ae - np.einsum('mb,me->be',l1,t1,optimize=True)
  X2 = np.einsum('ijae,be->ijab',IJAB,X1,optimize=True)
  X2 -= np.einsum('ma,ijmb->ijab',l1,IJKA,optimize=True)
  X2 += np.einsum('ijae,eb->ijab',l2,F_ae,optimize=True) 
  l2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
  del X1, X2, IJKA
  # P(ij) terms
  X1 = G_mi + np.einsum('me,je->mj',t1,l1,optimize=True)
  X2 = np.einsum('imab,mj->ijab',IJAB,X1,optimize=True)
  X2 += np.einsum('ie,jeab->ijab',l1,IABC,optimize=True)
  X2 += np.einsum('imab,jm->ijab',l2,F_mi,optimize=True) 
  l2_f += np.transpose(X2,axes=(1,0,2,3)) - X2 
  del X1, X2, IABC, IJAB
  # P(ij,ab) terms
  X2 = np.einsum('imae,jebm->ijab',l2,W_mbej,optimize=True)/Nkp
  X2 += np.einsum('ia,jb->ijab',l1,F_me,optimize=True)*Nkp
  l2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
  l2_f -= np.transpose(X2,axes=(0,1,3,2))
  l2_f += np.transpose(X2,axes=(1,0,3,2))
  del X2, W_mbej
  l2_f /= D2
  return l2_f

#########################################################################
# CCSD Lambda2 amplitude equation with explicit loops over k points
#########################################################################
def l2Eq3k(mol_out,scratch,Nkp,Ktable,t1,l1,l2,F_ae,F_mi,F_me,G_ae,G_mi,D2):
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{mol_out}-Wmnij.npy",mmap_mode='r')
  W_abef = np.load(f"{scratch}/{mol_out}-Wabef.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  l2_f = np.copy(IJAB)
  ind_A = np.arange(Nkp).reshape(1,1,Nkp)
  X2 = np.zeros(l2_f.shape,dtype=l2_f.dtype)
  B = Ktable
  ind_J = np.arange(Nkp).reshape(1,Nkp,1)
  for E in range(Nkp):
    # W_abef
    F = Ktable[:,:,E]
    WF = W_abef[E][F[:,:,None],ind_A]
    l2F = l2[:,:,E]
    l2_f += 0.5*np.einsum('IJijef,IJAefab->IJAijab',l2F,WF,
                          optimize=True)/Nkp
    del l2F, WF
    # W_mnij
    M = E
    N = F
    l2M = l2[M][N[:,:,None],ind_A]
    WM = W_mnij[:,:,M]
    l2_f += 0.5*np.einsum('IJijmn,IJAmnab->IJAijab',WM,l2M,
                          optimize=True)/Nkp
    del l2M, WM, F
    # W_mbej
    KE = Ktable[:,M,:][:,None,:]
    l2M = l2[:,M,:]
    WM = W_mbej[ind_J,KE,B] 
    X2 += np.einsum('IAimae,IJAjebm->IJAijab',l2M,WM,optimize=True)/Nkp
    X2[M,:,M] += np.einsum('ia,Jjb->Jijab',l1[M],F_me,optimize=True)*Nkp
  del W_abef, W_mnij, ind_A
  # P(ij,ab) terms
  l2_f += X2 - np.transpose(X2,axes=(1,0,2,4,3,5,6))
  l2_f -= transpose_l(X2,Ktable,'k')
  l2_f += np.transpose(transpose_l(X2,Ktable,'k'),axes=(1,0,2,4,3,5,6))
  del X2, W_mbej
  # P(ab) terms
  X1 = G_ae - np.einsum('Bmb,Bme->Bbe',l1,t1,optimize=True)
  X1k = X1[Ktable]
  X2 = np.einsum('IJAijae,IJAbe->IJAijab',IJAB,X1k,optimize=True)
  del X1, X1k
  X2 -= np.einsum('Ama,IJAijmb->IJAijab',l1,IJKA,optimize=True)
  X1 = F_ae[Ktable]
  X2 += np.einsum('IJAijae,IJAeb->IJAijab',l2,X1,optimize=True) 
  l2_f += X2 - transpose_l(X2,Ktable,'k')
  del X1, X2, IJKA
  # P(ij) terms
  X1 = G_mi + np.einsum('Mme,Mje->Mmj',t1,l1,optimize=True)
  X2 = np.einsum('IJAimab,Jmj->IJAijab',IJAB,X1,optimize=True)
  X2 += np.einsum('Iie,JIAjeab->IJAijab',l1,IABC,optimize=True)
  X2 += np.einsum('IJAimab,Jjm->IJAijab',l2,F_mi,optimize=True) 
  l2_f += np.transpose(X2,axes=(1,0,2,4,3,5,6)) - X2 
  del X1, X2, IABC, IJAB
  l2_f /= D2
  return l2_f

#########################################################################
# Form constant terms based on 1e perturbation X:
# <S|e^{-T}Xe^{T}|0> and <D|e^{-T}Xe^{T}|0>.
#########################################################################
def pert_rhs(PertSymm,Nkp,O2,V2,t1,t2,X_ij,X_ia,X_ab):
  # X is supposed to be in MO basis and already divided in oo, ov, and vv blocks
  #
  # Singles
  if(PertSymm == "Symm"):
    rhs1 = np.copy(X_ia)
  else:
    rhs1 = -np.copy(X_ia)
  rhs1 += np.einsum('kc,ikac->ia',np.conjugate(X_ia),t2,optimize=True)/Nkp
  rhs1 -= np.einsum('kc,ic,ka->ia',np.conjugate(X_ia),t1,t1,optimize=True)
  rhs1 += np.einsum('ic,ca->ia',t1,X_ab,optimize=True)
  rhs1 -= np.einsum('ik,ka->ia',X_ij,t1,optimize=True)
  # Doubles
  # P(ij) terms: -P(ij) t(kjab)(X(ik)+X(kc)t(ic))
  X1 = np.copy(X_ij) + np.einsum('ic,kc->ik',t1,np.conjugate(X_ia),optimize=True)
  X2 = -np.einsum('ik,kjab->ijab',X1,t2,optimize=True)
  rhs2 = X2 - np.transpose(X2,axes=(1,0,2,3))
  # P(ab) terms: P(ab) t(ijac)(X(cb)-X(kc)t(kb))
  X1 = np.copy(X_ab) - np.einsum('kc,kb->cb',np.conjugate(X_ia),t1,optimize=True)
  X2 = np.einsum('ijac,cb->ijab',t2,X1,optimize=True)
  rhs2 += X2 - np.transpose(X2,axes=(0,1,3,2))
  del X1, X2
  return rhs1, rhs2

#########################################################################
# Form constant terms based on 1e perturbation X:
# <S|e^{-T}Xe^{T}|0> and <D|e^{-T}Xe^{T}|0>
# with explicit loops over k points.
#########################################################################
def pert_rhs3k(PertSymm,Nkp,O2,V2,t1,t2,X_ij,X_ia,X_ab,Ktable):
  # X is supposed to be in MO basis and already divided in oo, ov, and
  # vv blocks
  #
  # Singles
  if(PertSymm == "Symm"):
    rhs1 = np.copy(X_ia)
  else:
    rhs1 = -np.copy(X_ia)
  rhs1 += np.einsum('Kkc,IKIikac->Iia',np.conjugate(X_ia),t2,optimize=True)/Nkp
  rhs1 -= np.einsum('Ikc,Iic,Ika->Iia',np.conjugate(X_ia),t1,t1,optimize=True)
  rhs1 += np.einsum('Iic,Ica->Iia',t1,X_ab,optimize=True)
  rhs1 -= np.einsum('Iik,Ika->Iia',X_ij,t1,optimize=True)
  # Doubles
  # P(ij) terms: -P(ij) t(kjab)(X(ik)+X(kc)t(ic))
  X1 = np.copy(X_ij) + np.einsum('Iic,Ikc->Iik',t1,np.conjugate(X_ia),optimize=True)
  X2 = -np.einsum('Iik,IJAkjab->IJAijab',X1,t2,optimize=True)
  rhs2 = X2 - np.transpose(X2,axes=(1,0,2,4,3,5,6))
  # P(ab) terms: P(ab) t(ijac)(X(cb)-X(kc)t(kb))
  X1 = np.copy(X_ab) - np.einsum('Akc,Akb->Acb',np.conjugate(X_ia),t1,optimize=True)
  X13k = X1[Ktable]
  X2 = np.einsum('IJAijac,IJAcb->IJAijab',t2,X13k,optimize=True)
  rhs2 += X2 - transpose_l(X2,Ktable,'k')
  del X1, X13k, X2
  return rhs1, rhs2

#########################################################################
# CCSD Tx1 (or EOM R1) amplitude equation
#########################################################################
def tx1Eq(mol_out,scratch,Nkp,O,V,tx1,tx2,t1,F_ae,F_mi,F_me,G_ae,G_mi,D1):
  # Constant term needs to be added outside (as it's not in the EOM eqs.)
  # It requires getting G_ae, G_mi = L_Interm(Nkp, IJAB, tx2)
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  tot_mem, avlb_mem = mem_check()
  O2 = 2*O
  V2 = 2*V
  o2v2gb = (O2**2)*(V2**2)*8/(1024**3)
  ov3gb = O2*(V2**3)*8/(1024**3)
  if(tx1.dtype == complex):
    o2v2gb *= 2    
    ov3gb *= 2     
  NkpS = Nkp*Nkp
  tx1_f = np.einsum('ie,ae->ia',tx1,F_ae,optimize=True)
  tx1_f -= np.einsum('mi,ma->ia',F_mi,tx1,optimize=True)
  tx1_f += np.einsum('me,maei->ia',tx1,W_mbej,optimize=True)/Nkp
  tx1_f += np.einsum('imae,me->ia', tx2, F_me, optimize=True)/Nkp
  if(avlb_mem < ov3gb+o2v2gb):
    for a in range(V2):
      tx1_f[:,a] -= 0.5 * np.einsum('imef,mef->i', tx2, IABC[:,a,:,:],
                                    optimize=True)/NkpS
  else:
    tx1_f -= 0.5 * np.einsum('imef,maef->ia', tx2, IABC,optimize=True)/NkpS
  tx1_f += 0.5 * np.einsum('nmea,nmie->ia', tx2, IJKA,optimize=True)/NkpS
  tx1_f += np.einsum('ib,ab->ia',t1,G_ae,optimize=True)
  tx1_f -= np.einsum('ji,ja->ia',G_mi,t1,optimize=True)
  tx1_f /= D1
  del IABC, IJKA, W_mbej
  return tx1_f

#########################################################################
# CCSD Tx1 (or EOM R1) amplitude equation with explicit loops over k points
#########################################################################
def tx1Eq3k(mol_out,scratch,Nkp,Ktable,O,V,tx1,tx2,t1,F_ae,F_mi,F_me,G_ae,
            G_mi,D1):
  # Constant term needs to be added outside (as it's not in the EOM eqs.)
  # It requires getting G_ae, G_mi = L_Interm(Nkp, IJAB, tx2)
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  tot_mem, avlb_mem = mem_check()
  O2 = 2*O
  V2 = 2*V
  NkpS = Nkp*Nkp
  tx1_f = np.einsum('Iie,Iae->Iia',tx1,F_ae,optimize=True)
  tx1_f -= np.einsum('Imi,Ima->Iia',F_mi,tx1,optimize=True)
  tx1_f += np.einsum('Mme,MIMmaei->Iia',tx1,W_mbej,optimize=True)/Nkp
  tx1_f += np.einsum('IMIimae,Mme->Iia', tx2, F_me, optimize=True)/Nkp
  tx1_f -= 0.5*np.einsum('IMEimef,MIEmaef->Iia',tx2,IABC,optimize=True)/NkpS
  X1 = transpose_l(tx2,Ktable,'k')
  tx1_f += 0.5*np.einsum('NMInmae,NMInmie->Iia',X1,IJKA,optimize=True)/NkpS
  del X1
  tx1_f += np.einsum('Iib,Iab->Iia',t1,G_ae,optimize=True)
  tx1_f -= np.einsum('Iji,Ija->Iia',G_mi,t1,optimize=True)
  tx1_f /= D1
  del IABC, IJKA, W_mbej
  return tx1_f

#########################################################################
# CCSD Tx2 (or EOM R2) amplitude equation
#########################################################################
def tx2Eq(mol_out,scratch,Nkp,O,V,tx1,tx2,t1,t2,F_ae,F_mi,F_me,G_ae,G_mi,D2):
  # Constant term needs to be added outside (as it's not in the EOM eqs.)
  # It requires getting G_ae, G_mi = L_Interm(Nkp, IJAB, tx2)
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{mol_out}-Wmnij.npy",mmap_mode='r')
  W_abef = np.load(f"{scratch}/{mol_out}-Wabef.npy",mmap_mode='r')
  W_efam = np.load(f"{scratch}/{mol_out}-Wefam.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  W_iemn = np.load(f"{scratch}/{mol_out}-Wiemn.npy",mmap_mode='r')
  tot_mem, avlb_mem = mem_check()
  O2 = 2*O
  V2 = 2*V
  v4gb = (V2**4)*8/(1024**3)
  o2v2gb = (O2**2)*(V2**2)*8/(1024**3)
  if(tx2.dtype == complex):
    v4gb *= 2
    o2v2gb *= 2
  NkpS = Nkp*Nkp
  if(avlb_mem < v4gb+2*o2v2gb):
    tx2_f = np.zeros((O2,O2,V2,V2),dtype=tx2.dtype)
    for a in range(V2):
      tx2_f[:,:,a,:] += 0.5*np.einsum('ijef,bef->ijb',tx2,W_abef[a,:,:,:],
                                      optimize=True)/Nkp
  else:
    tx2_f = 0.5*np.einsum('ijef,abef->ijab',tx2,W_abef,optimize=True)/Nkp
  tx2_f += 0.5*np.einsum('mnij,mnab->ijab',W_mnij,tx2,optimize=True)/Nkp
  del W_abef, W_mnij
  # P(ij) terms
  X0 = np.einsum('kc,kmcd->md',tx1,IJAB,optimize=True)/Nkp
  del IJAB
  X1 = G_mi + np.einsum('md,jd->mj',X0,t1,optimize=True)
  X1 -= np.einsum('kc,kmjc->mj',tx1,IJKA,optimize=True)/Nkp
  X2 = -np.einsum('imab,mj->ijab',t2,X1,optimize=True)
  X2 += np.einsum('ic,abcj->ijab',tx1,W_efam,optimize=True)
  X2 -= np.einsum('imab,mj->ijab',tx2,F_mi,optimize=True) # original
  tx2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
  del X1, X2, IJKA, W_efam
  # P(ab) terms
  X1 = G_ae - np.einsum('mb,md->bd',t1,X0,optimize=True)
  X1 += np.einsum('kc,kbcd->bd',tx1,IABC,optimize=True)/Nkp
  X2 = np.einsum('ijae,be->ijab',t2,X1,optimize=True)
  X2 -= np.einsum('ka,kbij->ijab',tx1,W_iemn,optimize=True)
  X2 += np.einsum('ijae,be->ijab',tx2,F_ae,optimize=True)
  tx2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
  del X0,X1,X2,IABC,W_iemn
  # P(ij,ab) terms
  X2 = np.einsum('imae,mbej->ijab',tx2,W_mbej,optimize=True)/Nkp
  tx2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
  tx2_f -= np.transpose(X2,axes=(0,1,3,2))
  tx2_f += np.transpose(X2,axes=(1,0,3,2))
  del X2, W_mbej
  # Divide by energy denominator
  tx2_f /= D2
  return tx2_f

#########################################################################
# CCSD Tx2 (or EOM R2) amplitude equation with explicit loops over k points
#########################################################################
def tx2Eq3k(mol_out,scratch,Nkp,Ktable,O,V,tx1,tx2,t1,t2,F_ae,F_mi,F_me,
            G_ae,G_mi,D2):
  # Constant term needs to be added outside (as it's not in the EOM eqs.)
  # It requires getting G_ae, G_mi = L_Interm(Nkp, IJAB, tx2)
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{mol_out}-Wmnij.npy",mmap_mode='r')
  W_abef = np.load(f"{scratch}/{mol_out}-Wabef.npy",mmap_mode='r')
  W_efam = np.load(f"{scratch}/{mol_out}-Wefam.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  W_iemn = np.load(f"{scratch}/{mol_out}-Wiemn.npy",mmap_mode='r')
  tot_mem, avlb_mem = mem_check()
  O2 = 2*O
  V2 = 2*V
  NkpS = Nkp*Nkp
  tx2_f = np.zeros(t2.shape,dtype=t2.dtype)
  #
  # Unavoidable loop over k point
  ind_I = np.arange(Nkp).reshape(Nkp,1)
  ind_A = np.arange(Nkp).reshape(1,Nkp)
  X3 = np.zeros(tx2_f.shape, dtype=tx2_f.dtype)
  KB = Ktable
  for M in range(Nkp):
    # W_abef
    I = M
    B = Ktable[I]
    tx2I = tx2[I]
    WI = W_abef[ind_A,B]
    tx2_f[I] += 0.5*np.einsum('JEijef,JAEabef->JAijab',tx2I,WI,
                              optimize=True)/Nkp
    del B,tx2I,WI
    # W_mnij
    N = Ktable[:,:,M]
    tx2N = tx2[M][N]
    WN = W_mnij[M][N,ind_I]
    tx2_f += 0.5*np.einsum('IJmnij,IJAmnab->IJAijab',WN,tx2N,
                           optimize=True)/Nkp
    del N,WN,tx2N
    #P(ij,ab)
    E = Ktable[:, M, :]
    ind_E = E[:,None,:]
    tx2E = tx2[:,M,:]
    WE = W_mbej[M][KB,ind_E]
    X3 += np.einsum('IAimae,IJAmbej->IJAijab',tx2E,WE,optimize=True)/Nkp
    del tx2E, WE, E
  del W_abef, W_mnij, W_mbej, KB
  # P(ij,ab) terms
  tx2_f += X3 - np.transpose(X3,axes=(1,0,2,4,3,5,6))
  tx2_f -= transpose_l(X3,Ktable,'k')
  tx2_f += np.transpose(transpose_l(X3,Ktable,'k'),axes=(1,0,2,4,3,5,6))
  del X3
  # P(ij) terms
  X0 = np.einsum('Kkc,KMKkmcd->Mmd',tx1,IJAB,optimize=True)/Nkp
  del IJAB
  X1 = G_mi + np.einsum('Mmd,Mjd->Mmj',X0,t1,optimize=True)
  X1 -= np.einsum('Kkc,KMMkmjc->Mmj',tx1,IJKA,optimize=True)/Nkp
  X2 = -np.einsum('IJAimab,Jmj->IJAijab',t2,X1,optimize=True)
  del X1, IJKA
  X1 = transpose_l(W_efam,Ktable,'j')
  X2 += np.einsum('Iic,AJIajcb->IJAijab',tx1,X1,optimize=True)
  del X1, W_efam
  X2 -= np.einsum('IJAimab,Jmj->IJAijab',tx2,F_mi,optimize=True) # original
  tx2_f += X2 - np.transpose(X2,axes=(1,0,2,4,3,5,6))
  del X2
  # P(ab) terms
  X1 = G_ae - np.einsum('Bmb,Bmd->Bbd',t1,X0,optimize=True)
  X1 += np.einsum('Kkc,KBKkbcd->Bbd',tx1,IABC,optimize=True)/Nkp
  X13k = X1[Ktable]
  X2 = np.einsum('IJAijae,IJAbe->IJAijab',t2,X13k,optimize=True)
  del X13k, X1, IABC
  X1 = transpose_l(W_iemn,Ktable,'j')  
  X2 -= np.einsum('Aka,AJIkjib->IJAijab',tx1,X1,optimize=True)
  del X1, W_iemn
  F3k = F_ae[Ktable]
  X2 += np.einsum('IJAijae,IJAbe->IJAijab',tx2,F3k,optimize=True)
  del F3k
  tx2_f += X2 - transpose_l(X2,Ktable,'k')
  del X0,X2
  # Divide by energy denominator
  tx2_f /= D2
  return tx2_f

#########################################################################
# CCSD Xi amplitudes for LR and EOM gradients
#########################################################################
def Xi(T,mol_out,scratch,Nkp,O2,tx1,tx2,l1,l2,t1,F_ae,F_mi,F_me,D2):
  # L can be the ground or excited state Lambda amplitudes
  # Tx can be the LR Tx or the EOM R amplitudes
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  tot_mem, avlb_mem = mem_check()
  o4gb = (O2**4)*8/(1024**3)
  if(tx1.dtype == complex): o4gb *= 2
  if(avlb_mem < 2*o4gb):
    Yimjk = np.lib.format.open_memmap(f"{scratch}/{mol_out}-Yimjk.npy",
                                      mode='w+',shape=(O2,O2,O2,O2),
                                      dtype=tx1.dtype) 
    Zkijm = np.lib.format.open_memmap(f"{scratch}/{mol_out}-Zkijm.npy",
                                      mode='w+',shape=(O2,O2,O2,O2),
                                      dtype=tx1.dtype)
  else:
    Yimjk = np.zeros((O2,O2,O2,O2),dtype=tx1.dtype)
    Zkijm = np.zeros((O2,O2,O2,O2),dtype=tx1.dtype)
#
  if T==1:
    NkpS = Nkp*Nkp
    # Term 1
    # Xi1 : R1*Lg*H
    #       -R(jb)*(Lg(ja)*X(ib) + Lg(ib)*X(ja))
    #             _                  _                 _
    #       <0|L[[H,O1],R]|0> = <0|L(HR)conn|1> - <0|L(HR)disc|1>
    # P(ab) terms
    X2 = np.einsum('ijae,eb->ijab',l2,F_ae,optimize=True) 
    X1 = X2 - np.transpose(X2,axes=(0,1,3,2))
    del X2
    # P(ij) terms
    X2 = np.einsum('imab,jm->ijab',l2,F_mi,optimize=True) 
    X1 += np.transpose(X2,axes=(1,0,2,3)) - X2 
    del X2
    # P(ij,ab)-like terms
    X2 = np.einsum('imae,jebm->ijab',l2,W_mbej,optimize=True)/Nkp
    X2 += np.einsum('ia,jb->ijab',l1,F_me,optimize=True)*Nkp
    X1 += X2 + np.transpose(X2,axes=(1,0,3,2))
    del X2, W_mbej
    X1 -= l2*D2
    Xi1 = -np.einsum('ijab,jb->ia',X1,tx1,optimize=True)/Nkp
    del X1
    #
    fmiprod  = np.einsum('ia,ia->',F_mi,np.conjugate(F_mi),optimize=True)
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    Xi2prod = 0
    print(f"Term 1: Fmi= {fmiprod/Nkp}")
    print(f"Term 1: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # term 2
    # Xi1 : -Lg(ijdb)t(md)R(kjcb)<mk||ac>
    X1 = np.einsum('kjcb,mkac->jmab',tx2,IJAB,optimize=True)/Nkp
    X2 = np.einsum('ijdb,md->ijmb',l2,t1,optimize=True)
    Xi1 -= np.einsum('ijmb,jmab->ia',X2,X1,optimize=True)/Nkp
    del X1, X2
    #
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    print(f"Term 2: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # Term 3
    # Xi1 : Lg(jibd)R(jkbc)<kd||ca>
    #     : Lg(jmba)R(jkbc)[<ki||mc>+t(md)<ki||dc>]
    # Xi2 : P(ij,ab)Lg(kica)[R(kmcd)-R(mc)t(kd)-t(mc)R(kd)]<mj||db>
    Yikdc = np.einsum('jibd,jkbc->ikdc',l2,tx2,optimize=True)/Nkp
    Xi1 += np.einsum('ikdc,kdca->ia',Yikdc,IABC,optimize=True)/NkpS
    X2 = IJKA + np.einsum('md,kidc->kimc',t1,IJAB,optimize=True)
    Xi1 += np.einsum('mkac,kimc->ia',Yikdc,X2,optimize=True)/NkpS
    del X2
    X2 = np.einsum('kica,mc->kima',l2,tx1,optimize=True)
    Yikdc -= np.einsum('kima,kd->imad',X2,t1,optimize=True)
    X2 = np.einsum('kica,mc->kima',l2,t1,optimize=True)
    Yikdc -= np.einsum('kima,kd->imad',X2,tx1,optimize=True)
    del X2
    X2 = np.einsum('imad,mjdb->ijab',Yikdc,IJAB,optimize=True)/Nkp
    Xi2 = X2 - np.transpose(X2,axes=(0,1,3,2))
    Xi2 -= np.transpose(X2,axes=(1,0,2,3))
    Xi2 += np.transpose(X2,axes=(1,0,3,2))
    del Yikdc, X2
    #
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    Xi2prod = np.einsum('ijab,ijab->',Xi2,np.conjugate(Xi2),optimize=True)
    print(f"Term 3: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # Term 4
    # Xi1 :  1/2R(jkbc)Lg(jkbd)<di||ca>
    #     :  1/2R(jmbd)Lg(jmba)t(kc)<ik||cd>
    #     : -1/2R(jmbd)Lg(jmbc)t(kc)<ik||ad>
    # Xi2 : -1/2P(ab)R(kmcd)Lg(kmbd)<ij||ac>
    Ycd = 0.5*np.einsum('jkbc,jkbd->cd',tx2,l2,optimize=True)/NkpS
    Xi1 -= np.einsum('cd,idca->ia',Ycd,IABC,optimize=True)/Nkp
    X2 = np.einsum('kc,ikcd->id',t1,IJAB,optimize=True)/Nkp
    Xi1 += np.einsum('da,id->ia',Ycd,X2,optimize=True)
    del X2
    X2 = np.einsum('dc,kc->kd',Ycd,t1,optimize=True)
    Xi1 -= np.einsum('kd,ikad->ia',X2,IJAB,optimize=True)/Nkp
    del X2
    X2 = -np.einsum('cb,ijac->ijab',Ycd,IJAB,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    del Ycd, X2
    #
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    Xi2prod = np.einsum('ijab,ijab->',Xi2,np.conjugate(Xi2),optimize=True)
    print(f"Term 4: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # Term 5
    # Xi1 : -1/2Lg(id)R(kmcd)<km||ca>
    # Xi2 : -1/2P(ab)Lg(ijad)R(kmcd)<km||cb>
    #     :    -P(ab)Lg(ijad)R(md)t(kc)<km||cb>
    #     :    -P(ab)Lg(ijad)t(md)R(kc)<km||cb>
    #     :     P(ab)Lg(ijad)R(kc)<kd||cb>
    Zda = -0.5*np.einsum('kmcd,kmca->da',tx2,IJAB,optimize=True)/NkpS
    Xi1 += np.einsum('id,da->ia',l1,Zda,optimize=True)
    X2 = np.einsum('kc,kmcb->mb',t1,IJAB,optimize=True)/Nkp
    Zda -= np.einsum('md,mb->db',tx1,X2,optimize=True)
    del X2
    X2 = np.einsum('kc,kmcb->mb',tx1,IJAB,optimize=True)/Nkp
    Zda -= np.einsum('md,mb->db',t1,X2,optimize=True) 
    del X2
    Zda += np.einsum('kc,kdcb->db',tx1,IABC,optimize=True)/Nkp 
    X2 = np.einsum('ijad,db->ijab',l2,Zda,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    del Zda, X2
    #
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    Xi2prod = np.einsum('ijab,ijab->',Xi2,np.conjugate(Xi2),optimize=True)
    print(f"Term 5: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # Term 6
    # Xi1 :  1/2Lg(jkbc)R(jmbc)<im||ka>
    #     : -1/2Lg(jibc)R(jmbc)t(kd)<mk||ad>
    #     : -1/2Lg(jkbc)R(jmbc)t(kd)<im||ad>
    #     :     Lg(jb)R(jmbd)<im||ad>
    # Xi2 : -1/2P(ij)Lg(jmcd)R(kmcd)<ik||ab>
    Ykm = 0.5*np.einsum('jkbc,jmbc->km',l2,tx2,optimize=True)/NkpS
    Xi1 += np.einsum('km,imka->ia',Ykm,IJKA,optimize=True)/Nkp
    X2 = np.einsum('ik,jkab->ijab',Ykm,IJAB,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(1,0,2,3))
    del X2
    X2 = np.einsum('kd,mkad->ma',t1,IJAB,optimize=True)/Nkp
    Xi1 -= np.einsum('im,ma->ia',Ykm,X2,optimize=True)
    del X2
    X2 = -np.einsum('km,kd->md',Ykm,t1,optimize=True)
    X2 += np.einsum('jb,jmbd->md',l1,tx2,optimize=True)/Nkp
    Xi1 += np.einsum('md,imad->ia',X2,IJAB,optimize=True)/Nkp
    del Ykm, X2
    #
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    Xi2prod = np.einsum('ijab,ijab->',Xi2,np.conjugate(Xi2),optimize=True)
    print(f"Term 6: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # Term 7
    # Xi1 : -1/2Lg(ka)R(kmcd)<im||cd>
    # Xi2 : -1/2P(ij)Lg(kjab)R(kmcd)<im||cd>
    #     :    -P(ij)Lg(kjab)R(kc)t(md)<im||cd>
    #     :    -P(ij)Lg(kjab)t(kc)R(md)<im||cd>
    #     :    -P(ij)Lg(kjab)R(mc)<im||kc>
    Zik = -0.5*np.einsum('imcd,kmcd->ik',IJAB,tx2,optimize=True)/NkpS
    Xi1 += np.einsum('ik,ka->ia',Zik,l1,optimize=True)
    X2 = np.einsum('md,imcd->ic',t1,IJAB,optimize=True)/Nkp
    Zik -= np.einsum('ic,kc->ik',X2,tx1,optimize=True)
    del X2
    X2 = np.einsum('md,imcd->ic',tx1,IJAB,optimize=True)/Nkp
    Zik -= np.einsum('ic,kc->ik',X2,t1,optimize=True)
    del X2
    Zik += np.einsum('mikc,mc->ik',IJKA,tx1,optimize=True)/Nkp
    # Another change to the IJKA order of contraction
    # Zik -= np.einsum('imkc,mc->ik',IJKA,tx1,optimize=True)/Nkp
    X2 = np.einsum('ik,kjab->ijab',Zik,l2,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(1,0,2,3))
    del Zik, X2
    #
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    Xi2prod = np.einsum('ijab,ijab->',Xi2,np.conjugate(Xi2),optimize=True)
    print(f"Term 7: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # Term 8
    # Xi1 : -1/4Lg(jkac)R(jkbd)<ic||bd>
    #     :  1/4Lg(jkac)t(mc)R(jkbd)<im||bd>
    # Xi2 :  1/4Lg(kmab)R(kmcd)<ij||cd>
    #     :     Lg(kmab)R(kc)t(md)<ij||cd>
    #     :     Lg(kmab)R(kc)<ij||cm>
    if(avlb_mem < 2*o4gb):
      for i in range(O2):
        Yimjk[i,:,:,:] = 0.25*np.einsum('jcd,kmcd->jkm',IJAB[i,:,:,:],
                                      tx2,optimize=True)/Nkp
        X2 = np.einsum('mjk,mc->cjk',Yimjk[i,:,:,:],t1,optimize=True)
        X2 -= 0.25*np.einsum('cbd,jkbd->cjk',IABC[i,:,:,:],tx2,optimize=True)/Nkp
        Xi1[i,:] += np.einsum('cjk,jkac->a',X2,l2,optimize=True)/NkpS
    else:
      Yimjk += 0.25*np.einsum('ijcd,kmcd->ijkm',IJAB,tx2,optimize=True)/Nkp
      X2 = np.einsum('imjk,mc->icjk',Yimjk,t1,optimize=True)
      X2 -= 0.25*np.einsum('icbd,jkbd->icjk',IABC,tx2,optimize=True)/Nkp
      Xi1 += np.einsum('icjk,jkac->ia',X2,l2,optimize=True)/NkpS
    del X2
    X2 = IJKA + np.einsum('ijdc,md->ijmc',IJAB,t1,optimize=True)
    if(avlb_mem < 2*o4gb):
      for i in range(O2):
        Yimjk[i,:,:,:] -= np.einsum('jmc,kc->jkm',X2[i,:,:,:],tx1,optimize=True)
        Xi2[i,:,:,:] += np.einsum('jkm,kmab->jab',Yimjk[i,:,:,:],l2,optimize=True)/Nkp
    else:
      Yimjk -= np.einsum('ijmc,kc->ijkm',X2,tx1,optimize=True)
      Xi2 += np.einsum('ijkm,kmab->ijab',Yimjk,l2,optimize=True)/Nkp
    del Yimjk, X2
    #
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    Xi2prod = np.einsum('ijab,ijab->',Xi2,np.conjugate(Xi2),optimize=True)
    print(f"Term 8: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # Term 9
    # Xi1 : 1/4Lg(kicd)R(jmcd)<jm||ka>
    #     : 1/4Lg(kicd)R(jmcd)t(kb)<jm||ba>
    # Xi2 : 1/4Lg(ijcd)R(kmcd)<km||ab>
    #     :    Lg(ijcd)R(kc)t(md)<km||ab>
    if(avlb_mem < 2*o4gb):
      for i in range(O2):
        Zkijm[i,:,:,:] = 0.25*np.einsum('jcd,kmcd->jkm',l2[i,:,:,:],
                                      tx2,optimize=True)/Nkp
    else:
      Zkijm += 0.25*np.einsum('ijcd,kmcd->ijkm',l2,tx2,optimize=True)/Nkp
    X2 = IJKA + np.einsum('jmba,kb->jmka',IJAB,t1,optimize=True)
    if(avlb_mem < 2*o4gb):
      for k in range(O2):
        Xi1 += np.einsum('ijm,jma->ia',Zkijm[k,:,:,:],X2[:,:,k,:],optimize=True)/NkpS
    else:
      Xi1 += np.einsum('kijm,jmka->ia',Zkijm,X2,optimize=True)/NkpS
    del X2
    X2 = np.einsum('ijcd,kc->ijkd',l2,tx1,optimize=True)
    if(avlb_mem < 2*o4gb):
      for i in range(O2):
        Zkijm[i,:,:,:] += np.einsum('jkd,md->jkm',X2[i,:,:,:],t1,optimize=True)
    else:
      Zkijm += np.einsum('ijkd,md->ijkm',X2,t1,optimize=True)
    Xi2 += np.einsum('ijkm,kmab->ijab',Zkijm,IJAB,optimize=True)/Nkp
    del Zkijm, X2
    #
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    Xi2prod = np.einsum('ijab,ijab->',Xi2,np.conjugate(Xi2),optimize=True)
    print(f"Term 9: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # Term 10
    # Xi2 :  P(ij,ab)Lg(ia)R(kc)<kj||cb>
    #     :    -P(ab)Lg(ka)R(kc)<ij||cb>
    #     :    -P(ij)Lg(ic)R(kc)<kj||ab>
    X1 = np.einsum('kc,kjcb->jb',tx1,IJAB,optimize=True)/Nkp
    X2 = np.einsum('ia,jb->ijab',l1,X1,optimize=True)*Nkp
    Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    Xi2 -= np.transpose(X2,axes=(1,0,2,3))
    Xi2 += np.transpose(X2,axes=(1,0,3,2))
    del X1, X2
    X1 = np.einsum('ka,kc->ac',l1,tx1,optimize=True)
    X2 = -np.einsum('ac,ijcb->ijab',X1,IJAB,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2
    X1 = np.einsum('ic,kc->ik',l1,tx1,optimize=True)
    X2 = -np.einsum('ik,kjab->ijab',X1,IJAB,optimize=True)
    Xi2 += X2 - np.transpose(X2,axes=(1,0,2,3))
    del X1, X2
    #
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    Xi2prod = np.einsum('ijab,ijab->',Xi2,np.conjugate(Xi2),optimize=True)
    print(f"Term 10: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # Term 11
    # Xi2 :  P(ij,ab)Lg(ikac)R(mc)<jm||kb>
    #     : -P(ij,ab)Lg(ikac)R(kd)<jc||db>
    X1 = np.einsum('mc,jmkb->jbkc',tx1,IJKA,optimize=True)
    X1 -= np.einsum('kd,jcdb->jbkc',tx1,IABC,optimize=True)
    X2 = np.einsum('ikac,jbkc->ijab',l2,X1,optimize=True)/Nkp
    Xi2 += X2 - np.transpose(X2,axes=(0,1,3,2))
    Xi2 -= np.transpose(X2,axes=(1,0,2,3))
    Xi2 += np.transpose(X2,axes=(1,0,3,2))
    del X1, X2
    #
    Xi1prod = np.einsum('ia,ia->',Xi1,np.conjugate(Xi1),optimize=True)
    Xi2prod = np.einsum('ijab,ijab->',Xi2,np.conjugate(Xi2),optimize=True)
    print(f"Term 11: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
    #
    # Term 12
    # Xi2 : -Lg(ijcd)R(kc)<kd||ab>
    X1 = np.einsum('ijcd,kc->ijkd',l2,tx1,optimize=True)
    Xi2 -= np.einsum('ijkd,kdab->ijab',X1,IABC,optimize=True)/Nkp
  del IABC, IJAB, IJKA
  if(avlb_mem < 2*o4gb):
    os.system(f"rm {scratch}/{mol_out}-Yimjk.npy")
    os.system(f"rm {scratch}/{mol_out}-Zkijm.npy")
  return Xi1, Xi2

#########################################################################
# CCSD Xi amplitudes for LR and EOM gradients with explicit loops over
# k points
#########################################################################
def Xi3k(mol_out,scratch,Nkp,Ktable,O2,V2,tx1,tx2,l1,l2,t1,F_ae,F_mi,
         F_me,D2):
  # L can be the ground or excited state Lambda amplitudes
  # Tx can be the LR Tx or the EOM R amplitudes
  IABC = np.load(f"{scratch}/{mol_out}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{mol_out}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{mol_out}-IJKA.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{mol_out}-Wmbej.npy",mmap_mode='r')
  #
  NkpS = Nkp*Nkp
  # Term 1
  # Xi1 : R1*Lg*H
  #       -R(jb)*(Lg(ja)*X(ib) + Lg(ib)*X(ja))
  #             _                  _                 _
  #       <0|L[[H,O1],R]|0> = <0|L(HR)conn|1> - <0|L(HR)disc|1>
  #MC I'm not sure whether X1 should be built as hkl(g) or hkh(k). The
  #latter would be easier to make but I'm not sure about the ij and ab
  #transpositions.
  # X2 = np.zeros(l2.shape,dtype=l2.dtype)
  X3 = np.zeros(l2.shape,dtype=l2.dtype)
  # X4 = np.zeros(l2.shape,dtype=l2.dtype)
  # P(ab) terms
  Fae3k = F_ae[Ktable]
  X2 = np.einsum('IJAijae,IJAeb->IJAijab',l2,Fae3k,optimize=True) 
  X1 = X2 - transpose_l(X2,Ktable,'k')
  del X2, Fae3k
  # P(ij) terms
  X2 = np.einsum('IJAimab,Jjm->IJAijab',l2,F_mi,optimize=True) 
  X1 += np.transpose(X2,axes=(1,0,2,4,3,5,6)) - X2 
  del X2
  B = Ktable
  ind_J = np.arange(Nkp).reshape(1,Nkp,1)
  for M in range (Nkp):
    E = Ktable[:,M,:]
    l2M = l2[:,M,:]
    ind_E = E[:,None,:]
    WM = W_mbej[ind_J,ind_E,B]
    X3 += np.einsum('IAimae,IJAjebm->IJAijab',l2M,WM,optimize=True)/Nkp
    del l2M, E, ind_E, WM
    I = M
    X3[I,:,I] += np.einsum('ia,Jjb->Jijab',l1[I],F_me,optimize=True)*Nkp
  X1 += X3 + np.transpose(transpose_l(X3,Ktable,'k'),axes=(1,0,2,4,3,5,6))
  # X1 = X3 + np.transpose(transpose_l(X3,Ktable,'k'),axes=(1,0,2,4,3,5,6))

  # for I in range(Nkp):
    # # P(ab) terms
    # X2[I,:,I] += np.einsum('Jijae,Jeb->Jijab',l2[I,:,I],F_ae,
    #                         optimize=True) 
    # # P(ij) terms
    # X3[I,:,I] += np.einsum('Jimab,Jjm->Jijab',l2[I,:,I],F_mi,
    #                          optimize=True) 
    # # P(ij,ab)-like terms
    # X4[I,:,I] += np.einsum('Mimae,JMJjebm->Jijab',l2[I,:,I],W_mbej,
    #                          optimize=True)/Nkp
    # X4[I,:,I] += np.einsum('ia,Jjb->Jijab',l1[I],F_me,
    #                          optimize=True)*Nkp
  # X1 = X2 - transpose_l(X2,Ktable,'k')
  # X1 += np.transpose(X3,axes=(1,0,2,4,3,5,6)) - X3 
  # X1 += X4 + np.transpose(transpose_l(X4,Ktable,'k'),axes=(1,0,2,4,3,5,6))
  # del X2, X3, X4, W_mbej
  X1 -= l2*D2
  Xi1 = -np.einsum('IJIijab,Jjb->Iia',X1,tx1,optimize=True)/Nkp
  del X1, X3
  #
  fmiprod  = np.einsum('Iia,Iia->',F_mi,np.conjugate(F_mi),optimize=True)
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  Xi2prod = 0
  print(f"Term 1: Fmi= {fmiprod/Nkp}")
  print(f"Term 1: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 2
  # Xi1 : -Lg(ijdb)t(md)R(kjcb)<mk||ac>
  ind_J = np.arange(Nkp).reshape(Nkp,1,1)
  X1 = np.zeros((Nkp,Nkp,Nkp,O2,V2,O2,V2), dtype=tx2.dtype)
  for N in range(Nkp):
    C = Ktable[:,N,:]
    tx2C = tx2[N][ind_J,C[None,:,:]]
    IJABC = IJAB[:,N,:]
    # Note that X1 ordering is diffent than with collective indexes to
    # maintain the indices correspondence with order in Ktable
    X1 += np.einsum('JMAnjcb,MAmnac->JAMjamb',tx2C,IJABC,
                    optimize=True)/Nkp
  X2 = np.einsum('IJMijdb,Mmd->IJMijmb',l2,t1,optimize=True)
  Xi1 -= np.einsum('IJlijmb,JIljamb->Iia',X2,X1,optimize=True)/Nkp
  del X1, X2
  #
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  print(f"Term 2: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 3
  # Xi1 : Lg(jibd)R(jkbc)<kd||ca>
  #     : Lg(jmba)R(jkbc)[<ki||mc>+t(md)<ki||dc>]
  # Xi2 : P(ij,ab)Lg(kica)[R(kmcd)-R(mc)t(kd)-t(mc)R(kd)]<mj||db>
  Yikdc = np.zeros((Nkp,Nkp,Nkp,O2,V2,O2,V2),dtype=tx1.dtype)
  ind_I = np.arange(Nkp).reshape(Nkp,1)
  for J in range(Nkp):
    B = Ktable[J]
    l2D = l2[J][ind_I,B] # (I,D,j,i,b,d)
    ind_K = np.arange(Nkp).reshape(1, Nkp, 1)
    tx2D = tx2[J][ind_K,B[:,None,:]] # (I,K,D,j,k,b,c)
    # Yikdc is stored with a different index order compared to the
    # collective index code to maintain correspondence with with
    # Ktable ordering Ktable[K,D,I]
    Yikdc += np.einsum('IDjibd,IKDjkbc->KDIkdic',l2D,tx2D,
                       optimize=True)/Nkp
  # KC[I,K,D] = Ktable[K,D,I] -- Yikdc's own formula
  KC = np.transpose(Ktable,axes=(2,0,1)) 
  IDX_K2 = np.arange(Nkp).reshape(1, Nkp, 1)
  IDX_D2 = np.arange(Nkp).reshape(1, 1, Nkp)
  IABCG = IABC[IDX_K2,IDX_D2,KC]  # (I,K,D,k,d,c,a)
  Xi1 += np.einsum('KDIkdic,IKDkdca->Iia',Yikdc,IABCG,optimize=True)/NkpS
  del IABCG,KC
  X2 = IJKA + np.einsum('Mmd,NIMnidc->NIMnimc',t1,IJAB,optimize=True)
  Xi1 += np.einsum('NIMnamc,NIMnimc->Iia',Yikdc,X2,optimize=True)/NkpS
  del X2
  #
  # this part maybe complicated because the order of the indices for
  # the X2 contraction is the opposite of that in Y in terms of Ktable
  # indexing.
  X2 = np.einsum('NIMnica,Mmc->NIMnima',l2,tx1,optimize=True)
  N = Ktable  # N[M,A,I] = Ktable[M,A,I] -- Yikdc's own naive formula, used directly
  IDX_I = np.arange(Nkp).reshape(1, 1, Nkp)  # I is the 3rd position
  IDX_M = np.arange(Nkp).reshape(Nkp, 1, 1)  # M is the 1st position
  X2G = X2[N,IDX_I,IDX_M]                    # (M,A,I,n,i,m,a)
  T1G = t1[N]                                # (M,A,I,n,d)
  Yikdc -= np.einsum('MAInima,MAInd->MAImaid', X2G, T1G, optimize=True)
  del X2, T1G, X2G
  X2 = np.einsum('NIMnica,Mmc->NIMnima',l2,t1,optimize=True)
  X2G = X2[N,IDX_I,IDX_M]                    # (M,A,I,n,i,m,a)
  T1G = tx1[N]                               # (M,A,I,n,d)
  Yikdc -= np.einsum('MAInima,MAInd->MAImaid', X2G, T1G, optimize=True)
  del X2, T1G, N, X2G
  X2 = np.zeros(tx2.shape,dtype=tx2.dtype)
  IDX_J = np.arange(Nkp).reshape(1, 1, Nkp)
  for M in range(Nkp):
    KD = Ktable[M]           # KD[A,I] = Ktable[M,A,I]
    YD = Yikdc[M]            # (A,I,m,a,i,d) -- no gather needed
    IJABD = IJAB[M][IDX_J,KD[:,:,None]]  # (A,I,J,m,j,d,b)
    X2 += np.einsum('AImaid,AIJmjdb->IJAijab',YD,IJABD,optimize=True)/Nkp
    del KD, YD, IJABD
  Xi2 = X2 - transpose_l(X2,Ktable,'k')
  Xi2 -= np.transpose(X2,axes=(1,0,2,4,3,5,6))
  Xi2 += np.transpose(transpose_l(X2,Ktable,'k'),axes=(1,0,2,4,3,5,6))
  del Yikdc, X2
  #
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  Xi2prod = np.einsum('IJAijab,IJAijab->',Xi2,np.conjugate(Xi2),optimize=True)
  print(f"Term 3: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 4
  # Xi1 :  1/2R(jkbc)Lg(jkbd)<di||ca>
  #     :  1/2R(jmbd)Lg(jmba)t(kc)<ik||cd>
  #     : -1/2R(jmbd)Lg(jmbc)t(kc)<ik||ad>
  # Xi2 : -1/2P(ab)R(kmcd)Lg(kmbd)<ij||ac>
  tx2C = transpose_l(tx2,Ktable,'k')
  l2D = transpose_l(l2,Ktable,'k')
  Ycd = 0.5*np.einsum('JKCjkcb,JKCjkdb->Ccd',tx2C,l2D,optimize=True)/NkpS
  del tx2C, l2D
  Xi1 -= np.einsum('Ccd,ICCidca->Iia',Ycd,IABC,optimize=True)/Nkp
  X2 = np.einsum('Kkc,IKKikcd->Iid',t1,IJAB,optimize=True)/Nkp
  Xi1 += np.einsum('Ida,Iid->Iia',Ycd,X2,optimize=True)
  del X2
  X2 = np.einsum('Kdc,Kkc->Kkd',Ycd,t1,optimize=True)
  Xi1 -= np.einsum('Kkd,IKIikad->Iia',X2,IJAB,optimize=True)/Nkp
  del X2
  YcdB = Ycd[Ktable]
  X2 = -np.einsum('IJAcb,IJAijac->IJAijab',YcdB,IJAB,optimize=True)
  Xi2 += X2 - transpose_l(X2,Ktable,'k')
  del Ycd, YcdB, X2
  #
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  Xi2prod = np.einsum('IJAijab,IJAijab->',Xi2,np.conjugate(Xi2),optimize=True)
  print(f"Term 4: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 5
  # Xi1 : -1/2Lg(id)R(kmcd)<km||ca>
  # Xi2 : -1/2P(ab)Lg(ijad)R(kmcd)<km||cb>
  #     :    -P(ab)Lg(ijad)R(md)t(kc)<km||cb>
  #     :    -P(ab)Lg(ijad)t(md)R(kc)<km||cb>
  #     :     P(ab)Lg(ijad)R(kc)<kd||cb>
  tx2D = transpose_l(tx2,Ktable,'k')
  IJABA = transpose_l(IJAB,Ktable,'k')
  Zda = -0.5*np.einsum('KMDkmdc,KMDkmac->Dda',tx2D,IJABA,optimize=True)/NkpS
  del tx2D, IJABA
  Xi1 += np.einsum('Iid,Ida->Iia',l1,Zda,optimize=True)
  X2 = np.einsum('Kkc,KMKkmcb->Mmb',t1,IJAB,optimize=True)/Nkp
  Zda -= np.einsum('Dmd,Dmb->Ddb',tx1,X2,optimize=True)
  del X2
  X2 = np.einsum('Kkc,KMKkmcb->Mmb',tx1,IJAB,optimize=True)/Nkp
  Zda -= np.einsum('Dmd,Dmb->Ddb',t1,X2,optimize=True) 
  del X2
  Zda += np.einsum('Kkc,KDKkdcb->Ddb',tx1,IABC,optimize=True)/Nkp
  ZdaB = Zda[Ktable]
  X2 = np.einsum('IJAijad,IJAdb->IJAijab',l2,ZdaB,optimize=True)
  Xi2 += X2 - transpose_l(X2,Ktable,'k')
  del Zda, ZdaB, X2
  #
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  Xi2prod = np.einsum('IJAijab,IJAijab->',Xi2,np.conjugate(Xi2),optimize=True)
  print(f"Term 5: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 6
  # Xi1 :  1/2Lg(jkbc)R(jmbc)<im||ka>
  #     : -1/2Lg(jibc)R(jmbc)t(kd)<mk||ad>
  #     : -1/2Lg(jkbc)R(jmbc)t(kd)<im||ad>
  #     :     Lg(jb)R(jmbd)<im||ad>
  # Xi2 : -1/2P(ij)Lg(jmcd)R(kmcd)<ik||ab>
  Ykm = 0.5*np.einsum('JKBjkbc,JKBjmbc->Kkm',l2,tx2,optimize=True)/NkpS
  Xi1 += np.einsum('Kkm,IKKimka->Iia',Ykm,IJKA,optimize=True)/Nkp
  X2 = np.einsum('Iik,JIAjkab->IJAijab',Ykm,IJAB,optimize=True)
  Xi2 += X2 - np.transpose(X2,axes=(1,0,2,4,3,5,6))
  del X2
  X2 = np.einsum('Kkd,MKMmkad->Mma',t1,IJAB,optimize=True)/Nkp
  Xi1 -= np.einsum('Iim,Ima->Iia',Ykm,X2,optimize=True)
  del X2
  X2 = -np.einsum('Mkm,Mkd->Mmd',Ykm,t1,optimize=True)
  X2 += np.einsum('Jjb,JMJjmbd->Mmd',l1,tx2,optimize=True)/Nkp
  Xi1 += np.einsum('Mmd,IMIimad->Iia',X2,IJAB,optimize=True)/Nkp
  del Ykm, X2
  #
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  Xi2prod = np.einsum('IJAijab,IJAijab->',Xi2,np.conjugate(Xi2),optimize=True)
  print(f"Term 6: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 7
  # Xi1 : -1/2Lg(ka)R(kmcd)<im||cd>
  # Xi2 : -1/2P(ij)Lg(kjab)R(kmcd)<im||cd>
  #     :    -P(ij)Lg(kjab)R(kc)t(md)<im||cd>
  #     :    -P(ij)Lg(kjab)t(kc)R(md)<im||cd>
  #     :    -P(ij)Lg(kjab)R(mc)<im||kc>
  Zik = -0.5*np.einsum('IMCimcd,IMCkmcd->Iik',IJAB,tx2,optimize=True)/NkpS
  Xi1 += np.einsum('Iik,Ika->Iia',Zik,l1,optimize=True)
  X2 = np.einsum('Mmd,IMIimcd->Iic',t1,IJAB,optimize=True)/Nkp
  Zik -= np.einsum('Iic,Ikc->Iik',X2,tx1,optimize=True)
  del X2
  X2 = np.einsum('Mmd,IMIimcd->Iic',tx1,IJAB,optimize=True)/Nkp
  Zik -= np.einsum('Iic,Ikc->Iik',X2,t1,optimize=True)
  del X2
  Zik += np.einsum('MIImikc,Mmc->Iik',IJKA,tx1,optimize=True)/Nkp
  # Another change to the IJKA order of contraction
  # Zik -= np.einsum('imkc,mc->ik',IJKA,tx1,optimize=True)/Nkp
  X2 = np.einsum('Iik,IJAkjab->IJAijab',Zik,l2,optimize=True)
  Xi2 += X2 - np.transpose(X2,axes=(1,0,2,4,3,5,6))
  del Zik, X2
  #
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  Xi2prod = np.einsum('IJAijab,IJAijab->',Xi2,np.conjugate(Xi2),optimize=True)
  print(f"Term 7: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 8
  # Xi1 : -1/4Lg(jkac)R(jkbd)<ic||bd>
  #     :  1/4Lg(jkac)t(mc)R(jkbd)<im||bd>
  # Xi2 :  1/4Lg(kmab)R(kmcd)<ij||cd>
  #     :     Lg(kmab)R(kc)t(md)<ij||cd>
  #     :     Lg(kmab)R(kc)<ij||cm>
  Yimjk = np.zeros((Nkp,Nkp,Nkp,O2,O2,O2,O2),dtype=tx1.dtype)
  ind_K = np.arange(Nkp).reshape(1,Nkp)
  X2 = np.zeros((Nkp,Nkp,Nkp,O2,V2,O2,O2),dtype=tx2.dtype)
  for I in range(Nkp):
    KM = Ktable[I]
    IJABI = IJAB[I]
    tx2I = tx2[ind_K,KM]
    Yimjk[I] += 0.25*np.einsum('JCijcd,JKCkmcd->JKijkm',IJABI,tx2I,
                               optimize=True)/Nkp
    IABCI = IABC[I]
    X2[I] -= 0.25*np.einsum('CBicbd,CJBjkbd->CJicjk',IABCI,tx2I,
                            optimize=True)/Nkp
    del IJABI, IABCI, tx2I, KM
  X2 += np.einsum('IMJimjk,Mmc->IMJicjk',Yimjk,t1,optimize=True)
  X3 = transpose_l(X2,Ktable,'j')
  Xi1 += np.einsum('IKJikjc,JKIjkac->Iia',X3,l2,optimize=True)/NkpS
  del X2, X3
  X2 = IJKA + np.einsum('IJMijdc,Mmd->IJMijmc',IJAB,t1,optimize=True)
  X3 = transpose_l(X2,Ktable,'k')
  Yimjk -= np.einsum('IJKijcm,Kkc->IJKijkm',X3,tx1,optimize=True)
  del X2, X3
  for K in range(Nkp):
    KM = Ktable[:,:,K]
    YimjkM = Yimjk[:,:,K]
    ind_A = np.arange(Nkp).reshape(1,1,Nkp)
    l2M = l2[K][KM[:,:,None],ind_A]
    Xi2 += np.einsum('IJijkm,IJAkmab->IJAijab',YimjkM,l2M,
                     optimize=True)/Nkp
    del KM, YimjkM, l2M
  del Yimjk
  #
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  Xi2prod = np.einsum('IJAijab,IJAijab->',Xi2,np.conjugate(Xi2),optimize=True)
  print(f"Term 8: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 9
  # Xi1 : 1/4Lg(kicd)R(jmcd)<jm||ka>
  #     : 1/4Lg(kicd)R(jmcd)t(kb)<jm||ba>
  # Xi2 : 1/4Lg(ijcd)R(kmcd)<km||ab>
  #     :    Lg(ijcd)R(kc)t(md)<km||ab>
  Zkijm = np.zeros((Nkp,Nkp,Nkp,O2,O2,O2,O2),dtype=tx1.dtype)
  for I in range(Nkp):
    KM = Ktable[I]
    l2I = l2[I]
    tx2I = tx2[ind_K,KM]
    Zkijm[I] += 0.25*np.einsum('JCijcd,JKCkmcd->JKijkm',l2I,tx2I,
                               optimize=True)/Nkp
    del l2I, tx2I, KM
  X2 = IJKA + np.einsum('JMKjmba,Kkb->JMKjmka',IJAB,t1,optimize=True)
  X3 = transpose_l(X2,Ktable,'j')
  Xi1 += np.einsum('KIJkijm,JIKjakm->Iia',Zkijm,X3,optimize=True)/NkpS
  del X2, X3
  X2 = np.einsum('IJKijcd,Kkc->IJKijkd',l2,tx1,optimize=True)
  X1 = t1[Ktable]
  Zkijm += np.einsum('IJKijkd,IJKmd->IJKijkm',X2,X1,optimize=True)
  del X1, X2
  for K in range(Nkp):
    KM = Ktable[:,:,K]
    ZkijmM = Zkijm[:,:,K]
    ind_A = np.arange(Nkp).reshape(1,1,Nkp)
    IJABM = IJAB[K][KM[:,:,None],ind_A]
    Xi2 += np.einsum('IJijkm,IJAkmab->IJAijab',ZkijmM,IJABM,
                     optimize=True)/Nkp
    del KM, ZkijmM, IJABM
  del Zkijm
  #
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  Xi2prod = np.einsum('IJAijab,IJAijab->',Xi2,np.conjugate(Xi2),optimize=True)
  print(f"Term 9: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 10
  # Xi2 :  P(ij,ab)Lg(ia)R(kc)<kj||cb>
  #     :    -P(ab)Lg(ka)R(kc)<ij||cb>
  #     :    -P(ij)Lg(ic)R(kc)<kj||ab>
  X1 = np.einsum('Kkc,KJKkjcb->Jjb',tx1,IJAB,optimize=True)/Nkp
  X2 = np.zeros(Xi2.shape,dtype=Xi2.dtype)
  for I in range(Nkp):      
    l1I = l1[I]
    X2[I,:,I] += np.einsum('ia,Jjb->Jijab',l1I,X1,optimize=True)*Nkp
    del l1I
  Xi2 += X2 - transpose_l(X2,Ktable,'k')
  Xi2 -= np.transpose(X2,axes=(1,0,2,4,3,5,6))
  Xi2 += np.transpose(transpose_l(X2,Ktable,'k'),axes=(1,0,2,4,3,5,6))
  del X1, X2
  X1 = np.einsum('Aka,Akc->Aac',l1,tx1,optimize=True)
  X2 = -np.einsum('Aac,IJAijcb->IJAijab',X1,IJAB,optimize=True)
  Xi2 += X2 - transpose_l(X2,Ktable,'k')
  del X1, X2
  X1 = np.einsum('Iic,Ikc->Iik',l1,tx1,optimize=True)
  X2 = -np.einsum('Iik,IJAkjab->IJAijab',X1,IJAB,optimize=True)
  Xi2 += X2 - np.transpose(X2,axes=(1,0,2,4,3,5,6))
  del X1, X2
  #
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  Xi2prod = np.einsum('IJAijab,IJAijab->',Xi2,np.conjugate(Xi2),optimize=True)
  print(f"Term 10: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 11
  # Xi2 :  P(ij,ab)Lg(ikac)R(mc)<jm||kb>
  #     : -P(ij,ab)Lg(ikac)R(kd)<jc||db>
  X1 = np.einsum('Cmc,JCKjmkb->JKCjkcb',tx1,IJKA,optimize=True)
  X1 -= np.einsum('Kkd,JCKjcdb->JKCjkcb',tx1,IABC,optimize=True)
  ind_J = np.arange(Nkp).reshape(1,Nkp,1)
  X2 = np.zeros(Xi2.shape,dtype=Xi2.dtype)
  for K in range(Nkp):
    C = Ktable[:,K,:]
    l2C = l2[:,K,:]
    X1C = X1[:,K,:][ind_J,C[:,None,:]]
    X2 += np.einsum('IAikac,IJAjkcb->IJAijab',l2C,X1C,
                   optimize=True)/Nkp
    del l2C, X1C
  del X1
  Xi2 += X2 - transpose_l(X2,Ktable,'k')
  Xi2 -= np.transpose(X2,axes=(1,0,2,4,3,5,6))
  Xi2 += np.transpose(transpose_l(X2,Ktable,'k'),axes=(1,0,2,4,3,5,6))
  del X2
  #
  Xi1prod = np.einsum('Iia,Iia->',Xi1,np.conjugate(Xi1),optimize=True)
  Xi2prod = np.einsum('IJAijab,IJAijab->',Xi2,np.conjugate(Xi2),optimize=True)
  print(f"Term 11: Xi1= {Xi1prod/Nkp}, Xi2= {Xi2prod/Nkp**3}")
  #
  # Term 12
  # Xi2 : -Lg(ijcd)R(kc)<kd||ab>
  X1 = np.einsum('IJKijcd,Kkc->IJKijkd',l2,tx1,optimize=True)
  ind_A = np.arange(Nkp).reshape(1,1,Nkp)
  for K in range(Nkp):
    D = Ktable[:,:,K]
    X1D = X1[:,:,K]
    IABCD = IABC[K][D[:,:,None],ind_A]
    Xi2 -= np.einsum('IJijkd,IJAkdab->IJAijab',X1D,IABCD,
                     optimize=True)/Nkp
    del D, X1D, IABCD
  del X1
  del IABC, IJAB, IJKA
  return Xi1, Xi2

#########################################################################
# CCSD rho1 transition density for LR and EOM 
#########################################################################
def TrDen1(T, O2, NB2, Nkp, tx1, tx2, l1, l2, t1, t2):
  # For now, implement only the LR term: <0|(1+Lg)[e^{-T}{p^{+}q}e^{T},X^{B}]|0>
  # Density is returned in MO basis
  if T==1:
    NkpS = Nkp*Nkp
    rho1 = np.zeros((NB2,NB2),dtype=tx1.dtype)
    # AI block
    # There are no contributions for the LR function
    #
    # IJ block
    # - R(ic)L(jc) -1/2Rx(ikcd)Lg(jkcd)
    X1 = -np.einsum('ic,jc->ij',tx1,l1,optimize=True)
    X1 -= 0.5*np.einsum('ikcd,jkcd->ij',tx2,l2,optimize=True)/NkpS
    rho1[:O2,:O2] = np.copy(X1)
    # AB block
    # L(ka)R(kb) +1/2Lg(kmca)Rx(kmcb)
    X2 = np.einsum('ka,kb->ab',l1,tx1,optimize=True)
    X2 += 0.5*np.einsum('kmca,kmcb->ab',l2,tx2,optimize=True)/NkpS
    rho1[O2:,O2:] = np.copy(X2)
    # IA block
    # Rx(ia) -t(ka)[Rx(ic)Lg(kc) + 1/2Rx(imcd)Lg(kmcd)]
    # -t(ic)[Rx(ka)Lg(ck) + 1/2Rx(kmad)Lg(cdkm)]
    rho1[:O2,O2:] = np.copy(tx1)
    rho1[:O2,O2:] += np.einsum('ik,ka->ia',X1,t1,optimize=True)
    rho1[:O2,O2:] -= np.einsum('ic,ca->ia',t1,X2,optimize=True)
    del X1, X2
    # + Rx(ikac)Lg(ck)
    rho1[:O2,O2:] += np.einsum('ikac,kc->ia',tx2,l1,optimize=True)/Nkp
    # -1/2t(kmad)Rx(ic)Lg(kmcd)
    X2 = 0.5*np.einsum('kmca,kmcd->ad',t2,l2,optimize=True)/NkpS
    rho1[:O2,O2:] -= np.einsum('id,ad->ia',tx1,X2,optimize=True)
    del X2
    # -1/2 t(imcd)Rx(ka)Lg(kmcd)
    X1 = 0.5*np.einsum('imcd,kmcd->ik',t2,l2,optimize=True)/NkpS
    rho1[:O2,O2:] -= np.einsum('ik,ka->ia',X1,tx1,optimize=True)
  return rho1

#########################################################################
# CCSD rho1 transition density for LR and EOM with explicit loops over
# k points
#########################################################################
def TrDen1_1k(O2,NB2,Nkp,Ktable,tx1,tx2,l1,l2,t1,t2):
  # For now, implement only the LR term: <0|(1+Lg)[e^{-T}{p^{+}q}e^{T},X^{B}]|0>
  # Density is returned in MO basis
  NkpS = Nkp*Nkp
  rho1 = np.zeros((Nkp,NB2,NB2),dtype=tx1.dtype)
  # AI block
  # There are no contributions for the LR function
  #
  # IJ block
  # - R(ic)L(jc) -1/2Rx(ikcd)Lg(jkcd)
  X1 = -np.einsum('Iic,Ijc->Iij',tx1,l1,optimize=True)
  X1 -= 0.5*np.einsum('IKCikcd,IKCjkcd->Iij',tx2,l2,optimize=True)/NkpS
  rho1[:,:O2,:O2] = np.copy(X1)
  # AB block
  # L(ka)R(kb) +1/2Lg(kmca)Rx(kmcb)
  X2 = np.einsum('Aka,Akb->Aab',l1,tx1,optimize=True)
  l2c = transpose_l(l2,Ktable,'k')
  tx2c = transpose_l(tx2,Ktable,'k')
  X2 += 0.5*np.einsum('KMAkmac,KMAkmbc->Aab',l2c,tx2c,optimize=True)/NkpS
  del tx2c
  rho1[:,O2:,O2:] = np.copy(X2)
  # IA block
  # Rx(ia) -t(ka)[Rx(ic)Lg(kc) + 1/2Rx(imcd)Lg(kmcd)]
  # -t(ic)[Rx(ka)Lg(ck) + 1/2Rx(kmad)Lg(cdkm)]
  rho1[:,:O2,O2:] = np.copy(tx1)
  rho1[:,:O2,O2:] += np.einsum('Iik,Ika->Iia',X1,t1,optimize=True)
  rho1[:,:O2,O2:] -= np.einsum('Iic,Ica->Iia',t1,X2,optimize=True)
  del X1, X2
  # + Rx(ikac)Lg(ck)
  rho1[:,:O2,O2:] += np.einsum('IKIikac,Kkc->Iia',tx2,l1,optimize=True)/Nkp
  # -1/2t(kmad)Rx(ic)Lg(kmcd)
  t2c = transpose_l(t2,Ktable,'k')
  X2 = 0.5*np.einsum('KMAkmac,KMAkmdc->Aad',t2c,l2c,optimize=True)/NkpS
  rho1[:,:O2,O2:] -= np.einsum('Iid,Iad->Iia',tx1,X2,optimize=True)
  del X2, l2c, t2c
  # -1/2 t(imcd)Rx(ka)Lg(kmcd)
  X1 = 0.5*np.einsum('IMCimcd,IMCkmcd->Iik',t2,l2,optimize=True)/NkpS
  rho1[:,:O2,O2:] -= np.einsum('Iik,Ika->Iia',X1,tx1,optimize=True)
  return rho1

#########################################################################
# Function to put a linearized matrix into lower triangular form and
# then square it
#########################################################################
def square_m(NDim,Lin,MType,Mat,MatSq):
  #
  # NDim : leading dimension of the matrix
  # Lin: T = Mat is linearized and needs to be reshaped into square.
  #      F = Mat is already in square form but stored lower/upper triangular
  # MType: Sym  = Square Mat in symmetrical form
  #        ASym = Square Mat in anti-symmetrical form
  #        Herm = Square Mat in Hermitian form
  #        AHer = Square Mat in anti-Hermitian form
  if (Lin):
    off = 0
    for N in range(NDim):
      MatSq[N,:N+1] = np.copy(Mat[off:off+N+1])
      off += N+1
  if(MType == "Sym"):
    MatSq = MatSq + MatSq.T
  elif(MType == "ASym"):
    MatSq = MatSq - MatSq.T
  elif(MType == "Herm"):
    MatSq = MatSq + np.conjugate(MatSq).T
  elif(MType == "AHer"):
    MatSq = MatSq - np.conjugate(MatSq).T
  else:
    print(f"Wrong matrix type in square_m: {MType}")
    exit()
  np.fill_diagonal(MatSq,np.diag(MatSq)/2)
  return MatSq

#########################################################################
# Function to form the auxiliary arrays for the Fourier Transform
#########################################################################
def fill_kl(ipbc):
  # ipbc: integer array containing PBC info
  # kp: output array with k-point values in [-pi,pi) range
  # l_list: output integer array with index over repeated cells: [0,+1,-1,+2,-2,...
  npdir = ipbc[0]
  nmtpbc = ipbc[1]
  nrecip = ipbc[9]
  # Read cell list from ipbc
  l_listall = np.array(ipbc[21:]).reshape((nmtpbc,3))
  if(npdir == 1):
    l_list = l_listall[:,0].reshape((1,nmtpbc))
  elif(npdir == 2):
    l_list = l_listall[:,:2].reshape((nmtpbc,2))
    l_list = np.transpose(l_list,axes=(1,0))
  else:
    l_list = np.transpose(l_listall,axes=(1,0))
    # l_list = l_listall
  print(f"l_list : npdir = {npdir}, length {len(l_list)}\n {l_list}\n")
  # Read number of k points in each direction from ipbc
  shift = ipbc[11]
  ndimk = ipbc[12:15]
  # ndimk = [4,6,0]
  print(f"ndimk: {ndimk}, max {max(ndimk)}\n")
  # exit()
  # #Build l_list
  # l_list = [0]
  # for i in range(1,nmtpbc,2):
  #   # l_list.append(-(i//2 + 1))
  #   l_list.append(i//2 + 1)
  #   l_list.append(-(i//2 + 1))
  # Build kp list
  # kp = []
  # ndimktot = np.zeros((npdir))
  # ndimktot = []
  #
  # kp is dimentioned [npdir,max(ndimk)] to handle grids with different
  # number of points in different directions. Along the direction with
  # fewer k points, say n, only the first ndimk[n] elements are filled
  # and the rest are zero.
  Nkp = 1
  if nrecip == 1:
    kp = [0]
  else:
    # kp = np.zeros((npdir,max(ndimk)))
    kp = []
    for n in range(npdir):
      Nkp *= ndimk[n]
      pi_frac = np.pi/ndimk[n]
      nk = 0
      kpn = []
      for k in range(-ndimk[n]+shift,ndimk[n]+shift,2):
        nk += 1
        if(k >= ndimk[n]):
          kpn.append(pi_frac * (k-2*ndimk[n]))
        else:
          kpn.append(pi_frac * k)
      # kp[n,:ndimk[n]] = np.copy(kpn)
      kp.append(kpn)
      # ndimktot[n] = nk
      # ndimktot.append(nk)
    # if(npdir == 3):
    #   kp = np.array(kp).reshape((ndimk[0],ndimk[1],ndimk[2]))
    # elif(npdir == 2):
    #   kp = np.array(kp).reshape((ndimk[0],ndimk[1]))
  # kp = np.array(kp)
  # print(f"kp1 : Nkp= {Nkp}, shape {kp.shape}, length {len(kp)}, size {np.size(kp)}\n {kp}\n")
  print(f"kp1 : Nkp= {Nkp}, \n {kp}\n")
  # exit()
  # #Build kp
  # kp = []
  # if nrecip == 1:
  #   kp = [0]
  # elif nrecip % 2 == 0:
  #   for k in range(1, nrecip, 2):
  #     kp.append((np.pi * (k - nrecip) ) / nrecip)
  #   for k in range(1, nrecip, 2):
  #     kp.append((np.pi * (nrecip - k) ) / nrecip)
  # elif nrecip % 2 != 0 and nrecip != 1:
  #   tmpn = np.ceil(nrecip/2) - 1
  #   for k in range(int(tmpn + 1)):
  #     kp.append((np.pi * (k - tmpn) ) / tmpn )
  #   for k in range(1,int(tmpn)):
  #     kp.append((np.pi * (tmpn - k) ) / tmpn )
  # print(f"kp2 : {kp}\n")
  # exit()
  return Nkp, kp, l_list

#########################################################################
# Function to evalute the coefficients of the Fourier tranform in
# linearized form
#########################################################################
def fourier_coef(ipbc,dk,dir_dk):
  # dk: = F: regular MO(k) basis,
  #     = T: dS/dK in MO(k) basis, dir_dk: direction of k-derivative
  npdir = ipbc[0]
  nmtpbc = ipbc[1]
  Nkp, kp, l_list = fill_kl(ipbc)
  # co = np.einsum('nk,nl->nkl',kp,l_list,optimize=True)
  # cof = np.cos(co) - 1j*np.sin(co)
  # if(dk):
  #   lcof = -1j*np.array(l_list)
  #   cofdk = np.einsum('nkl,nl->nkl',cof,lcof,optimize=True)
  #   cof = cofdk
  # if(npdir == 1):
  #   coff = cof[0,:,:]
  # elif(npdir == 2):
  #   coff = np.einsum('yl,xl->yxl',cof[1,:,:],cof[0,:,:],optimize=True)
  # else:
  #   coff = np.einsum('zl,yl,xl->zyxl',cof[2,:,:],cof[1,:,:],cof[0,:,:],
  #                    optimize=True)
  cof = []
  # cofdk = []
  for n in range(npdir):
    co = np.einsum('k,l->kl',kp[n],l_list[n,:],optimize=True)
    cof.append(np.cos(co) - 1j*np.sin(co))
  if(dk):
    lcof = -1j*np.array(l_list[dir_dk])
    cofdk = np.einsum('kl,l->kl',cof[dir_dk],lcof,optimize=True)
    cof[dir_dk] = np.copy(cofdk)
    # prod2 = np.einsum('hkl,hkl->',cof,cof,optimize=True)
    # prod3 = np.einsum('kl,kl->',cofdk,cofdk,optimize=True)
    # print(f"passa di qua, {dir_dk}, {prod1}, {prod2}, {prod3}")
    # exit()
  if(npdir == 1):
    coff = cof[0]
  elif(npdir == 2):
    coff = np.einsum('yl,xl->yxl',cof[1],cof[0],optimize=True)
  else:
    coff = np.einsum('zl,yl,xl->zyxl',cof[2],cof[1],cof[0],
                     optimize=True)
  coff = coff.reshape((Nkp,nmtpbc))
  return coff
  
#########################################################################
# Funtion to evaluate reciprocal vectors
#########################################################################
def reciprocal(npdir,tv):
  # npdir: number of periodic directions
  # tv: tranlation vectors in real space
  # reciprocal vectors are returned in b_vecs
  two_pi = round(2*np.pi,10)
  if(npdir == 1):
    b_vecs = two_pi * np.array(tv[0]) / np.dot(tv[0],tv[0])
    b_vecs = b_vecs.reshape((1,3))
  elif(npdir == 2):
    # We need a rotation matrix by pi/2 anticlockwise around an exis
    # perpendicular to the two translation vectors (this bit was built
    # with Claude)
    # 1. Axis orthogonal to the plane of v1, v2 (right-hand rule: v1 -> v2)
    axis = np.cross(tv[0],tv[1])
    norm = np.linalg.norm(axis)
    axis = axis / norm  # normalize to unit vector
    # 2. Skew-symmetric cross-product matrix K of the axis
    kx, ky, kz = axis
    K = np.array([
        [0, -kz,  ky],
        [kz,  0, -kx],
        [-ky, kx,  0]
    ])
    # 3. Rodrigues' rotation formula for theta = 90 degrees
    #    sin(90°) = 1, cos(90°) = 0  ->  R = I + K + K^2
    I = np.eye(3)
    rot = I + K + K @ K
    print(f"Rot: I: {I}\n K: {K}\n rot: {rot}\n")
    # Now form reciprocal vectors
    b_vecs = np.zeros((2,3))
    for n in range(npdir):
      n1 = (n+1)%npdir
      rot_a = np.einsum('ij,j->i',rot,tv[n1],optimize=True)
      print(f"n1: {n1}, rot_a: {rot_a}")
      b_vecs[n,:] = two_pi*rot_a/np.dot(tv[n],rot_a)
      print(f"b_vecs: {b_vecs[n,:]}")
  elif(npdir == 3):
    Vol = np.dot(tv[0],np.cross(tv[1],tv[2]))
    fact = two_pi/Vol
    b_vecs = np.zeros((3,3))
    for n in range(npdir):
      n1 = (n+1)%npdir
      n2 = (n+2)%npdir
      b_vecs[n,:] = fact * np.cross(tv[n1],tv[n2])
  else:
    print(f"Number of periodic directions is wrong: NPdir = {npdir}\n")
    exit()
  return b_vecs

#########################################################################
# Funtion to form a linearized map of the k-point mesh
#########################################################################
def form_map_kp(npdir,ndimk):
  map_kp = []
  if(npdir == 1):
    for k1 in range(ndimk[0]):
      map_kp.append([k1])
  elif(npdir == 2):
    for k2 in range(ndimk[1]):
      for k1 in range(ndimk[0]):
        map_kp.append([k1,k2])
  elif(npdir == 3):
    for k3 in range(ndimk[2]):
      for k2 in range(ndimk[1]):
        for k1 in range(ndimk[0]):
          map_kp.append([k1,k2,k3])
  return map_kp

#########################################################################
# Funtion to determine the fourth k point for 2ERIs based on the other
# three and momentum conservation
#########################################################################
def momentum_cons(npdir,ndimk,map_kp,axis,n,k,h,g):
  # Integral is expected in physicist notation <n(1)k(2)|h(1)g(2)>
  #
  # Momentum conservation requires k_n - k_h + k_k - k_g = mod(G)
  #
  # However, the indices n,k,h,g come in linearized form and we need
  # to extract the the coordinate of each k point on the grid
  # axis: 0-3, says which k point is requested
  #
  if(axis <0 or axis>3):
    print(f"Wrong k point request in momentum_cons: axis={axis}.\n")
    exit()
  if(npdir == 0):
    kp = 0
  elif(npdir == 1):
    n1 = np.array(map_kp[n])
    k1 = np.array(map_kp[k])
    h1 = np.array(map_kp[h])
    g1 = np.array(map_kp[g])
    if(axis == 0):
      n1 = abs(-(-h1+k1-g1)%ndimk[0])
      kp = map_kp.index([n1]) 
    elif(axis == 1):
      k1 = abs(-(n1-h1-g1)%ndimk[0])
      kp = map_kp.index([k1]) 
    elif(axis == 2):
      h1 = abs((n1+k1-g1)%ndimk[0])
      kp = map_kp.index([h1]) 
    else:
      g1 = abs((n1-h1+k1)%ndimk[0])
      kp = map_kp.index([g1]) 
    # print(f"n1: {n1}, k1: {k1}, h1: {h1}, g1: {g1}, g: {g}")
    # g = abs((n-h+k)%ndimk[0])
  elif(npdir == 2):
    n1,n2 = np.array(map_kp[n])
    k1,k2 = np.array(map_kp[k])
    h1,h2 = np.array(map_kp[h])
    g1,g2 = np.array(map_kp[g])
    if(axis == 0):
      n1 = abs(-(-h1+k1-g1)%ndimk[0])
      n2 = abs(-(-h2+k2-g1)%ndimk[1])
      kp = map_kp.index([n1,n2]) 
    elif(axis == 1):
      k1 = abs(-(n1-h1-g1)%ndimk[0])
      k2 = abs(-(n2-h2-g2)%ndimk[1])
      kp = map_kp.index([k1,k2]) 
    elif(axis == 2):
      h1 = abs((n1+k1-g1)%ndimk[0])
      h2 = abs((n2+k2-g2)%ndimk[1])
      kp = map_kp.index([h1,h2]) 
    else:
      g1 = abs((n1-h1+k1)%ndimk[0])
      g2 = abs((n2-h2+k2)%ndimk[1])
      kp = map_kp.index([g1,g2]) 
    # print(f"n:{n,n1,n2}, k:{k,k1,k2}, h:{h,h1,h2}, g:{g,g1,g2}")
  elif(npdir == 3):
    n1,n2,n3 = np.array(map_kp[n])
    k1,k2,k3 = np.array(map_kp[k])
    h1,h2,h3 = np.array(map_kp[h])
    if(axis == 0):
      n1 = abs(-(-h1+k1-g1)%ndimk[0])
      n2 = abs(-(-h2+k2-g2)%ndimk[1])
      n3 = abs(-(-h3+k3-g3)%ndimk[2])
      kp = map_kp.index([n1,n2,n3]) 
    elif(axis == 1):
      k1 = abs(-(n1-h1-g1)%ndimk[0])
      k2 = abs(-(n2-h2-g2)%ndimk[1])
      k3 = abs(-(n3-h3-g3)%ndimk[2])
      kp = map_kp.index([k1,k2,k3]) 
    elif(axis == 2):
      h1 = abs((n1+k1-g1)%ndimk[0])
      h2 = abs((n2+k2-g2)%ndimk[1])
      h3 = abs((n3+k3-g3)%ndimk[2])
      kp = map_kp.index([h1,h2,h3]) 
    else:
      g1 = abs((n1-h1+k1)%ndimk[0])
      g2 = abs((n2-h2+k2)%ndimk[1])
      g3 = abs((n3-h3+k3)%ndimk[2])
      kp = map_kp.index([g1,g2,g3]) 
    # print(f"n:{n,n1,n2,n3}, k:{k,k1,k2,k3}, h:{h,h1,h2,h3}, g:{g,g1,g2,g3}")
  else:
    print(f"Number of periodic directions is wrong in mom_cons: {npdir}.\n")
    exit()
  return kp

#########################################################################
# Function to build table of momentum conserving k points for 2e
# arrays when using compressed storage
#########################################################################
def get_ktable(npdir, ndimk):
  """
  Build the full momentum-conservation table Ktable[ki,kj,kk] = kl for
  the linearized k-point mesh produced by form_map_kp, equivalent to
  calling momentum_cons(npdir, ndimk, map_kp, axis=3, n=ki, k=kj, h=kk, g=_)
  for every (ki,kj,kk), but fully vectorized (no explicit index loops and
  no linear map_kp.index() scans).

  Convention (physicist notation <n(1)k(2)|h(1)g(2)>, per direction d):
      kp(n)_d - kp(h)_d + kp(k)_d - kp(g)_d = 0  (mod ndimk[d])
  i.e. for the ERI tensor I[ki,kj,kk,i,j,k,l] (kl implicit):
      ki_d + kj_d - kk_d - kl_d = 0  (mod ndimk[d])

  Parameters
  ----------
  npdir : int
      Number of periodic directions (0, 1, 2, or 3).
  ndimk : sequence of int
      Number of k points along each periodic direction (length >= npdir).

  Returns
  -------
  Ktable : ndarray, shape (Nkp,Nkp,Nkp), int
      Nkp = prod(ndimk[:npdir]) (Nkp=1 if npdir==0).
      Ktable[ki,kj,kk] gives kl in the same linearized indexing as
      form_map_kp(npdir, ndimk).
  """
  if npdir == 0:
    return np.zeros((1, 1, 1), dtype=int)
  #
  map_kp = form_map_kp(npdir, ndimk)
  coords = np.array(map_kp)    # (Nkp, npdir), same order as map_kp
  dims = np.array(ndimk[:npdir])
  #
  # linear index weights matching form_map_kp's nesting (dim 0 fastest-varying)
  weights = np.concatenate(([1], np.cumprod(dims[:-1]))) if npdir > 1 else np.array([1])
  #
  Ci = coords[:, None, None, :]  # varies with ki
  Cj = coords[None, :, None, :]  # varies with kj
  Ck = coords[None, None, :, :]  # varies with kk
  Cl = (Ci - Ck + Cj) % dims     # per-direction conservation
  Ktable = (Cl @ weights).astype(int)   # back to linearized kl index
  #
  return Ktable

#########################################################################
# Function to transpose the last orbital index in k-compress 2e
# arrays, where the k point is implicit
#########################################################################
def transpose_l(I, Ktable, target):
  """
  Swap the implicit last orbital index l of a k-point ERI tensor
  I[ki,kj,kk,i,j,k,l] with one of the explicit orbital indices i, j, or k,
  recomputing the k-point indices so crystal-momentum conservation holds
  for the new axis order.

  Convention: Ktable[ki,kj,kk] = kl satisfies
      k(ki) - k(kk) + k(kj) - k(kl) = G   (a reciprocal lattice vector)
  i.e. k(ki) + k(kj) - k(kk) - k(kl) = G.

  Parameters
  ----------
  I : ndarray, shape (Nkp,Nkp,Nkp,NO,NO,NO,NO), complex
      ERI tensor I[ki,kj,kk,i,j,k,l]; kl is implicit.
  Ktable : ndarray, shape (Nkp,Nkp,Nkp), int
      Momentum-conservation lookup table under the convention above.
  target : {'i', 'j', 'k'}
      Which orbital slot to swap with l.

  Returns
  -------
  ndarray, same shape/dtype as I:
      target='i' -> I_new[kl,kj,kk,l,j,k,i]
      target='j' -> I_new[ki,kl,kk,i,l,k,j]
      target='k' -> I_new[ki,kj,kl,i,j,l,k]
  """
  Nkp = Ktable.shape[0]
  ar = np.arange(Nkp)
  #
  if target == 'i':
    # ki = Ktable[kl,kk,kj]  (since ki+kj-kk-kl=G is symmetric in swapping
    # the roles of ki<->kl together with kj<->kk)
    ki = Ktable.transpose(0, 2, 1)  # ki[kl,kj,kk] = Ktable[kl,kk,kj]
    kj = ar.reshape(1, Nkp, 1)
    kk = ar.reshape(1, 1, Nkp)
    gathered = I[ki, kj, kk]             # gather over k-axes only
    return np.swapaxes(gathered, 3, 6)   # i <-> l
  elif target == 'j':
    # kj = Ktable[kk,kl,ki]
    ki_ = ar.reshape(Nkp, 1, 1)
    kj = Ktable.transpose(2, 1, 0)     # kj[ki,kl,kk] = Ktable[kk,kl,ki]
    kk = ar.reshape(1, 1, Nkp)
    gathered = I[ki_, kj, kk]
    return np.swapaxes(gathered, 4, 6)   # j <-> l
  elif target == 'k':
    # kk = Ktable[ki,kj,kl]  -- no transpose needed: the convention is
    # symmetric under swapping kk <-> kl directly (ki+kj-kk-kl=G is
    # unchanged in form if kk and kl trade places)
    ki_ = ar.reshape(Nkp, 1, 1)
    kj_ = ar.reshape(1, Nkp, 1)
    kk = Ktable                        # kk[ki,kj,kl] = Ktable[ki,kj,kl]
    gathered = I[ki_, kj_, kk]
    return np.swapaxes(gathered, 5, 6)   # k <-> l
  else:
    raise ValueError("target must be one of 'i', 'j', 'k'")

#########################################################################
# Function to perform the Fourier tranform of a 2-index array
#########################################################################
def fourier(FT,ipbc,MatIn,dk,dir_dk):
  # FT: "Dir" = R -> k
  #     "Inv" = k -> R
  # ipbc: integer array containing PBC info
  # MatIn : input array (real for Dir/complex for Inv)
  # MatOut : output array (complex for Dir/real for Inv)
  # dk: = F: regular MO(k) basis, = T: dS/dK in MO(k) basis
  #                               dir_dk: direction of k-derivative
  #
  # npdir = ipbc[0]
  # nmtpbc = ipbc[1]
  # # ndimk = ipbc[12:15]
  # Nkp, kp, l_list = fill_kl(ipbc)
  # co = np.einsum('nk,nl->nkl',kp,l_list,optimize=True)
  # cof = np.cos(co) - 1j*np.sin(co)
  # if(dk):
  #   lcof = -1j*np.array(l_list)
  #   cofdk = np.einsum('nkl,nl->nkl',cof,lcof,optimize=True)
  #   cof = cofdk
  # # cof = np.ones((3,max(ndimk),nmtpbc),dtype=complex)
  # # Nkp = 1
  # # for n in range(npdir):
  # #   Nkp *= ndimk[n]
  # #   cof[n,:,:] = np.cos(co[n,:,:]) - 1j*np.sin(co[n,:,:])
  # if(npdir == 1):
  #   coff = cof[0,:,:]
  # elif(npdir == 2):
  #   coff = np.einsum('yl,xl->yxl',cof[1,:,:],cof[0,:,:],optimize=True)
  # else:
  #   coff = np.einsum('zl,yl,xl->zyxl',cof[2,:,:],cof[1,:,:],cof[0,:,:],
  #                    optimize=True)
  # coff = coff.reshape((Nkp,nmtpbc))
  # # coff = np.zeros((Nkp,nmtpbc),dtype=complex)
  # # kk = 0
  # # for kz in range(ndimk[2]):
  # #   for ky in range(ndimk[1]):
  # #     for kx in range(ndimk[0]):
  # #       coff[kk,:] = np.einsum('l,l,l->l',cof[2,kz,:],cof[1,ky,:],cof[0,kx,:],optimize=True)
  # #       
  # # cof[n,:,:] = np.cos(co[n,:,:]) - 1j*np.sin(co[n,:,:])
  # # cof = np.cos(co) + 1j*np.sin(co)
  cof = fourier_coef(ipbc,dk,dir_dk)
  if(FT == "Dir"):
    MatOut = np.einsum('kl,ln->kn',cof,MatIn,optimize=True)
    # if(dk):
    #   lcof = -1j*np.array(l_list)
    #   # lcof = 1j*np.array(l_list)
    #   MatOut = np.einsum('kl,l,ln->kn',cof,lcof,MatIn,optimize=True)
    # else:
    #   MatOut = np.einsum('kl,ln->kn',cof,MatIn,optimize=True)
  elif(FT == "Inv"):
    MatOut = np.einsum('kl,kn->ln', cof, MatIn, optimize=True, dtype=real)
    print(f"Inverse FT needs to be tested")
    exit()
  else:
    print(f"Wrong call to fourier: {FT}")
    exit()
  return MatOut

#########################################################################
# Function for AO(k)<->MO(k) tranformation for a 2-index array
#########################################################################
def basis_tran(Opt,LinIn,LinOut,MType,NDim,Nkp,MOCoef,MatIn):
  # Opt: "Dir" = AO(k)->MO(k)
  #      "Inv" = MO(k)->AO(k)
  # LinIn: T = MatIn is linearized and needs to be reshaped into square.
  #        F = MatIn is already in square form but stored lower/upper triangular
  # LinOut: T = MatOut is returned linearized 
  #         F = MatOut is returned square 
  # MType: Sym  = Square Mat in symmetrical form
  #        ASym = Square Mat in anti-symmetrical form
  #        Herm = Square Mat in Hermitian form
  #        AHer = Square Mat in anti-Hermitian form
  # NDim: leading dimension of input matrix
  # Nkp: number og k points
  # MOCoef: array to MO(k) coefficients
  # MatIn: input array [Nkp,:]
  # MatOut: output array
  # All arrays are expected to be complex
  if(Opt=="Dir"):
    if(LinIn):
      mat_k = np.zeros((Nkp,NDim,NDim),dtype=complex)
      for k in range(Nkp):
        mat_k[k,:,:] = square_m(NDim,True,MType,MatIn[k,:],mat_k[k,:,:])
      temp = np.einsum("kin,knm->kim",np.conjugate(MOCoef),mat_k,optimize=True)
      MatOut = np.einsum("kjm,kim->kij",MOCoef,temp,optimize=True)
    else:
      temp = np.einsum("kin,knm->kim",np.conjugate(MOCoef),MatIn,optimize=True)
      MatOut = np.einsum("kjm,kim->kij",MOCoef,temp,optimize=True)
    if(LinOut):
      print(f"This LinOut:{LinOut} is not implemented yet in basis_tran")
      exit()
  elif(Opt=="Inv"):
    print(f"This Opt:{Opt} is not implemented yet in basis_tran")
    exit()
  else:
    print(f"Wrong Opt in basis_tran: {Opt}")
    exit()
  return MatOut

#########################################################################
# Function to compute the molecular mass in grams/mole and create a
# text list of the atoms in the molecule/unit cell
#########################################################################
def mol_mass(atoms_list):
  # atoms_list: list of atoms in the molecule
  elements_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',\
                 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti',\
                 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',\
                 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',\
                 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',\
                 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',\
                 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',\
                 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',\
                 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',\
                 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Ct', 'Es', 'Fm', 'Md', 'No',\
                 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',\
                 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
  elements_dict = {'H' : 1.008,'He' : 4.003, 'Li' : 6.941, 'Be' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'Ne' : 20.180, 'Na' : 22.990, 'Mg' : 24.305,\
                 'Al' : 26.982, 'Si' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'Cl' : 35.453, 'Ar' : 39.948, 'K' : 39.098, 'Ca' : 40.078,\
                 'Sc' : 44.956, 'Ti' : 47.867, 'V' : 50.942, 'Cr' : 51.996,\
                 'Mn' : 54.938, 'Fe' : 55.845, 'Co' : 58.933, 'Ni' : 58.693,\
                 'Cu' : 63.546, 'Zn' : 65.38, 'Ga' : 69.723, 'Ge' : 72.631,\
                 'As' : 74.922, 'Se' : 78.971, 'Br' : 79.904, 'Kr' : 84.798,\
                 'Rb' : 84.468, 'Sr' : 87.62, 'Y' : 88.906, 'Zr' : 91.224,\
                 'Nb' : 92.906, 'Mo' : 95.95, 'Tc' : 98.907, 'Ru' : 101.07,\
                 'Rh' : 102.906, 'Pd' : 106.42, 'Ag' : 107.868, 'Cd' : 112.414,\
                 'In' : 114.818, 'Sn' : 118.711, 'Sb' : 121.760, 'Te' : 126.7,\
                 'I' : 126.904, 'Xe' : 131.294, 'Cs' : 132.905, 'Ba' : 137.328,\
                 'La' : 138.905, 'Ce' : 140.116, 'Pr' : 140.908, 'Nd' : 144.243,\
                 'Pm' : 144.913, 'Sm' : 150.36, 'Eu' : 151.964, 'Gd' : 157.25,\
                 'Tb' : 158.925, 'Dy': 162.500, 'Ho' : 164.930, 'Er' : 167.259,\
                 'Tm' : 168.934, 'Yb' : 173.055, 'Lu' : 174.967, 'Hf' : 178.49,\
                 'Ta' : 180.948, 'W' : 183.84, 'Re' : 186.207, 'Os' : 190.23,\
                 'Ir' : 192.217, 'Pt' : 195.085, 'Au' : 196.967, 'Hg' : 200.592,\
                 'Tl' : 204.383, 'Pb' : 207.2, 'Bi' : 208.980, 'Po' : 208.982,\
                 'At' : 209.987, 'Rn' : 222.081, 'Fr' : 223.020, 'Ra' : 226.025,\
                 'Ac' : 227.028, 'Th' : 232.038, 'Pa' : 231.036, 'U' : 238.029,\
                 'Np' : 237, 'Pu' : 244, 'Am' : 243, 'Cm' : 247, 'Bk' : 247,\
                 'Ct' : 251, 'Es' : 252, 'Fm' : 257, 'Md' : 258, 'No' : 259,\
                 'Lr' : 262, 'Rf' : 261, 'Db' : 262, 'Sg' : 266, 'Bh' : 264,\
                 'Hs' : 269, 'Mt' : 268, 'Ds' : 271, 'Rg' : 272, 'Cn' : 285,\
                 'Nh' : 284, 'Fl' : 289, 'Mc' : 288, 'Lv' : 292, 'Ts' : 294,\
                 'Og' : 294}
  mol_weight = 0
  atoms_list_names = []
  for i in range(len(atoms_list)):
    atom = elements_list[atoms_list[i]-1]
    weight = elements_dict.get(atom)
    mol_weight += weight
    atoms_list_names.append(atom)
  return mol_weight, atoms_list_names

#########################################################################
# Function to print out tensors
#########################################################################
def print_tensor(mol_out,PertType,iw,W,tensor,tensorDQ,alpha_mix,atoms_list):
  # mol_out: output file
  # PertType: tensor type
  # iw: current frequency for the printing
  # W: frequency value
  # tensor: tensor array
  # tensorDQ: temporary dipole-quadrupole tensor
  # alpha_mix: mixed-gauge dipole_L-dipole_V tensor
  # atoms_list: list of atoms in the molecule/unit cell
  #
  if(PertType == "OR_L" or PertType == "OR_V" or
     PertType == "FullOR_L" or PertType == "FullOR_V" ):
    # Evaluate conversion factors for OR:
    # fact1: a.u. --> deg/[dm (g/ml)] 
    # fact2: deg/[dm (g/ml)] --> deg L /(dm mol)
    mol_weight, atoms_list_names = mol_mass(atoms_list)
    fact1 = 72e6 * hbar**2 * N_A * W**2 / (c**2 * m_e**2 * mol_weight)
    fact2 = fact1*mol_weight/1000
  #
  if(PertType == "DipE"):
    #
    # Electric Dipole-Electric Dipole Length Gauge
    #
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n DipE(LG)-DipE(LG) Polarizability in a.u. for W = {W:.6f} a.u.\n")
    # Symmetrize
    tensor[iw,:,:] = (tensor[iw,:,:] + tensor[iw,:,:].T)/2
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
  elif(PertType == "DipEV" and iw>0):
    #
    # Electric Dipole-Electric Dipole Modificed Velocity Gauge
    #
    # For velocity gauge tensors, remove static limit before printing
    tensor[iw,:,:] -= tensor[0,:,:]
    # Then divide by frequency squared
    tensor[iw,:,:] /= -W**2
    # Symmetrize
    tensor[iw,:,:] = (tensor[iw,:,:] + tensor[iw,:,:].T)/2
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n DipE(MVG)-DipE(MVG) Polarizability in a.u. for W = {W:.6f} a.u.\n")
      writer.write(f" Static limit removed\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
  elif(PertType == "DipEV" and iw==0):
    #
    # Electric Dipole-Electric Dipole Modificed Velocity Gauge
    # Unphysical static limit
    #
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n DipE(MVG)-DipE(MVG) (Unphysical) Static Polarizability in a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {-tensor[iw,ip,0].real:+.6f} {-tensor[iw,ip,1].real:+.6f} {-tensor[iw,ip,2].real:+.6f}\n")
  elif(PertType == "OR_L"):
    #
    # Beta (Electric Dipole-Magnetic Dipole) Origin-Invariant Length Gauge
    #
    # Print LG beta tensor
    tensor[iw,:,:] /= -4*W
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] LG Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
    # Print alpha(L,V) tensor   
    alpha_mix[iw,:,:] /= -2*W
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Alpha(L,V) [DipE-DipE] Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {alpha_mix[iw,ip,0].real:+.6f} {alpha_mix[iw,ip,1].real:+.6f} {alpha_mix[iw,ip,2].real:+.6f}\n")
    #
    # Compute LG(OI) transformation
    U, s, Vh = np.linalg.svd(alpha_mix[iw,:,:], full_matrices=True, compute_uv=True)
    if(np.linalg.det(U)<0): U = -U
    if(np.linalg.det(Vh)<0): Vh = -Vh
    tensor[iw,:,:] = np.einsum('ij,ik,lk->jl',np.conjugate(U),tensor[iw,:,:],np.conjugate(Vh),optimize=True)
    # Print LG(OI) beta tensor
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] LG(OI) Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
    #
    # Rotate LG(OI) back using the symmetric alpha(L,V) eigenvectors
    alpha_mix[iw,:,:] = (alpha_mix[iw,:,:] + np.conjugate(alpha_mix[iw,:,:]).T)/2
    # Order in decreasing order as SVD
    s, U0 = np.linalg.eig(alpha_mix[iw,:,:])
    desc_s = np.argsort(s)[::-1]
    Us = U0[:,desc_s]
    if(np.linalg.det(Us)<0): Us = -Us
    UU = np.einsum('ki,kj->ij',Us,U,optimize=True)
    for i in range(3):
      if (UU[i,i]<0): Us[:,i] = -Us[:,i]
    tensor[iw,:,:] = np.einsum('ji,ik,lk->jl',Us,tensor[iw,:,:],np.conjugate(Us),optimize=True)
    # Print rotated LG(OI) beta tensor
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Rotated Beta [DipE-DipM] LG(OI) Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
  elif(PertType == "OR_V" and iw>0):
    #
    # Beta (Electric Dipole-Magnetic Dipole) Modificed Velocity Gauge
    #
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] VG Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
    # For velocity gauge tensors, remove static limit before printing
    tensor[iw,:,:] -= tensor[0,:,:]
    # Then divide by frequency squared
    tensor[iw,:,:] /= -4*W**2
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] MVG Polarizability in a.u. for W = {W:.6f} a.u.\n")
      writer.write(f" Static limit removed\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
  elif(PertType == "OR_V" and iw==0):
    #
    # Beta (Electric Dipole-Magnetic Dipole) Modificed Velocity Gauge
    # Unphysical static limit
    #
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] VG (Unphysical) Static Polarizability in a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {-(tensor[iw,ip,0]/4).real:+.6f} {-(tensor[iw,ip,1]/4).real:+.6f} {-(tensor[iw,ip,2]/4).real:+.6f}\n")
  elif(PertType == "FullOR_V" and iw>0):
    #
    # Full OR Modificed Velocity Gauge
    # Beta (Electric Dipole-Magnetic Dipole) +  
    # A (Electric Dipole-Electric Quadrupole)  
    #
    # For velocity gauge tensors, remove static limit before printing
    tensor[iw,:,:] += tensor[0,:,:]
    tensorDQ[iw,:,:] += tensorDQ[0,:,:]
    # Beta 
    # Divide by frequency squared
    tensor[iw,:,:] /= -4*W**2
    # Symmetrize beta tensor
    tensor[iw,:,:] = (tensor[iw,:,:]+tensor[iw,:,:].T)/2
    trace = np.trace(tensor[iw,:,:])
    np.fill_diagonal(tensor[iw,:,:],np.diag(tensor[iw,:,:])-trace)
    tensor[iw,:,:] /= 2
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Beta contribution to full OR tensor MVG in a.u. for W = {W:.6f} a.u.\n")
      writer.write(f" Static limit removed\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
    # A 
    # Symmetrize A tensor
    # Indexing note: remember that Python indices start at 0, not 1
    DQ = np.zeros((3,3),dtype=tensor.dtype)
    DQ[0,0] = tensorDQ[iw,1,4]-tensorDQ[iw,2,3]
    DQ[1,0] = (tensorDQ[iw,1,5]-tensorDQ[iw,2,1]+tensorDQ[iw,2,0]-tensorDQ[iw,0,4])/2
    DQ[2,0] = (tensorDQ[iw,1,2]-tensorDQ[iw,2,5]+tensorDQ[iw,0,3]-tensorDQ[iw,1,0])/2
    DQ[1,1] = tensorDQ[iw,2,3]-tensorDQ[iw,0,5]
    DQ[2,1] = (tensorDQ[iw,2,4]-tensorDQ[iw,0,2]+tensorDQ[iw,0,1]-tensorDQ[iw,1,3])/2
    DQ[2,2] = tensorDQ[iw,0,5]-tensorDQ[iw,1,4]
    DQ[0,1] = DQ[1,0] 
    DQ[0,2] = DQ[2,0] 
    DQ[1,2] = DQ[2,1] 
    # Divide by frequency squared
    DQ /= -8*W**2
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n A contribution to full OR tensor MVG in a.u. for W = {W:.6f} a.u.\n")
      writer.write(f" Static limit removed\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {DQ[ip,0].real:+.6f} {DQ[ip,1].real:+.6f} {DQ[ip,2].real:+.6f}\n")
    # Full tensor
    tensor[iw,:,:] += DQ
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n B (beta + A) tensor MVG in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
  elif(PertType == "FullOR_V" and iw==0):
    #
    # Full OR Modificed Velocity Gauge
    # Beta (Electric Dipole-Magnetic Dipole) +  
    # A (Electric Dipole-Electric Quadrupole)  
    # Unphysical static limit
    #
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] VG (Unphysical) Static Polarizability in a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {-(tensor[iw,ip,0]/4).real:+.6f} {-(tensor[iw,ip,1]/4).real:+.6f} {-(tensor[iw,ip,2]/4).real:+.6f}\n")
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n A [DipE-DipM] VG (Unphysical) Static Polarizability in a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {-(tensorDQ[iw,ip,0]/4).real:+.6f} {-(tensorDQ[iw,ip,1]/4).real:+.6f} {-(tensorDQ[iw,ip,2]/4).real:+.6f} {-(tensorDQ[iw,ip,3]/4).real:+.6f} {-(tensorDQ[iw,ip,4]/4).real:+.6f} {-(tensorDQ[iw,ip,5]/4).real:+.6f}\n")
  elif(PertType == "FullOR_L"):
    #
    # Full OR Origin-Invariant Length Gauge LG(OI)
    # Beta (Electric Dipole-Magnetic Dipole) +  
    # A (Electric Dipole-Electric Quadrupole)  
    #
    # Compute and print regular LG tensors first 
    # # Beta 
    # # Divide by frequency
    tensor[iw,:,:] /= -4*W
    # Symmetrize beta tensor
    tensor_lg = np.zeros((3,3),dtype=tensor.dtype)
    tensor_lg = (tensor[iw,:,:]+tensor[iw,:,:].T)/2
    trace = np.trace(tensor_lg)
    np.fill_diagonal(tensor_lg,np.diag(tensor_lg)-trace)
    tensor_lg /= 2
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Beta contribution to full OR tensor LG in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor_lg[ip,0].real:+.6f} {tensor_lg[ip,1].real:+.6f} {tensor_lg[ip,2].real:+.6f}\n")
    # A 
    # Symmetrize A tensor
    # Indexing note: remember that Python indices start at 0, not 1
    DQ = np.zeros((3,3),dtype=tensor.dtype)
    DQ[0,0] = tensorDQ[iw,1,4]-tensorDQ[iw,2,3]
    DQ[1,0] = (tensorDQ[iw,1,5]-tensorDQ[iw,2,1]+tensorDQ[iw,2,0]-tensorDQ[iw,0,4])/2
    DQ[2,0] = (tensorDQ[iw,1,2]-tensorDQ[iw,2,5]+tensorDQ[iw,0,3]-tensorDQ[iw,1,0])/2
    DQ[1,1] = tensorDQ[iw,2,3]-tensorDQ[iw,0,5]
    DQ[2,1] = (tensorDQ[iw,2,4]-tensorDQ[iw,0,2]+tensorDQ[iw,0,1]-tensorDQ[iw,1,3])/2
    DQ[2,2] = tensorDQ[iw,0,5]-tensorDQ[iw,1,4]
    DQ[0,1] = DQ[1,0] 
    DQ[0,2] = DQ[2,0] 
    DQ[1,2] = DQ[2,1] 
    # Divide by frequency
    DQ /= -8*W
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n A contribution to full OR tensor LG in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {DQ[ip,0].real:+.6f} {DQ[ip,1].real:+.6f} {DQ[ip,2].real:+.6f}\n")
    # Full tensor
    tensor_lg += DQ
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n B (beta + A) tensor LG in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor_lg[ip,0].real:+.6f} {tensor_lg[ip,1].real:+.6f} {tensor_lg[ip,2].real:+.6f}\n")
    del tensor_lg
    #
    # Now do LG(OI)
    # Print alpha(L,V) tensor   
    alpha_mix[iw,:,:] /= -2*W
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Alpha(L,V) [DipE-DipE] Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {alpha_mix[iw,ip,0].real:+.6f} {alpha_mix[iw,ip,1].real:+.6f} {alpha_mix[iw,ip,2].real:+.6f}\n")
    #
    # Compute LG(OI) transformation
    U, s, Vh = np.linalg.svd(alpha_mix.real[iw,:,:], full_matrices=True, compute_uv=True)
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n U\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {U[ip,0]:+.6f} {U[ip,1]:+.6f} {U[ip,2]:+.6f}\n")
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Vh \n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {Vh[0,ip]:+.6f} {Vh[1,ip]:+.6f} {Vh[2,ip]:+.6f}\n")
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n sigma \n")
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f" {ip+1} {s[0]:+.6f} {s[1]:+.6f} {s[2]:+.6f}\n")
    if(np.linalg.det(U)<0): U = -U
    if(np.linalg.det(Vh)<0): Vh = -Vh
    #
    # Rotate LG(OI) back using the symmetric alpha(L,V) eigenvectors
    alpha_mix[iw,:,:] = (alpha_mix[iw,:,:] + np.conjugate(alpha_mix[iw,:,:]).T)/2
    # Order in decreasing order as SVD
    s, U0 = np.linalg.eig(alpha_mix.real[iw,:,:])
    desc_s = np.argsort(s)[::-1]
    Us = U0[:,desc_s]
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Us for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {Us[ip,0]} {Us[ip,1]} {Us[ip,2]}\n")
    if(np.linalg.det(Us)<0): Us = -Us
    UU = np.einsum('ki,kj->ij',Us,U,optimize=True)
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n UU for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {UU[ip,0]:+.6f} {UU[ip,1]:+.6f} {UU[ip,2]:+.6f}\n")
    for i in range(3):
      if (UU[i,i]<0): Us[:,i] = -Us[:,i]
    # transform Beta
    tensor[iw,:,:] = np.einsum('ij,ik,lk->jl',np.conjugate(U),tensor[iw,:,:],np.conjugate(Vh),optimize=True)
    # transform Beta back
    tensor[iw,:,:] = np.einsum('ji,ik,lk->jl',Us,tensor[iw,:,:],np.conjugate(Us),optimize=True)
    # transform A
    # We need to first expand it to a 3x3x3 tensor, then transform it,
    # and finally contract it back to 3x6 form
    tensorDQ3 = np.zeros((3,3,3),dtype=tensor.dtype)
    tensorDQ3[0,0,0] = tensorDQ[iw,0,0]
    tensorDQ3[1,0,0] = tensorDQ[iw,1,0]
    tensorDQ3[2,0,0] = tensorDQ[iw,2,0]
    tensorDQ3[0,1,1] = tensorDQ[iw,0,1]
    tensorDQ3[1,1,1] = tensorDQ[iw,1,1]
    tensorDQ3[2,1,1] = tensorDQ[iw,2,1]
    tensorDQ3[0,2,2] = tensorDQ[iw,0,2]
    tensorDQ3[1,2,2] = tensorDQ[iw,1,2]
    tensorDQ3[2,2,2] = tensorDQ[iw,2,2]
    tensorDQ3[0,0,1] = tensorDQ[iw,0,3]
    tensorDQ3[1,0,1] = tensorDQ[iw,1,3]
    tensorDQ3[2,0,1] = tensorDQ[iw,2,3]
    tensorDQ3[0,0,2] = tensorDQ[iw,0,4]
    tensorDQ3[1,0,2] = tensorDQ[iw,1,4]
    tensorDQ3[2,0,2] = tensorDQ[iw,2,4]
    tensorDQ3[0,1,2] = tensorDQ[iw,0,5]
    tensorDQ3[1,1,2] = tensorDQ[iw,1,5]
    tensorDQ3[2,1,2] = tensorDQ[iw,2,5]
    tensorDQ3[0,1,0] = tensorDQ3[0,0,1]
    tensorDQ3[1,1,0] = tensorDQ3[1,0,1]
    tensorDQ3[2,1,0] = tensorDQ3[2,0,1]
    tensorDQ3[0,2,0] = tensorDQ3[0,0,2]
    tensorDQ3[1,2,0] = tensorDQ3[1,0,2]
    tensorDQ3[2,2,0] = tensorDQ3[2,0,2]
    tensorDQ3[0,2,1] = tensorDQ3[0,1,2]
    tensorDQ3[1,2,1] = tensorDQ3[1,1,2]
    tensorDQ3[2,2,1] = tensorDQ3[2,1,2]
    tensorDQ3 = np.einsum('ij,ikm,lk,nm->jln',np.conjugate(U),tensorDQ3,np.conjugate(Vh),np.conjugate(Vh),optimize=True)
    # transform A back
    tensorDQ3 = np.einsum('ji,ikm,lk,nm->jln',Us,tensorDQ3,np.conjugate(Us),np.conjugate(Us),optimize=True)
    tensorDQ[iw,0,0] = tensorDQ3[0,0,0]
    tensorDQ[iw,1,0] = tensorDQ3[1,0,0]
    tensorDQ[iw,2,0] = tensorDQ3[2,0,0]
    tensorDQ[iw,0,1] = tensorDQ3[0,1,1]
    tensorDQ[iw,1,1] = tensorDQ3[1,1,1]
    tensorDQ[iw,2,1] = tensorDQ3[2,1,1]
    tensorDQ[iw,0,2] = tensorDQ3[0,2,2]
    tensorDQ[iw,1,2] = tensorDQ3[1,2,2]
    tensorDQ[iw,2,2] = tensorDQ3[2,2,2]
    tensorDQ[iw,0,3] = tensorDQ3[0,0,1]
    tensorDQ[iw,1,3] = tensorDQ3[1,0,1]
    tensorDQ[iw,2,3] = tensorDQ3[2,0,1]
    tensorDQ[iw,0,4] = tensorDQ3[0,0,2]
    tensorDQ[iw,1,4] = tensorDQ3[1,0,2]
    tensorDQ[iw,2,4] = tensorDQ3[2,0,2]
    tensorDQ[iw,0,5] = tensorDQ3[0,1,2]
    tensorDQ[iw,1,5] = tensorDQ3[1,1,2]
    tensorDQ[iw,2,5] = tensorDQ3[2,1,2]
    del tensorDQ3
    # Symmetrize beta tensor
    tensor[iw,:,:] = (tensor[iw,:,:]+tensor[iw,:,:].T)/2
    trace = np.trace(tensor[iw,:,:])
    np.fill_diagonal(tensor[iw,:,:],np.diag(tensor[iw,:,:])-trace)
    tensor[iw,:,:] /= 2
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n Beta contribution to full OR tensor LG(OI) in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
    # A 
    # Symmetrize A tensor
    # Indexing note: remember that Python indices start at 0, not 1
    DQ = np.zeros((3,3),dtype=tensor.dtype)
    DQ[0,0] = tensorDQ[iw,1,4]-tensorDQ[iw,2,3]
    DQ[1,0] = (tensorDQ[iw,1,5]-tensorDQ[iw,2,1]+tensorDQ[iw,2,0]-tensorDQ[iw,0,4])/2
    DQ[2,0] = (tensorDQ[iw,1,2]-tensorDQ[iw,2,5]+tensorDQ[iw,0,3]-tensorDQ[iw,1,0])/2
    DQ[1,1] = tensorDQ[iw,2,3]-tensorDQ[iw,0,5]
    DQ[2,1] = (tensorDQ[iw,2,4]-tensorDQ[iw,0,2]+tensorDQ[iw,0,1]-tensorDQ[iw,1,3])/2
    DQ[2,2] = tensorDQ[iw,0,5]-tensorDQ[iw,1,4]
    DQ[0,1] = DQ[1,0] 
    DQ[0,2] = DQ[2,0] 
    DQ[1,2] = DQ[2,1] 
    # Divide by frequency squared
    DQ /= -8*W
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n A contribution to full OR tensor LG(OI) in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {DQ[ip,0].real:+.6f} {DQ[ip,1].real:+.6f} {DQ[ip,2].real:+.6f}\n")
    # Full tensor
    tensor[iw,:,:] += DQ
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"\n B (beta + A) tensor LG(OI) in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
  #
  if(PertType == "OR_L" or PertType == "OR_V" or
     PertType == "FullOR_L" or PertType == "FullOR_V" ):
    # Print conversion factors for OR tensor
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f" Conversion factors\n a.u. --> deg/[dm (g/ml)]: {fact1:+.6f} \n a.u. --> deg L /(dm mol): {fact2:+.6f}\n")
  #
  return
