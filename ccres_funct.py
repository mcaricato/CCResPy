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
def DEk(T, NOrb2k, OrbE):
  # This is used to compute the U matrix in dC/dk = UC
  # Orbital energies are assumed to be real and stored as NOrb*2*Nkp
  if T==1:
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
def denom(T, O2, V2, kp, Fock, W):
  if T==1:
    NB2 = O2+V2
    if(kp):
      # PBC
      Nkp = len(kp)
      O2k = O2*Nkp
      V2k = V2*Nkp
      kp2 = kp
    else:
      # Molecular
      Nkp = 1
      O2k = O2
      V2k = V2
      kp2 = np.zeros((1))
    pi2 = round(2*np.pi,10)
    D1 = np.ones((Nkp,O2,Nkp,V2),dtype=Fock.dtype)
    D2 = np.ones((Nkp,O2,Nkp,O2,Nkp,V2,Nkp,V2),dtype=Fock.dtype)
    Fock = Fock.reshape((Nkp,NB2,Nkp,NB2))
    # D1 denominator
    NksumS = 0
    for n in range(Nkp):
      for k in range(Nkp):
        kn = kp2[n]
        kk = kp2[k]
        ktot = round(kn-kk,10)
        if(abs(ktot) < 1e-8 or abs(ktot%pi2) < 1e-8): 
          NksumS += 1
          for a in range(V2):
            for i in range(O2):
              D1[n,i,k,a]=Fock[n,i,n,i]-Fock[k,a+O2,k,a+O2] - W
    D1 = D1.reshape((O2k,V2k))
    if(NksumS != Nkp):
      print(f"Issue with k-point count for singles denominator: {NksumS} != {Nkp} ")
      exit()
    # D2 denominator
    NksumD = 0
    for n in range(Nkp):
      for k in range(Nkp):
        for h in range(Nkp):
          for g in range(Nkp):
            # IJAB          
            kn = kp2[n]
            kh = kp2[h]
            kk = kp2[k]
            kg = kp2[g]
            ktot = round(kn-kh+kk-kg,10)
            if(abs(ktot) < 1e-8 or abs(ktot%pi2) < 1e-8): 
              NksumD += 1
              for i in range(O2):
                deni = Fock[n,i,n,i] 
                for j in range(O2):
                  denj = deni + Fock[k,j,k,j]
                  for a in range(V2):
                    dena = denj - Fock[h,a+O2,h,a+O2]
                    for b in range(V2):
                      D2[n,i,k,j,h,a,g,b] = dena - Fock[g,b+O2,g,b+O2] - W
    if(NksumD != Nkp*Nkp*Nkp):
      print(f"Issue with k-point count for singles denominator: {NksumD} != {Nkp*Nkp*Nkp} ")
      exit()
    D2 = D2.reshape((O2k,O2k,V2k,V2k))
    Fock = Fock.reshape((Nkp*NB2,Nkp*NB2))
    return D1, D2

##########################################################################
# Wrapper routine for iterative solution of CCSD amplitude equations
##########################################################################
def AmpIt(AmpType,molecule,scratch,O,V,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,
          tau,F_ae,F_mi,F_me,rhs1,rhs2,D1,D2,t1,t2,l1,l2,tx1,tx2,ipbc):
# def AmpIt(AmpType,molecule,scratch,O,V,Nkp,MaxIt,ThrE,ThrA,scfE,Fock,IJKL,
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
  # st1 = []
  # st2 = []
  st1 = np.lib.format.open_memmap(f"{scratch}/{molecule}-DIISa1.npy",
                                  mode='w+',shape=(MaxD,*t1.shape),
                                  dtype=Fock.dtype) 
  st2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-DIISa2.npy",
                                  mode='w+',shape=(MaxD,*t2.shape),
                                  dtype=Fock.dtype) 
  # Start loop
  start0=time.time()
  with open(f"{molecule}.txt","a") as writer:
    writer.write(f"Iter.  DIIS     DE-{AmpType}(au)    Delta-DE(au)    Time(s)\n")
  while not_conver and N< MaxIt:
    start = time.time()
    N +=1
    E_Corr1 = E_Corr2
    if(AmpType == "T"):
      # Ground state T amplitudes
      if(N==1):
        # Initialize DIIS amplitudes with guess
        # st1 = []
        # st2 = []
        # st1.append(t1.reshape(np.size(t1)))
        # st2.append(t2.reshape(np.size(t2)))
        # np.save(f"{scratch}/{molecule}-DIISa1",st1)
        # np.save(f"{scratch}/{molecule}-DIISa2",st2)
        st1[0,:,:] = np.copy(t1)
        st2[0,:,:,:,:] = np.copy(t2)
        del st1, st2
      # Calculate intermediates
      tau_tilde = tau_tildeEq(1, Nkp, t1, t2)
      tau = tauEq(1, Nkp, t1, t2)
      F_ae,F_mi,F_me = T_interm(molecule,scratch,O,V,Nkp,Fock,t1,t2,
                                tau_tilde,tau)
      # F_ae,F_mi,F_me,W_mnij,W_mbej = T_interm(1,O,V,Nkp,Fock,t1,t2,
      #                                         IJKL,IABC,IJAB,IABJ,
      #                                         IJKA,tau_tilde,tau)
      # Amplitude iteration
      t1_f = t1Eq(molecule,scratch,O,V,Nkp,Fock,t1,t2,F_ae,F_mi,F_me,D1)
      t2_f = t2Eq(1,molecule,scratch,Nkp,t1,t2,
                  tau,F_ae,F_mi,F_me,D2)
      del F_ae,F_mi,F_me
      # t1_f = t1Eq(1,O,Nkp,Fock,t1,t2,IABC,IJKA,IABJ,F_ae,F_mi,F_me,D1)
      # t2_f = t2Eq(1,molecule,scratch,Nkp,t1,t2,IABC,IJAB,IJKA,IABJ,
      #             tau,F_ae,F_mi,F_me,W_mnij,W_mbej,D2)
      # del F_ae,F_mi,F_me,W_mnij,W_mbej
      # Check for convergence
      tau = tauEq(1, Nkp, t1_f, t2_f)
      IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
      not_conver,E_Corr2,t1,t2 = AmpConv(AmpType,O,Nkp,t1,t2,t1_f,t2_f,tau,
                                         Fock,D1,IJAB,ThrE,ThrA,E_Corr1)
      del t1_f, t2_f, IJAB
      # DIIS extrapolation
      t1, t2, DoDIIS = DIIS(scratch,molecule,O,V,N,MaxD,ThrA,RepD,t1,t2)
      a1 = t1
      a2 = t2
    elif (AmpType == "L"):
      # Ground state Lambda (or Z) amplitudes
      if(N==1):
        # Initialize DIIS amplitudes with guess
        # st1 = []
        # st2 = []
        # st1.append(l1.reshape(np.size(l1)))
        # st2.append(l2.reshape(np.size(l2)))
        # np.save(f"{scratch}/{molecule}-DIISa1",st1)
        # np.save(f"{scratch}/{molecule}-DIISa2",st2)
        st1[0,:,:] = np.copy(l1)
        st2[0,:,:,:,:] = np.copy(l2)
        del st1, st2
      # Calculate intermediates
      G_ae, G_mi = L_Interm(1,Nkp,t2,l2)
      # Amplitude iteration
      l1_f = l1Eq(1,molecule,scratch,Nkp,t1,l1,l2,F_ae,F_mi,F_me,G_ae,G_mi,D1)
      l2_f = l2Eq(1,molecule,scratch,Nkp,t1,l1,l2,F_ae,F_mi,F_me,G_ae,G_mi,D2)
      # l1_f = l1Eq(1,Nkp,t1,l1,l2,IJAB,IABC,IJKA,W_efam,W_iemn,W_mbej,F_ae,
      #             F_mi,F_me,G_ae,G_mi,D1)
      # l2_f = l2Eq(1,molecule,scratch,Nkp,t1,l1,l2,IABC,IJAB,IJKA,F_ae,F_mi,
      #             F_me,G_ae,G_mi,W_mnij,W_mbej,D2)
      tau_tilde = tauEq(1, Nkp, l1_f, l2_f)
      # Check for convergence
      IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
      not_conver, E_Corr2, l1, l2 = AmpConv(AmpType,O,Nkp,l1,l2,l1_f,l2_f,
                                            tau_tilde,Fock,D1,IJAB,ThrE,ThrA,
                                            E_Corr1)
      del l1_f, l2_f, G_ae, G_mi, IJAB 
      # DIIS extrapolation
      l1, l2, DoDIIS = DIIS(scratch,molecule,O,V,N,MaxD,ThrA,RepD,l1,l2)
      a1 = l1
      a2 = l2
    elif (AmpType == "Tx"):
      # Perturbed T amplitudes
      if(N==1):
        # Initialize DIIS amplitudes with guess
        # st1 = []
        # st2 = []
        # st1.append(tx1.reshape(np.size(tx1)))
        # st2.append(tx2.reshape(np.size(tx2)))
        # np.save(f"{scratch}/{molecule}-DIISa1",st1)
        # np.save(f"{scratch}/{molecule}-DIISa2",st2)
        st1[0,:,:] = np.copy(tx1)
        st2[0,:,:,:,:] = np.copy(tx2)
        del st1, st2
      # Calculate intermediates
      IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
      G_ae, G_mi = L_Interm(1,Nkp,IJAB,tx2)
      del IJAB
      # Amplitude iteration
      tx1_f = tx1Eq(1,molecule,scratch,Nkp,tx1,tx2,t1,F_ae,F_mi,F_me,G_ae,G_mi,D1)
      # tx1_f = tx1Eq(1,Nkp,tx1,tx2,t1,IABC,IJKA,W_mbej,F_ae,F_mi,F_me,G_ae,G_mi,D1)
      tx1_f -= rhs1/D1.real
      tx2_f = tx2Eq(1,molecule,scratch,Nkp,tx1,tx2,t1,t2,F_ae,F_mi,F_me,G_ae,G_mi,D2)
      # tx2_f = tx2Eq(1,molecule,scratch,Nkp,tx1,tx2,t1,t2,IABC,IJAB,IJKA,
      #               F_ae,F_mi,F_me,G_ae,G_mi,W_mnij,W_efam,W_iemn,
      #               W_mbej,D2)
      tx2_f -= rhs2/D2.real
      # Check for convergence
      not_conver, E_Corr2, tx1, tx2 = AmpConv(AmpType,O,Nkp,tx1,tx2,tx1_f,tx2_f,tau,
                                              Fock,rhs1,rhs2,ThrE,ThrA,E_Corr1)
      del tx1_f, tx2_f, G_ae, G_mi 
      # DIIS extrapolation
      tx1, tx2, DoDIIS = DIIS(scratch,molecule,O,V,N,MaxD,ThrA,RepD,tx1,tx2)
      a1 = tx1
      a2 = tx2
    else :
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"Amplitude type {AmpType} is not implemented. ")
      exit()
    textA = f"{N:4}     {DoDIIS}   {E_Corr2:+.10f}     {E_Corr2-E_Corr1:+.2e}       {time.time()-start:.2f}"
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"{textA}\n")
  if(not_conver):
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"{AmpType} amplitude equations convergence failure\n")
    exit()
  else:
    tot_mem, avlb_mem = mem_check()
    if(AmpType == "T"):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f"E(CCSD) = {scfE+E_Corr2:+.10f} au \n")      
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"{AmpType} amplitude equations converged in {time.time()-start0:.2f}s, AvlMem: {avlb_mem:.2f} GB\n\n")
  # Delete DIIS files
  os.system(f"rm {scratch}/{molecule}-DIISa1.npy")
  os.system(f"rm {scratch}/{molecule}-DIISa2.npy")
  return a1, a2

##########################################################################
# Evaluate convergence criteria and update amplitudes for amplitude
# iterations
##########################################################################
def AmpConv(AmpType,O,Nkp,a1,a2,a1_f,a2_f,tau,Fock,I1Int,I2Int,ThrE,ThrA,
            E_Corr1):
  DiffA1 = abs(np.max(abs(a1_f)-abs(a1)))
  DiffA2 = abs(np.max(abs(a2_f)-abs(a2)))
  a1RMSE = np.sqrt(np.sum((abs(a1_f)-abs(a1))**(2))/np.size(a1))
  a2RMSE = np.sqrt(np.sum((abs(a2_f)-abs(a2))**(2))/np.size(a2))
  a1 = np.copy(a1_f)
  a2 = np.copy(a2_f)
  NkpC = Nkp*Nkp*Nkp
  if(AmpType == "T"):
    # Here I2int should be the IJAB integrals
    E_Corr2 = E_CCSD(O,Nkp,Fock,a1,I2Int,tau)
  elif(AmpType == "L"):
    # Here I2int should be the IJAB integrals
    E_Corr2 = E_CCSD(O,Nkp,np.conjugate(Fock),a1,np.conjugate(I2Int),tau)
  elif (AmpType == "Tx"):
    # Here I1/2int should be the right hand side perturbations
    E_Corr2 = -0.25*np.einsum('ijab,ijab->',np.conjugate(I2Int),a2,optimize=True)/NkpC
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
def DIIS(scratch,molecule,O,V,Iter,MaxD,Thr,RepD,amp1,amp2):
  # Iter: current iteration
  # MaxD: size of the extrapolation space + 1 (for the constraint)
  # Thr: threshold on error to activate DIIS step
  # RepD: perform extrapolation every RepD iterations
  # amp1/2: amplitudes to extrapolate
  # st1/2: saved amplitudes from previous iterations
  # e_DIIS: save errors between iterations
  # B: DIIS matrix
  amp_type = amp1.dtype
  # sizeA1 = np.size(amp1)
  # sizeA2 = np.size(amp2)
  if(amp_type != amp2.dtype):
    print(f"Amplitude type mismatch: a1={amp_type} vs a2={amp2.dtype}")
    exit()
  ThrD = Thr/100
  st1 = np.load(f"{scratch}/{molecule}-DIISa1.npy",mmap_mode='r+')
  st2 = np.load(f"{scratch}/{molecule}-DIISa2.npy",mmap_mode='r+')
  if(Iter < MaxD):
    st1[Iter,:,:] = np.copy(amp1)
    st2[Iter,:,:,:,:] = np.copy(amp2)
    # print(f"Iter:{Iter}")
  else:
    # Shift amplitudes down by 1
    # (The explicit loop is as fast as np.roll but it uses less memory)
    for n in range(MaxD-1):
      st1[n,:,:] = np.copy(st1[n+1,:,:])
      st2[n,:,:,:,:] = np.copy(st2[n+1,:,:,:,:])
    st1[MaxD-1,:,:] = np.copy(amp1)
    st2[MaxD-1,:,:,:,:] = np.copy(amp2)
  # for n in range(MaxD):
  #   pamp1 = np.einsum('kl,kl->',np.conjugate(amp1),amp1,optimize=True)
  #   prd1 = np.einsum('kl,kl->',np.conjugate(st1[n,:,:]),st1[n,:,:],optimize=True)
  #   prd2 = np.einsum('klij,klij->',np.conjugate(st2[n,:,:,:,:]),st2[n,:,:,:,:],optimize=True)
  #   print(f"n:{n}, amp1: {pamp1.real}, st1: {prd1.real}, st2: {prd2.real:.2f} \n")
  # st1 = list(np.load(f"{scratch}/{molecule}-DIISa1.npy"))
  # st2 = list(np.load(f"{scratch}/{molecule}-DIISa2.npy"))
  # st1.append(amp1.reshape(np.size(amp1)))
  # st2.append(amp2.reshape(np.size(amp2)))
  # if len(st1)!= len(st2):
  #   print(f"String length mismatch in DIIS: {len(st1)}, {len(st2)}\n")
  #   exit()
  # if len(st1) > MaxD:
  #   # Remove the oldest information 
  #   del st1[0]
  #   del st2[0]
  # len1 = len(st1)
  # np.save(f"{scratch}/{molecule}-DIISa1",st1)
  # np.save(f"{scratch}/{molecule}-DIISa2",st2)
  del st1, st2
  DoDIIS = "F"
  if Iter%RepD==0:
#  if len1==MaxD and (Iter%RepD==0):
    # del amp1, amp2
    B = np.zeros((MaxD,MaxD),dtype=amp_type)
    # ev1 = np.zeros((MaxD-1,sizeA1),dtype=amp_type)
    ev1 = np.lib.format.open_memmap(f"{scratch}/{molecule}-ev1.npy",
                                    mode='w+',shape=(MaxD-1,*amp1.shape),
                                    dtype=amp_type) 
    st1 = np.load(f"{scratch}/{molecule}-DIISa1.npy",mmap_mode='r')
    # st1 = list(np.load(f"{scratch}/{molecule}-DIISa1.npy",mmap_mode='r'))
    for l in range(MaxD-1):
      # ev1[l,:] = np.array(st1[l+1]) - np.array(st1[l])
      ev1[l,:,:] = st1[l+1,:,:] - st1[l,:,:]
    del st1
    B[:MaxD-1,:MaxD-1] += np.einsum('ikl,jkl->ij',np.conjugate(ev1),ev1,optimize=True)
    del ev1
    # ev2 = np.zeros((MaxD-1,sizeA2),dtype=amp_type)
    ev2 = np.lib.format.open_memmap(f"{scratch}/{molecule}-ev2.npy",
                                    mode='w+',shape=(MaxD-1,*amp2.shape),
                                    dtype=amp_type) 
    st2 = np.load(f"{scratch}/{molecule}-DIISa2.npy",mmap_mode='r')
    # st2 = list(np.load(f"{scratch}/{molecule}-DIISa2.npy",mmap_mode='r'))
    for l in range(MaxD-1):
      ev2[l,:,:,:,:] = st2[l+1,:,:,:,:] - st2[l,:,:,:,:]
      # ev2[l,:] = np.array(st2[l+1]) - np.array(st2[l])
    del st2
    B[:MaxD-1,:MaxD-1] += np.einsum('iklmn,jklmn->ij',np.conjugate(ev2),ev2,optimize=True)
    del ev2
    os.system(f"rm {scratch}/{molecule}-ev?.npy")
    B[MaxD-1,:] = 1
    B[:,MaxD-1] = 1
    B[MaxD-1,MaxD-1] = 0
    rhs = np.zeros(MaxD)
    rhs[MaxD-1] = 1
    ETest = np.max(abs(B[:MaxD-1,:MaxD-1]))
    csol = np.linalg.solve(B,rhs)
    csum = np.sum(csol[:MaxD-1])
    # print(f"B:{B}\n csol: {csol}\n csum:{csum}\n")
    if(abs(csum-1)>ThrD):
      print(f"Issue with coefficients in DIIS: sum_C = {csum}\n")
      exit()
    # amp1 = np.zeros((sizeA1),dtype=amp_type)
    # amp2 = np.zeros((sizeA2),dtype=amp_type)
    # st1 = list(np.load(f"{scratch}/{molecule}-DIISa1.npy",mmap_mode='r'))
    # st2 = list(np.load(f"{scratch}/{molecule}-DIISa2.npy",mmap_mode='r'))
    st1 = np.load(f"{scratch}/{molecule}-DIISa1.npy",mmap_mode='r')
    st2 = np.load(f"{scratch}/{molecule}-DIISa2.npy",mmap_mode='r')
    amp1 = 0
    amp2 = 0
    for p in range(MaxD-1):
      amp1 += st1[p+1,:,:] * csol[p]
      amp2 += st2[p+1,:,:,:,:] * csol[p]
      # amp1 += np.array(st1[p+1]) * csol[p]
      # amp2 += np.array(st2[p+1]) * csol[p]
    del st1, st2
    # amp1 = np.reshape(amp1,((2*O),(2*V)))
    # amp2 = np.reshape(amp2,((2*O),(2*O),(2*V),(2*V)))
    DoDIIS = "T"
  # else:
  #   del st1, st2
  return amp1, amp2, DoDIIS

##########################################################################
# tau_tilde intermediate for CCSD T equations
##########################################################################
def tau_tildeEq(T, Nkp, t1, t2):
  if T==1:
    tau_tilde = np.copy(t2)
    tau_tilde += 0.5*np.einsum('ia,jb->ijab',t1,t1,optimize=True)*Nkp 
    tau_tilde -= 0.5*np.einsum('ib,ja->ijab',t1,t1,optimize=True)*Nkp
  return tau_tilde

##########################################################################
# tau intermediate for CCSD T equations
##########################################################################
def tauEq(T, Nkp, t1, t2):
  if T==1:
    tau = np.copy(t2)
    tau += np.einsum('ia,jb->ijab',t1,t1,optimize=True)*Nkp
    tau -= np.einsum('ib,ja->ijab',t1,t1,optimize=True)*Nkp
  return tau

##########################################################################
# F and W intermediates for CCSD T equations
##########################################################################
# def T_interm(T,O,V,Nkp,Fock,t1,t2,IJKL,IABC,IJAB,IABJ,IJKA,tau_tilde,tau):
def T_interm(molecule,scratch,O,V,Nkp,Fock,t1,t2,tau_tilde,tau):
  # O,V are assumed to be multiplied by Nkp in a PBC calculation
  O2=2*O
  V2=2*V
  NkpS = Nkp*Nkp
  IABC = np.load(f"{scratch}/{molecule}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{molecule}-IABJ.npy",mmap_mode='r')
  IJKL = np.load(f"{scratch}/{molecule}-IJKL.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{molecule}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{molecule}-Wmnij.npy",mmap_mode='r+')
  W_mbej = np.load(f"{scratch}/{molecule}-Wmbej.npy",mmap_mode='r+')
  # if T==1:
  # F_ae
  st_time = time.time()
  F_ae = np.zeros((V2, V2),dtype=Fock.dtype)
  F_ae += (1 - np.eye(V2)) * Fock[O2:, O2:] 
  F_ae -= 0.5 * np.einsum('me,ma->ae', Fock[:O2, O2:], t1, optimize=True)
  F_ae += np.einsum('mf,mafe->ae', t1, IABC, optimize=True)/Nkp
  F_ae -= 0.5 * np.einsum('mnaf,mnef->ae',tau_tilde,IJAB,optimize=True)/NkpS
  # F_mi
  F_mi = np.zeros((O2, O2),dtype=Fock.dtype)
  F_mi += (1 - np.eye(O2)) * Fock[:O2, :O2]
  F_mi += 0.5 * np.einsum('ie,me->mi', t1, Fock[:O2, O2:], optimize=True)
  F_mi -= np.einsum('ne,nmie->mi', t1, IJKA, optimize=True)/Nkp
  F_mi += 0.5 * np.einsum('inef,mnef->mi', tau_tilde, IJAB, optimize=True)/NkpS
  # F_me
  F_me = np.zeros((O2, V2),dtype=Fock.dtype)
  F_me = np.copy(Fock[:O2, O2:])
  F_me += np.einsum('nf,mnef->me', t1, IJAB, optimize=True)/Nkp
  # W_mnij
  W_mnij[:,:,:,:] = np.copy(IJKL)
  W_mnij += np.einsum('je,mnie->mnij', t1, IJKA, optimize=True)
  W_mnij -= np.einsum('ie,mnje->mnij', t1, IJKA, optimize=True)
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
  del IABC, IJAB, IABJ, IJKL, IJKA, W_mnij, W_mbej
  return F_ae, F_mi, F_me
  # return F_ae, F_mi, F_me, W_mnij, W_mbej

#########################################################################
# CCSD T1 amplitude equation
#########################################################################
def t1Eq(molecule,scratch,O,V,Nkp,Fock,t1,t2,F_ae,F_mi,F_me,D1):
# def t1Eq(T,O,Nkp,Fock,t1,t2,IABC,IJKA,IABJ,F_ae,F_mi,F_me,D1):
  IABC = np.load(f"{scratch}/{molecule}-IABC.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{molecule}-IABJ.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{molecule}-IJKA.npy",mmap_mode='r')
  NkpS = Nkp*Nkp
  O2=2*O
  # V2=2*V
  # O2u = int(O2/Nkp)
  # V2u = int(V2/Nkp)
  # t1_f = np.zeros((Nkp,Nkp,O2u,V2u),dtype=Fock.dtype)
  # Fock = Fock.reshape((Nkp,O2u+V2u,Nkp,O2u+V2u))
  # D1 = D1.reshape((Nkp,O2u,Nkp,V2u))
  # t1 = t1.reshape((Nkp,O2u,Nkp,V2u))
  # t1 = np.transpose(t1,axes=(0,2,1,3))
  # F_ae = F_ae.reshape((Nkp,V2u,Nkp,V2u))
  # F_ae = np.transpose(F_ae,axes=(0,2,1,3))
  # F_mi = F_mi.reshape((Nkp,O2u,Nkp,O2u))
  # F_mi = np.transpose(F_mi,axes=(0,2,1,3))
  # F_me = F_me.reshape((Nkp,O2u,Nkp,V2u))
  # F_me = np.transpose(F_me,axes=(0,2,1,3))
  # IABC = IABC.reshape((Nkp,O2u,Nkp,V2u,Nkp,V2u,Nkp,V2u))
  # IJKA = IJKA.reshape((Nkp,O2u,Nkp,O2u,Nkp,O2u,Nkp,V2u))
  # IABJ = IABJ.reshape((Nkp,O2u,Nkp,V2u,Nkp,V2u,Nkp,O2u))
  # t2 = t2.reshape((Nkp,O2u,Nkp,O2u,Nkp,V2u,Nkp,V2u))
  # for k in range(Nkp):
  #   t1_f[k,k,:,:] = np.copy(Fock[k,:O2u,k,O2u:])  
  #   t1_f[k,k,:,:] += np.einsum('ie,ae->ia',t1[k,k,:,:],F_ae[k,k,:,:],optimize=True)
  #   t1_f[k,k,:,:] -= np.einsum('ma,mi->ia',t1[k,k,:,:],F_mi[k,k,:,:],optimize=True)
  #   t1_f[k,k,:,:] += np.einsum('ihmahe,hhme->ia',t2[k,:,:,:,k,:,:,:],F_me, optimize=True)/Nkp
  #   t1_f[k,k,:,:] -= 0.5 * np.einsum('ihmleof,hmaleof->ia',t2[k,:,:,:,:,:,:,:],IABC[:,:,k,:,:,:,:,:],optimize=True)/NkpS
  #   t1_f[k,k,:,:] += 0.5 * np.einsum('hmlnaoe,lnhmioe->ia',t2[:,:,:,:,k,:,:,:],IJKA[:,:,:,:,k,:,:,:],optimize=True)/NkpS
  #   t1_f[k,k,:,:] += np.einsum('hhnf,hnahfi->ia',t1,IABJ[:,:,k,:,:,:,k,:],optimize=True)/Nkp
  #   t1_f[k,k,:,:] /= D1[k,:,k,:]
  # t1_f = np.transpose(t1_f,axes=(0,2,1,3))
  # t1_f = t1_f.reshape((O2,V2))
  # Fock = Fock.reshape((O2+V2,O2+V2))
  # D1 = D1.reshape((O2,V2))
  # t1 = np.transpose(t1,axes=(0,2,1,3))
  # t1 = t1.reshape((O2,V2))
  # F_ae = np.transpose(F_ae,axes=(0,2,1,3))
  # F_ae = F_ae.reshape((V2,V2))
  # F_mi = np.transpose(F_mi,axes=(0,2,1,3))
  # F_mi = F_mi.reshape((O2,O2))
  # F_me = np.transpose(F_me,axes=(0,2,1,3))
  # F_me = F_me.reshape((O2,V2))
  # t2 = t2.reshape((O2,O2,V2,V2))
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
# CCSD T2 amplitude equation
#########################################################################
def t2Eq(T,molecule,scratch,Nkp,t1,t2,tau,F_ae,
         F_mi,F_me,D2):
# def t2Eq(T,molecule,scratch,Nkp,t1,t2,IABC,IJAB,IJKA,IABJ,tau,F_ae,
#          F_mi,F_me,W_mnij,W_mbej,D2):
  IABC = np.load(f"{scratch}/{molecule}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{molecule}-IABJ.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{molecule}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{molecule}-Wmnij.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{molecule}-Wmbej.npy",mmap_mode='r')
  if T==1:
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
    X2 = np.einsum('imae,mbej->ijab',t2,W_mbej,optimize=True)/Nkp
    X2 += np.einsum('ma,mbij->ijab',t1,X1,optimize=True)
    t2_f += X2 - np.transpose(X2,axes=(1,0,2,3))
    t2_f -= np.transpose(X2,axes=(0,1,3,2))
    t2_f += np.transpose(X2,axes=(1,0,3,2))
    del X1, X2, W_mbej
    # tau terms
    if(f"{scratch}/{molecule}-ABCD.npy"):
      X1 = np.load(f"{scratch}/{molecule}-ABCD.npy",mmap_mode='r')
      t2_f += 0.5*np.einsum('ijef,abef->ijab',tau,X1,optimize=True)/Nkp
      del X1
    else:
      print(f"ABCD integrals file is missing in t2Eq\n")
      exit()
    t2_f += 0.5*np.einsum('mnab,mnij->ijab',tau,W_mnij,optimize=True)/Nkp
    del W_mnij
    # Add o3v3 work to avoid storing v4 intermediate (it also saves on
    # permutation work)
    X1 = np.einsum('ijef,mbef->ijmb',tau,IABC,optimize=True)/Nkp
    X2 = -0.5*np.einsum('ma,ijmb->ijab',t1,X1,optimize=True)
    t2_f += X2 - np.transpose(X2,axes=(0,1,3,2))
    del X1, X2, IABC
    t2_f /= D2    
  return t2_f

#########################################################################
# CCSD energy
#########################################################################
def E_CCSD(O,Nkp,Fock,t1,Int2,tau):
  O2 = 2*O
  NkpC = Nkp*Nkp*Nkp
  E_Corr2_1 = np.einsum('ia,ia->', t1, np.conjugate(Fock[:O2, O2:]),optimize=True)/Nkp
  E_Corr2_2 = 0.25 * np.einsum('ijab,ijab->', tau,Int2,optimize=True)/NkpC
  E_Corr2 = E_Corr2_1.real + E_Corr2_2.real
  return E_Corr2

#########################################################################
# Define constant intermediates for CCSD Lambda and response equations
#########################################################################
def Const_Interm(T,molecule,scratch,Nkp,t1,t2,tau,F_ae,F_mi,F_me):
# def Const_Interm(T,molecule,scratch,Nkp,t1,t2,tau,IJAB,IABJ,IJKA,IABC,
#                  F_ae,F_mi,F_me,W_mnij,W_mbej):
  IABC = np.load(f"{scratch}/{molecule}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
  IABJ = np.load(f"{scratch}/{molecule}-IABJ.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{molecule}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{molecule}-Wmnij.npy",mmap_mode='r')
  W_abef = np.load(f"{scratch}/{molecule}-Wabef.npy",mmap_mode='r+')
  W_efam = np.load(f"{scratch}/{molecule}-Wefam.npy",mmap_mode='r+')
  W_mbej = np.load(f"{scratch}/{molecule}-Wmbej.npy",mmap_mode='r+')
  W_iemn = np.load(f"{scratch}/{molecule}-Wiemn.npy",mmap_mode='r+')
  if T==1:
    # Remember that the contraction for Lambda is over the opposite
    # one or two indices (same for W_mnij)
    F_ae -= 0.5*np.einsum('ma,me->ae',t1,F_me,optimize=True)    
    # The sign of this terms is wrong in Gauss' paper
    F_mi += 0.5*np.einsum('me,ie->mi',F_me,t1,optimize=True)
    # Here we are forming the tilde-W_abef intermediate as in the
    # paper, at the cost of doing a o2v4 contraction once. The
    # tilde-W_nmij is already as in the paper, as we already doubled
    # the IJAB contribution for the t2 equations.
    # if(f"{scratch}/{molecule}-Wabef.npy"):
    #   W_abef = np.load(f"{scratch}/{molecule}-Wabef.npy",mmap_mode='r+')
    # else:
    #   print(f"Wabef intermediate file is missing in Const_Interm\n")
    #   exit()
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
    del W_mbej
    # These intermediates are new
    W_efam[:,:,:,:] = np.einsum('mnef,na->efam',t2,F_me,optimize=True)
    W_efam -= np.transpose(np.conjugate(IABC),axes=(2,3,1,0)) 
    W_efam += np.einsum('efag,mg->efam',W_abef,t1,optimize=True)
    # if(f"{scratch}/{molecule}-Wabef.npy"):
    #   np.save(f"{scratch}/{molecule}-Wabef",W_abef)
    #   del W_abef
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
    X1 = - np.einsum('ne,nfam->efam',t1,WW_mbej,optimize=True)
    X1 += np.einsum('nega,mnfg->efam',IABC,t2,optimize=True)/Nkp
    X2 = X1 - np.transpose(X1,axes=(1,0,2,3))
    W_efam += X2
    del X1,X2
    X1 = np.einsum('mf,iefn->iemn',t1,WW_mbej,optimize=True)
    X1 += np.einsum('iomf,noef->iemn',IJKA,t2,optimize=True)/Nkp
    X2 = X1 - np.transpose(X1,axes=(0,1,3,2))
    W_iemn += X2
    del X1, X2, WW_mbej, W_iemn
  del IABC, IJAB, IABJ, IJKA
  return F_ae, F_mi
  # return F_ae, F_mi, W_mbej, W_efam, W_iemn

#########################################################################
# Define changing intermediates for CCSD Lambda equations
#########################################################################
def L_Interm(T, Nkp, t2, l2):
  if T==1:
    NkpS = Nkp*Nkp
    G_ae = -0.5*np.einsum('mnaf,mnef->ae',l2,t2,optimize=True)/NkpS
    G_mi = 0.5*np.einsum('mnef,inef->mi',t2,l2,optimize=True)/NkpS
  return G_ae, G_mi

#########################################################################
# CCSD Lambda1 amplitude equation
#########################################################################
def l1Eq(T,molecule,scratch,Nkp,t1,l1,l2,F_ae,F_mi,F_me,G_ae,G_mi,D1):
# def l1Eq(T,Nkp,t1,l1,l2,IJAB,IABC,IJKA,W_efam,W_iemn,W_mbej,F_ae,F_mi,
#          F_me,G_ae,G_mi,D1):
  IABC = np.load(f"{scratch}/{molecule}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{molecule}-IJKA.npy",mmap_mode='r')
  W_efam = np.load(f"{scratch}/{molecule}-Wefam.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{molecule}-Wmbej.npy",mmap_mode='r')
  W_iemn = np.load(f"{scratch}/{molecule}-Wiemn.npy",mmap_mode='r')
  if T==1:
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
# CCSD Lambda2 amplitude equation
#########################################################################
def l2Eq(T,molecule,scratch,Nkp,t1,l1,l2,F_ae,F_mi,F_me,G_ae,
         G_mi,D2):
# def l2Eq(T,molecule,scratch,Nkp,t1,l1,l2,IABC,IJAB,IJKA,F_ae,F_mi,F_me,G_ae,
#          G_mi,W_mnij,W_mbej,D2):
  IABC = np.load(f"{scratch}/{molecule}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{molecule}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{molecule}-Wmnij.npy",mmap_mode='r')
  W_abef = np.load(f"{scratch}/{molecule}-Wabef.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{molecule}-Wmbej.npy",mmap_mode='r')
  if T==1:
    l2_f = np.copy(IJAB)
    # if(f"{scratch}/{molecule}-Wabef.npy"):
    #   X1 = np.load(f"{scratch}/{molecule}-Wabef.npy",mmap_mode='r')
    #   l2_f += 0.5*np.einsum('ijef,efab->ijab',l2,X1,optimize=True)/Nkp
    #   del X1
    # else:
    #   print(f"Wabef file is missing in l2Eq\n")
    #   exit()
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
# Form constant terms based on 1e perturbation X:
# <S|e^{-T}Xe^{T}|0> and <D|e^{-T}Xe^{T}|0>.
#########################################################################
def pert_rhs(T, PertSymm, Nkp, O2, V2, t1, t2, X_ij, X_ia, X_ab):
  # X is supposed to be in MO basis and already divided in oo, ov, and vv blocks
  if T==1:
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
# CCSD Tx1 (or EOM R1) amplitude equation
#########################################################################
def tx1Eq(T,molecule,scratch,Nkp,tx1,tx2,t1,F_ae,F_mi,F_me,
          G_ae,G_mi,D1):
# def tx1Eq(T,Nkp,tx1,tx2,t1,IABC,IJKA,W_mbej,F_ae,F_mi,F_me,G_ae,G_mi,D1):
  # Constant term needs to be added outside (as it's not in the EOM eqs.)
  # It requires getting G_ae, G_mi = L_Interm(T, Nkp, IJAB, tx2)
  IABC = np.load(f"{scratch}/{molecule}-IABC.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{molecule}-IJKA.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{molecule}-Wmbej.npy",mmap_mode='r')
  if T==1:
    NkpS = Nkp*Nkp
    tx1_f = np.einsum('ie,ae->ia',tx1,F_ae,optimize=True)
    tx1_f -= np.einsum('mi,ma->ia',F_mi,tx1,optimize=True)
    tx1_f += np.einsum('me,maei->ia',tx1,W_mbej,optimize=True)/Nkp
    tx1_f += np.einsum('imae,me->ia', tx2, F_me, optimize=True)/Nkp
    tx1_f -= 0.5 * np.einsum('imef,maef->ia', tx2, IABC,optimize=True)/NkpS
    tx1_f += 0.5 * np.einsum('nmea,nmie->ia', tx2, IJKA,optimize=True)/NkpS
    tx1_f += np.einsum('ib,ab->ia',t1,G_ae,optimize=True)
    tx1_f -= np.einsum('ji,ja->ia',G_mi,t1,optimize=True)
    tx1_f /= D1
  del IABC, IJKA, W_mbej
  return tx1_f

#########################################################################
# CCSD Tx2 (or EOM R2) amplitude equation
#########################################################################
def tx2Eq(T,molecule,scratch,Nkp,tx1,tx2,t1,t2,F_ae,F_mi,F_me,G_ae,G_mi,D2):
# def tx2Eq(T,molecule,scratch,Nkp,tx1,tx2,t1,t2,IABC,IJAB,IJKA,F_ae,F_mi,
#           F_me,G_ae,G_mi,W_mnij,W_efam,W_iemn,W_mbej,D2):
  # Constant term needs to be added outside (as it's not in the EOM eqs.)
  # It requires getting G_ae, G_mi = L_Interm(T, Nkp, IJAB, tx2)
  IABC = np.load(f"{scratch}/{molecule}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{molecule}-IJKA.npy",mmap_mode='r')
  W_mnij = np.load(f"{scratch}/{molecule}-Wmnij.npy",mmap_mode='r')
  W_abef = np.load(f"{scratch}/{molecule}-Wabef.npy",mmap_mode='r')
  W_efam = np.load(f"{scratch}/{molecule}-Wefam.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{molecule}-Wmbej.npy",mmap_mode='r')
  W_iemn = np.load(f"{scratch}/{molecule}-Wiemn.npy",mmap_mode='r')
  if T==1:
    NkpS = Nkp*Nkp
    # if(f"{scratch}/{molecule}-Wabef.npy"):
    #   X1 = np.load(f"{scratch}/{molecule}-Wabef.npy",mmap_mode='r')
    #   tx2_f = 0.5*np.einsum('ijef,abef->ijab',tx2,X1,optimize=True)/Nkp
    #   del X1
    # else:
    #   print(f"Wabef file is missing in tx2Eq\n")
    #   exit()
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
# CCSD Xi amplitudes for LR and EOM gradients
#########################################################################
def Xi(T,molecule,scratch,Nkp,O2,tx1,tx2,l1,l2,t1,F_ae,F_mi,F_me,D2):
# def Xi(T,Nkp,tx1,tx2,l1,l2,t1,IABC,IJAB,IJKA,F_ae,F_mi,F_me,W_mbej,D2):
  # L can be the ground or excited state Lambda amplitudes
  # Tx can be the LR Tx or the EOM R amplitudes
  IABC = np.load(f"{scratch}/{molecule}-IABC.npy",mmap_mode='r')
  IJAB = np.load(f"{scratch}/{molecule}-IJAB.npy",mmap_mode='r')
  IJKA = np.load(f"{scratch}/{molecule}-IJKA.npy",mmap_mode='r')
  W_mbej = np.load(f"{scratch}/{molecule}-Wmbej.npy",mmap_mode='r')
  tot_mem, avlb_mem = mem_check()
  o4gb = (O2**4)*8/(1024**3)
  if(tx1.dtype == complex): o4gb *= 2
  if(avlb_mem < 2*o4gb):
    Yimjk = np.lib.format.open_memmap(f"{scratch}/{molecule}-Yimjk.npy",
                                      mode='w+',shape=(O2,O2,O2,O2),
                                      dtype=tx1.dtype) 
    Zkijm = np.lib.format.open_memmap(f"{scratch}/{molecule}-Zkijm.npy",
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
    # term 2
    # Xi1 : -Lg(ijdb)t(md)R(kjcb)<mk||ac>
    X1 = np.einsum('kjcb,mkac->jmab',tx2,IJAB,optimize=True)/Nkp
    X2 = np.einsum('ijdb,md->ijmb',l2,t1,optimize=True)
    Xi1 -= np.einsum('ijmb,jmab->ia',X2,X1,optimize=True)/Nkp
    del X1, X2
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
    else:
      Yimjk += 0.25*np.einsum('ijcd,kmcd->ijkm',IJAB,tx2,optimize=True)/Nkp
    X2 = np.einsum('imjk,mc->icjk',Yimjk,t1,optimize=True)
    X2 -= 0.25*np.einsum('icbd,jkbd->icjk',IABC,tx2,optimize=True)/Nkp
    Xi1 += np.einsum('icjk,jkac->ia',X2,l2,optimize=True)/NkpS
    del X2
    X2 = IJKA + np.einsum('ijdc,md->ijmc',IJAB,t1,optimize=True)
    Yimjk -= np.einsum('ijmc,kc->ijkm',X2,tx1,optimize=True)
    Xi2 += np.einsum('ijkm,kmab->ijab',Yimjk,l2,optimize=True)/Nkp
    del Yimjk, X2
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
    Xi1 += np.einsum('kijm,jmka->ia',Zkijm,X2,optimize=True)/NkpS
    del X2
    X2 = np.einsum('ijcd,kc->ijkd',l2,tx1,optimize=True)
    Zkijm += np.einsum('ijkd,md->ijkm',X2,t1,optimize=True)
    Xi2 += np.einsum('ijkm,kmab->ijab',Zkijm,IJAB,optimize=True)/Nkp
    del Zkijm, X2
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
    # Term 12
    # Xi2 : -Lg(ijcd)R(kc)<kd||ab>
    X1 = np.einsum('ijcd,kc->ijkd',l2,tx1,optimize=True)
    Xi2 -= np.einsum('ijkd,kdab->ijab',X1,IABC,optimize=True)/Nkp
  del IABC, IJAB, IJKA
  if(avlb_mem < 2*o4gb):
    os.system(f"rm {scratch}/{molecule}-Yimjk.npy")
    os.system(f"rm {scratch}/{molecule}-Zkijm.npy")
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
  # kp: output array with k-point values in [-pi,0] range
  # l_list: output integer array with index over repeated cells: [0,+1,-1,+2,-2,...
  nmtpbc = ipbc[1]
  nrecip = ipbc[9]
  #Build l_list
  l_list = [0]
  for i in range(1,nmtpbc,2):
    l_list.append(-(i//2 + 1))
    l_list.append(i//2 + 1)
  #Build kp
  kp = []
  if nrecip == 1:
    kp = [0]
  elif nrecip % 2 == 0:
    for k in range(1, nrecip, 2):
      kp.append((np.pi * (k - nrecip) ) / nrecip)
    for k in range(1, nrecip, 2):
      kp.append((np.pi * (nrecip - k) ) / nrecip)
  elif nrecip % 2 != 0 and nrecip != 1:
    tmpn = np.ceil(nrecip/2) - 1
    for k in range(int(tmpn + 1)):
      kp.append((np.pi * (k - tmpn) ) / tmpn )
    for k in range(1,int(tmpn)):
      kp.append((np.pi * (tmpn - k) ) / tmpn )
  return kp, l_list

#########################################################################
# Function to perform the Fourier tranform of a 2-index array
#########################################################################
def fourier(FT,ipbc,MatIn,dk):
  # FT: "Dir" = R -> k
  #     "Inv" = k -> R
  # ipbc: integer array containing PBC info
  # MatIn : input array (real for Dir/complex for Inv)
  # MatOut : output array (complex for Dir/real for Inv)
  # dk: = F: regular MO(k) basis, = T: dS/dK in MO(k) basis
  kp, l_list = fill_kl(ipbc)
  co = np.einsum('k,l', kp, l_list, optimize=True)
  cof = np.cos(co) + 1j*np.sin(co)
  if(FT == "Dir"):
    if(dk):
      lcof = 1j*np.array(l_list)
      MatOut = np.einsum('kl,l,ln->kn',cof,lcof,MatIn,optimize=True)
    else:
      MatOut = np.einsum('kl,ln->kn',cof,MatIn,optimize=True)
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
def print_tensor(molecule,PertType,iw,W,tensor,tensorDQ,alpha_mix):
  # molecule: output file
  # PertType: tensor type
  # iw: current frequency for the printing
  # W: frequency value
  # tensor: tensor array
  # tensorDQ: temporary dipole-quadrupole tensor
  # alpha_mix: mixed-gauge dipole_L-dipole_V tensor
  #
  if(PertType == "DipE"):
    #
    # Electric Dipole-Electric Dipole Length Gauge
    #
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n DipE(LG)-DipE(LG) Polarizability in a.u. for W = {W:.6f} a.u.\n")
    # Symmetrize
    tensor[iw,:,:] = (tensor[iw,:,:] + tensor[iw,:,:].T)/2
    # np.fill_diagonal(tensor[iw,:,:],np.diag(tensor[iw,:,:])/2)
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
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
    # np.fill_diagonal(tensor[iw,:,:],np.diag(tensor[iw,:,:])/2)
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n DipE(MVG)-DipE(MVG) Polarizability in a.u. for W = {W:.6f} a.u.\n")
      writer.write(f" Static limit removed\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
  elif(PertType == "DipEV" and iw==0):
    #
    # Electric Dipole-Electric Dipole Modificed Velocity Gauge
    # Unphysical static limit
    #
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n DipE(MVG)-DipE(MVG) (Unphysical) Static Polarizability in a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {-tensor[iw,ip,0].real:+.6f} {-tensor[iw,ip,1].real:+.6f} {-tensor[iw,ip,2].real:+.6f}\n")
  elif(PertType == "OR_L"):
    #
    # Beta (Electric Dipole-Magnetic Dipole) Origin-Invariant Length Gauge
    #
    # Print LG beta tensor
    tensor[iw,:,:] /= -4*W
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] LG Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
    # Print alpha(L,V) tensor   
    alpha_mix[iw,:,:] /= -2*W
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Alpha(L,V) [DipE-DipE] Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {alpha_mix[iw,ip,0].real:+.6f} {alpha_mix[iw,ip,1].real:+.6f} {alpha_mix[iw,ip,2].real:+.6f}\n")
    #
    # Compute LG(OI) transformation
    U, s, Vh = np.linalg.svd(alpha_mix[iw,:,:], full_matrices=True, compute_uv=True)
    if(np.linalg.det(U)<0): U = -U
    if(np.linalg.det(Vh)<0): Vh = -Vh
    
    # pippo1 = np.einsum('ji,ik,lk->jl',np.conjugate(U),tensor[iw,:,:],np.conjugate(Vh),optimize=True)
    # pippo2 = np.einsum('ij,ik,kl->jl',np.conjugate(U),tensor[iw,:,:],np.conjugate(Vh),optimize=True)
    # pippo3 = np.einsum('ji,ik,kl->jl',np.conjugate(U),tensor[iw,:,:],np.conjugate(Vh),optimize=True)
    tensor[iw,:,:] = np.einsum('ij,ik,lk->jl',np.conjugate(U),tensor[iw,:,:],np.conjugate(Vh),optimize=True)
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n Alpha eigenvalues {s} a.u.\n")
    # del U, s, Vh
    # Print LG(OI) beta tensor
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] LG(OI) Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n pippo 1 \n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {pippo1[ip,0].real:+.6f} {pippo1[ip,1].real:+.6f} {pippo1[ip,2].real:+.6f}\n")
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n pippo 2 \n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {pippo2[ip,0].real:+.6f} {pippo2[ip,1].real:+.6f} {pippo2[ip,2].real:+.6f}\n")
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n pippo 3 \n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {pippo3[ip,0].real:+.6f} {pippo3[ip,1].real:+.6f} {pippo3[ip,2].real:+.6f}\n")
    #
    # Rotate LG(OI) back using the symmetric alpha(L,V) eigenvectors
    alpha_mix[iw,:,:] = (alpha_mix[iw,:,:] + np.conjugate(alpha_mix[iw,:,:]).T)/2
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n Alpha Symm for W = {W:.6f} a.u.\n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {alpha_mix[iw,ip,0].real:+.6f} {alpha_mix[iw,ip,1].real:+.6f} {alpha_mix[iw,ip,2].real:+.6f}\n")
    # Order in decreasing order as SVD
    s, U0 = np.linalg.eig(alpha_mix[iw,:,:])
    desc_s = np.argsort(s)[::-1]
    Us = U0[:,desc_s]
    if(np.linalg.det(Us)<0): Us = -Us
    UU = np.einsum('ki,kj->ij',Us,U,optimize=True)
    for i in range(3):
      if (UU[i,i]<0): Us[:,i] = -Us[:,i]
    # U = U0[:,desc_s]
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n Alpha Symm eigenvalues {s[desc_s]} a.u.\n {U}\n")
    # pippo1 = np.einsum('ji,ik,lk->jl',U,alpha_mix[iw,:,:],np.conjugate(U),optimize=True)
    # pippo2 = np.einsum('ij,ik,kl->jl',U,alpha_mix[iw,:,:],np.conjugate(U),optimize=True)
    # pippo3 = np.einsum('ji,ik,kl->jl',U,alpha_mix[iw,:,:],np.conjugate(U),optimize=True)
    # pippo4 = np.einsum('ij,ik,lk->jl',U,alpha_mix[iw,:,:],np.conjugate(U),optimize=True)
    # 
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n Alpha Diag for W = {W:.6f} a.u.\n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {pippo1[ip,0].real:+.6f} {pippo1[ip,1].real:+.6f} {pippo1[ip,2].real:+.6f}\n")
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n Alpha Diag for W = {W:.6f} a.u.\n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {pippo2[ip,0].real:+.6f} {pippo2[ip,1].real:+.6f} {pippo2[ip,2].real:+.6f}\n")
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n Alpha Diag for W = {W:.6f} a.u.\n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {pippo3[ip,0].real:+.6f} {pippo3[ip,1].real:+.6f} {pippo3[ip,2].real:+.6f}\n")
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n Alpha Diag for W = {W:.6f} a.u.\n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {pippo4[ip,0].real:+.6f} {pippo4[ip,1].real:+.6f} {pippo4[ip,2].real:+.6f}\n")
    # pippo1 = np.einsum('ij,ik,lk->jl',Us,tensor[iw,:,:],np.conjugate(Us),optimize=True)
    # pippo2 = np.einsum('ij,ik,kl->jl',Us,tensor[iw,:,:],np.conjugate(Us),optimize=True)
    # pippo3 = np.einsum('ji,ik,kl->jl',Us,tensor[iw,:,:],np.conjugate(Us),optimize=True)
    tensor[iw,:,:] = np.einsum('ji,ik,lk->jl',Us,tensor[iw,:,:],np.conjugate(Us),optimize=True)
    # Print rotated LG(OI) beta tensor
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Rotated Beta [DipE-DipM] LG(OI) Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n pippo 1 \n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {pippo1[ip,0].real:+.6f} {pippo1[ip,1].real:+.6f} {pippo1[ip,2].real:+.6f}\n")
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n pippo 2 \n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {pippo2[ip,0].real:+.6f} {pippo2[ip,1].real:+.6f} {pippo2[ip,2].real:+.6f}\n")
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n pippo 3 \n")
    # for ip in range(3):
    #   with open(f"{molecule}.txt","a") as writer:
    #     writer.write(f" {ip+1} {pippo3[ip,0].real:+.6f} {pippo3[ip,1].real:+.6f} {pippo3[ip,2].real:+.6f}\n")
  elif(PertType == "OR_V" and iw>0):
    #
    # Beta (Electric Dipole-Magnetic Dipole) Modificed Velocity Gauge
    #
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] VG Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
    # For velocity gauge tensors, remove static limit before printing
    tensor[iw,:,:] -= tensor[0,:,:]
    # Then divide by frequency squared
    tensor[iw,:,:] /= -4*W**2
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] MVG Polarizability in a.u. for W = {W:.6f} a.u.\n")
      writer.write(f" Static limit removed\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
  elif(PertType == "OR_V" and iw==0):
    #
    # Beta (Electric Dipole-Magnetic Dipole) Modificed Velocity Gauge
    # Unphysical static limit
    #
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] VG (Unphysical) Static Polarizability in a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {-(tensor[iw,ip,0]/4).real:+.6f} {-(tensor[iw,ip,1]/4).real:+.6f} {-(tensor[iw,ip,2]/4).real:+.6f}\n")
  elif(PertType == "FullOR_V" and iw>0):
    #
    # Full OR Modificed Velocity Gauge
    # Beta (Electric Dipole-Magnetic Dipole) +  
    # A (Electric Dipole-Electric Quadrupole)  
    #
    # For velocity gauge tensors, remove static limit before printing
    tensor[iw,:,:] -= tensor[0,:,:]
    tensorDQ[iw,:,:] -= tensorDQ[0,:,:]
    # # Beta 
    # # Divide by frequency squared
    tensor[iw,:,:] /= -4*W**2
    # Symmetrize beta tensor
    tensor[iw,:,:] = (tensor[iw,:,:]+tensor[iw,:,:].T)/2
    trace = np.trace(tensor[iw,:,:])
    np.fill_diagonal(tensor[iw,:,:],np.diag(tensor[iw,:,:])-trace)
    tensor[iw,:,:] /= 2
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Beta contribution to full OR tensor MVG in a.u. for W = {W:.6f} a.u.\n")
      writer.write(f" Static limit removed\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
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
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n A contribution to full OR tensor MVG in a.u. for W = {W:.6f} a.u.\n")
      writer.write(f" Static limit removed\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {DQ[ip,0].real:+.6f} {DQ[ip,1].real:+.6f} {DQ[ip,2].real:+.6f}\n")
    # Full tensor
    tensor[iw,:,:] += DQ
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n B (beta + A) tensor MVG in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
  elif(PertType == "FullOR_V" and iw==0):
    #
    # Full OR Modificed Velocity Gauge
    # Beta (Electric Dipole-Magnetic Dipole) +  
    # A (Electric Dipole-Electric Quadrupole)  
    # Unphysical static limit
    #
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Beta [DipE-DipM] VG (Unphysical) Static Polarizability in a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {-(tensor[iw,ip,0]/4).real:+.6f} {-(tensor[iw,ip,1]/4).real:+.6f} {-(tensor[iw,ip,2]/4).real:+.6f}\n")
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n A [DipE-DipM] VG (Unphysical) Static Polarizability in a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
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
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Beta contribution to full OR tensor LG in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
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
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n A contribution to full OR tensor LG in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {DQ[ip,0].real:+.6f} {DQ[ip,1].real:+.6f} {DQ[ip,2].real:+.6f}\n")
    # Full tensor
    tensor_lg += DQ
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n B (beta + A) tensor LG in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor_lg[ip,0].real:+.6f} {tensor_lg[ip,1].real:+.6f} {tensor_lg[ip,2].real:+.6f}\n")
    del tensor_lg
    #
    # Now do LG(OI)
    # Print alpha(L,V) tensor   
    alpha_mix[iw,:,:] /= -2*W
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Alpha(L,V) [DipE-DipE] Polarizability in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {alpha_mix[iw,ip,0].real:+.6f} {alpha_mix[iw,ip,1].real:+.6f} {alpha_mix[iw,ip,2].real:+.6f}\n")
    #
    # Compute LG(OI) transformation
    U, s, Vh = np.linalg.svd(alpha_mix.real[iw,:,:], full_matrices=True, compute_uv=True)
    # print(f"detU = {np.linalg.det(U)}\n")
    # print(f"detVh = {np.linalg.det(Vh)}\n")
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n U\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {U[ip,0]:+.6f} {U[ip,1]:+.6f} {U[ip,2]:+.6f}\n")
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Vh \n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {Vh[0,ip]:+.6f} {Vh[1,ip]:+.6f} {Vh[2,ip]:+.6f}\n")
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n sigma \n")
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f" {ip+1} {s[0]:+.6f} {s[1]:+.6f} {s[2]:+.6f}\n")
    # detU = np.linalg.det(U)
    # detVh = np.linalg.det(Vh)
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n detU = {detU}, detVh = {detVh}\n")
    if(np.linalg.det(U)<0): U = -U
    if(np.linalg.det(Vh)<0): Vh = -Vh
    # transform Beta
    tensor[iw,:,:] = np.einsum('ij,ik,lk->jl',np.conjugate(U),tensor[iw,:,:],np.conjugate(Vh),optimize=True)
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
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Beta contribution to full OR tensor LG(OI) in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
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
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n A contribution to full OR tensor LG(OI) in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {DQ[ip,0].real:+.6f} {DQ[ip,1].real:+.6f} {DQ[ip,2].real:+.6f}\n")
    # Full tensor
    tensor[iw,:,:] += DQ
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n B (beta + A) tensor LG(OI) in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")
    #
    # Rotate LG(OI) back using the symmetric alpha(L,V) eigenvectors
    alpha_mix[iw,:,:] = (alpha_mix[iw,:,:] + np.conjugate(alpha_mix[iw,:,:]).T)/2
    # Order in decreasing order as SVD
    s, U0 = np.linalg.eig(alpha_mix.real[iw,:,:])
    desc_s = np.argsort(s)[::-1]
    Us = U0[:,desc_s]
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Us for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {Us[ip,0]} {Us[ip,1]} {Us[ip,2]}\n")
    # print(f"detUs = {np.linalg.det(Us)}\n")
    # detUs = np.linalg.det(Us)
    # with open(f"{molecule}.txt","a") as writer:
    #   writer.write(f"\n detUs = {detUs}\n")
    if(np.linalg.det(Us)<0): Us = -Us
    UU = np.einsum('ki,kj->ij',Us,U,optimize=True)
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n UU for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {UU[ip,0]:+.6f} {UU[ip,1]:+.6f} {UU[ip,2]:+.6f}\n")
    for i in range(3):
      if (UU[i,i]<0): Us[:,i] = -Us[:,i]
    tensor[iw,:,:] = np.einsum('ji,ik,lk->jl',Us,tensor[iw,:,:],np.conjugate(Us),optimize=True)
    # Print rotated LG(OI) beta tensor
    with open(f"{molecule}.txt","a") as writer:
      writer.write(f"\n Rotated B tensor LG(OI) in a.u. for W = {W:.6f} a.u.\n")
    for ip in range(3):
      with open(f"{molecule}.txt","a") as writer:
        writer.write(f" {ip+1} {tensor[iw,ip,0].real:+.6f} {tensor[iw,ip,1].real:+.6f} {tensor[iw,ip,2].real:+.6f}\n")

    
  return
