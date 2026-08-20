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
import importlib
from pathlib import Path
import datetime
import resource, platform, tracemalloc
from scipy.constants import angstrom, physical_constants, h, c
np.set_printoptions(precision=16,threshold=sys.maxsize,floatmode='fixed')
from ccres_funct import fourier_coef, fourier, basis_tran, fill_kl, square_m, denom, DEk, mem_check, mol_mass, form_map_kp, momentum_cons
np.set_printoptions(precision=16,threshold=sys.maxsize,floatmode='fixed')
# # GAUOPEN path
# sys.path.insert(0, '/Volumes/gaussian/gdv_j30p/')
# from gauopen import QCBinAr as qcb
# from gauopen import QCOpMat as qco
# import gauopen.qcmio as qcmio

##########################################################################
# Read input file and initialize output file
##########################################################################
def input_parser(input_file):
  # Define default Settings
  ThrE0 = 1e-8     # Energy convervegence threshold (au)
  # ThrA0 = ThrE0*100 # Amplitude convergence threshold
  MaxIt0 = 1000    # Max number of iterations
  WLlist0 = [500]   # List of wavelengths (nm) of external field
  MaxD0 = 6        # Max size of DIIS subspace
  RepD0 = 5        # Iterations interval between DIIS extrpolations 
  PertType0 = f"DipE" # Linear response function
  Kstore0 = f"collective" # Use collective indices for k points
  # tv = np.array([4.0, 0.0, 0.0])
  FreezeCore0 = True # Whether to do frozen core
  memory0 = 100    # Memory limit (GB) for Linux platforms
  #
  # Read input file
  file_name = importlib.import_module(input_file)
  # Input baf name
  try:
    mol_inp = file_name.molecule
  except AttributeError: 
    print("No molecule specified for calculation")
    exit()
  files = Path(f"{mol_inp}_txts")
  if os.path.isdir(files):
    pass
  else:
    print("Files directory does not exist")
    exit()
  # Output file name
  try:
    mol_out = file_name.output_file
  except AttributeError:
    # if no name is specified, use same name as input
    mol_out = mol_inp
  # 2ERIs
  try:
    eri = file_name.eri
  except AttributeError:
    print("2 electron integrals file not specified")
    exit()
  erifile = Path(f"{eri}.baf")
  if erifile.is_file():
    pass
  else:
    print("2ERI file does not exist")
    exit()
  # scratch path
  try: 
    scratch = file_name.scratch
  except AttributeError:
    print("Scratch directory not specified")
    exit()
  directory = Path(f"{scratch}")
  if directory.is_dir():
    pass
  else:
    print("Scratch directory does not exist")
    exit()
  # Gauopen path
  try: 
    path_gauopen = file_name.path_gauopen
  except AttributeError:
    print("Gauopen directory not specified")
    exit()
  fileg = Path(f"{path_gauopen}")
  if os.path.isdir(fileg):
    pass
  else:
    print("Gauopen directory does not exist")
    exit()
  # Memory limit (GB) for Linux platforms
  try: 
    memory = file_name.memory
  except AttributeError:
    memory = memory0
  # Energy convergence threshold
  try:
    ThrE = file_name.ThrE
  except AttributeError:
    ThrE = ThrE0
  # Amplitude convergence threshold
  try:
    ThrA = file_name.ThrA
  except AttributeError:
    ThrA = ThrE*100
  # Max number of iterations
  try:
    MaxIt = file_name.MaxIt
  except AttributeError:
    MaxIt = MaxIt0
  # List of wavelengths (nm) of external field
  try:
    WLlist = file_name.WLlist
  except AttributeError:
    WLlist = WLlist0
  # Max size of DIIS subspace
  try:
    MaxD = file_name.MaxD
  except AttributeError:
    MaxD = MaxD0
  # Iterations interval between DIIS extrpolations 
  try:
    RepD = file_name.RepD
  except AttributeError:
    RepD = RepD0
  # Linear response function
  try:
    PertType = file_name.PertType
  except AttributeError:
    PertType = PertType0
  # How to store K points in tensor arrays:
  # collective (default) or compress (make use of momentum conservation)
  try:
    Kstore = file_name.Kstore
  except AttributeError:
    Kstore = Kstore0
  # Whether to do frozen core
  try:
    FreezeCore = file_name.FreezeCore
  except AttributeError:
    FreezeCore = FreezeCore0
  # Translation vector (Ang)
  try:
    tv = file_name.tv
  except AttributeError:
    tv = [0.0,0.0,0.0]
  #
  # Convert wavelengths (nm) to frequency (au)
  hartree = physical_constants['Hartree energy'][0]
  Wlist = []
  if(PertType == "DipEV" or PertType == "OR_V" or PertType == "FullOR_V"):
    Wlist.append(0.0)
  for i in range(len(WLlist)): 
    value = float(WLlist[i])
    freq = (h*c*1e9) / (value*hartree)
    Wlist.append(freq)
  #
  return ThrE, ThrA, MaxIt, Wlist, MaxD, RepD, PertType, tv, FreezeCore, memory, scratch, path_gauopen, eri, mol_inp, mol_out, Kstore

##########################################################################
# Initialize output file
##########################################################################
def Initialize(mol_out,memory,ThrE,MaxIt,Wlist): 
  #Clean previous outputs
  os.system(f"rm {mol_out}.txt")
  tot_mem, avlb_mem = mem_check()
  tracemalloc.start()
  current_date = datetime.date.today()
  current_time = datetime.datetime.now()
  with open(f"{mol_out}.txt","a") as writer: 
    writer.write(f"CCResPy PROGRAM \n")
    writer.write(current_date.strftime("%m/%d/%Y "))  
    writer.write(current_time.strftime("%H:%M:%S \n"))
    writer.write(f"Platform: {platform.system()} -- Python v{platform.python_version()} -- NumPy v{np.version.version}\n")
    writer.write(f"Total Memory: {tot_mem:.2f}GB, Available Memory: {avlb_mem:.2f}GB \n")
  if platform.system() == "Linux":
    mem_limit = memory*(1024**3)                         
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit)) 
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)            
    soft /= 1024**3                                                
    hard /= 1024**3                                                
    with open(f"{mol_out}.txt","a") as writer:                    
      writer.write(f"Soft Memory Limit: {soft:.2f}GB, Hard Memory Limit: {hard:.2f}GB \n") 
  with open(f"{mol_out}.txt","a") as writer:
    writer.write(f"\nEnergy convergence threshold: {ThrE:.1e} au -- Max N Iterations: {MaxIt}\n")
  with open(f"{mol_out}.txt","a") as writer:
    writer.write(f"\nField frequency (a.u.) / wavelength (nm):\n")
  hartree = physical_constants['Hartree energy'][0]
  WLlist = []
  for i in range(len(Wlist)): 
    value = float(Wlist[i])
    if(value==0):
      WLlist.append("static")
    else:
      wl = (h*c*1e9) / (value*hartree)
      WLlist.append(wl)
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f"{Wlist[i]:f} / {WLlist[i]}\n")
  return 

##########################################################################
#Get O, NB, SCF energy, MO coefficients, Orbital energies ################
##########################################################################
def getFort(mol_inp,mol_out,FreezeCore):
  # Read atoms list
  atoms_list = []
  with open(f"{mol_inp}_txts/atoms_list.txt","r") as reader:
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
  with open(f"{mol_inp}_txts/geometry.txt","r") as reader:
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
  with open(f"{mol_inp}_txts/occ.txt","r") as reader:
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
  with open(f"{mol_out}.txt","a") as writer:
    writer.write(f"\nGeometry (Ang):\n") 
    for a in range(len(atoms_list)):
      writer.write(f"{atoms_list_names[a]} {geometry[a,0]:+.8f} {geometry[a,1]:+.8f} {geometry[a,2]:+.8f} \n") 
    writer.write(f"\nMolecular Mass: {mol_weight:+.8f} g/mol\n")
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
  with open(f"{mol_inp}_txts/scf.txt","r") as reader:
    text=[]
    for line in reader:
      text.append(line.split())
  scfE=float(text[0][0])
  #
  # Read PBC info if available
  ipbc=[]
  k_weights=[]
  if(os.path.exists(f"{mol_inp}_txts/pbc_info.txt")):
    #
    # Read PBC integers
    with open(f"{mol_inp}_txts/pbc_info.txt","r") as reader:
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
    with open(f"{mol_inp}_txts/k_weights.txt","r") as reader:
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
  with open(f"{mol_inp}_txts/mocoef.txt","r") as reader:
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
    # PBC. Gaussian works with half of the first Brullouin zone
    # (FBZ) in every reciprocal direction. Here we expand the MO
    # coefficients over the entire FBZ.
    # Number of k points:
    #   nkpnt: number of unique k points in Gaussian
    #   Nkp: total number of k points used in CCResPy
    for i in range(len(text)):
      for j in range(len(text[i])):
        MOCoef.append(complex(text[i][j]))
    # Nkp, kp, l_list = fill_kl(ipbc)
    Nkp, _, _ = fill_kl(ipbc)
    npdir = ipbc[0]
    # nrecip = ipbc[9]
    shift = ipbc[11]
    nkpnt = ipbc[10]
    ndimk = ipbc[12:15]
    # Nkp = 1
    # for n in range(npdir): Nkp *= ndimk[n]
    # Nkp = ndimk[0]
    if(npdir > 1):
      # Read k point grid over the half FBZ from Gaussian
      posgrid = np.load(f"{mol_inp}_txts/grid-coord.npy")
      print(f"posgrid: {posgrid.shape}, {np.size(posgrid)}, {posgrid}\n")
      if (nkpnt != len(posgrid)):
        with open(f"{mol_out}.txt","a") as writer:
          writer.write(f"Inconsistency in the number of k points in getFort.\n")
        exit()
      # Build k point grid over the full FBZ 
      kgrid = []
      if(npdir == 2):
        # Nkp *= ndimk[1]
        lenx = ndimk[0]
        leny = ndimk[1]
        for ny in range(leny):
          for nx in range(lenx):
            kgrid.append([nx+1,ny+1])
      else:
        # Nkp *= ndimk[1]*ndimk[2]
        lenx = ndimk[0]
        leny = ndimk[1]
        lenz = ndimk[2]
        kgrid = []
        for nz in range(lenz):
          for ny in range(leny):
            for nx in range(lenx):
              kgrid.append([nx+1,ny+1,nz+1])
      print(f"kgrid: shape {np.array(kgrid).shape}, length {len(kgrid)}, size {np.size(kgrid)}, Nkp {Nkp}\n {kgrid}\n")
    # Now expand the MO coefficient across the entire FBZ
    MOCoef0 = np.array(MOCoef).reshape((nkpnt,NB*NB))
    MOCoef = np.zeros((Nkp,NB*NB),dtype=complex)
    if(shift == 0 ):
      # This grid starts at -pi and includes the Gamma point (the
      # latter is at the end of the Gaussian list, i.e. at the end of
      # half of the FBZ, but in the middle of the full FBZ)
      MOCoef[0,:] = np.copy(MOCoef0[0,:])
      if(npdir == 1):
        MOCoef[Nkp//2,:] = np.copy(MOCoef0[-1,:])
        for n in range(1,nkpnt-1):
          MOCoef[n,:] = np.copy(MOCoef0[n,:])
          MOCoef[Nkp-n,:] = np.copy(np.conjugate(MOCoef0[n,:]))
      elif(npdir == 2):
        lenx = ndimk[0]
        leny = ndimk[1]
        ngamma = (lenx)*(leny//2) + lenx//2
        MOCoef[ngamma,:] = np.copy(MOCoef0[-1,:])
        for n in range(1,nkpnt-1):
          indf1 = list(posgrid[n])
          if(indf1[1] == 1):
            indf2 = [lenx + 2 - indf1[0],1]
          elif(indf1[0] == 1):
            indf2 = [1,leny + 2 - indf1[1]]
          else:
            indf2 = [lenx + 2 - indf1[0], leny + 2 - indf1[1]]
          indff1 = kgrid.index(indf1)
          indff2 = kgrid.index(indf2)
          print(f"k-point: {n}, pos:{indf1}, {indf2}, final-ks: {indff1} {indff2}\n")
          MOCoef[indff1,:] = np.copy(MOCoef0[n,:])
          MOCoef[indff2,:] = np.copy(np.conjugate(MOCoef0[n,:]))
          print(f"MO[k1]: {MOCoef[indff1,1]}\n MO(k2): {MOCoef[indff2,1]}\n")
    else:
      # This grid is shifted and does not include the Gamma point and the edges
      if(npdir == 1):
        for n in range(nkpnt):
          MOCoef[n,:] = np.copy(MOCoef0[n,:])
          MOCoef[Nkp-1-n,:] = np.copy(np.conjugate(MOCoef0[n,:]))
      elif(npdir == 2):
        lenx = ndimk[0]
        leny = ndimk[1]
        for n in range(nkpnt):
          indf1 = list(posgrid[n])
          indf2 = [lenx + 1 - indf1[0],leny + 1 - indf1[1]]
          indff1 = kgrid.index(indf1)
          indff2 = kgrid.index(indf2)
          MOCoef[indff1,:] = np.copy(MOCoef0[n,:])
          MOCoef[indff2,:] = np.copy(np.conjugate(MOCoef0[n,:]))
    # if(nrecip % 2 == 0):
    #   MOCoef = np.array(MOCoef).reshape((Nkp//2,NB,NB))
    #   MORev = np.flip(MOCoef,axis=0)
    #   MOCoef = np.append(MOCoef,np.conjugate(MORev))
    #   del MORev
    #   # MOCoef = np.append(MOCoef,np.conjugate(MOCoef))
    # else:
    #   nMO = Nkp//2+1
    #   MOCoef = np.array(MOCoef).reshape((nMO,NB*NB))
    #   MORev = np.flip(MOCoef,axis=0)
    #   MOCoef = np.append(MOCoef,np.conjugate(MORev[1:-1,:]))
    #   del MORev
    #   # MOCoef = np.append(MOCoef,np.conjugate(MOCoef[1:-1,:].reverse))
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
def getOvl(mol_inp,O,V,NB,ipbc,basis,dk,dir_dk,MOCoef):
  # basis: "AO" or "MO"
  # dk: = F: regular MO(k) basis, = T: dS/dK in MO(k) basis
  #                               dir_dk: direction of k-derivative
  NOrb = O + V
  with open(f"{mol_inp}_txts/overlap.txt","r") as reader:
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
      Nkp, _, _ = fill_kl(ipbc)
      # Nkp = len(kp)
      Ovl_k_lt = fourier("Dir",ipbc,Ovl_r,dk,dir_dk)
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
# Routine the read the Overlap with Nkp storage
########################################################
def getOvl1k(mol_inp,O,V,NB,ipbc,basis,dk,dir_dk,MOCoef):
  # basis: "AO" or "MO"
  # dk: = F: regular MO(k) basis, = T: dS/dK in MO(k) basis
  #                               dir_dk: direction of k-derivative
  NOrb = O + V
  with open(f"{mol_inp}_txts/overlap.txt","r") as reader:
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
      Nkp, _, _ = fill_kl(ipbc)
      # Nkp = len(kp)
      Ovl_k_lt = fourier("Dir",ipbc,Ovl_r,dk,dir_dk)
      OvlA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Ovl_k_lt)
      Ovl = np.zeros((Nkp,NOrb*2,NOrb*2),dtype=complex)
      for k in range(Nkp):
        # Fill out the alpha and beta blocks
        # oa-oa
        Ovl[k,:O,:O] = OvlA[k,:O,:O]
        # ob-ob
        Ovl[k,O:2*O,O:2*O] = OvlA[k,:O,:O]
        # va-va
        Ovl[k,2*O:2*O+V,2*O:2*O+V] = OvlA[k,O:,O:]
        # vb-vb
        Ovl[k,2*O+V:,2*O+V:] = OvlA[k,O:,O:]
        # oa-va
        Ovl[k,:O,2*O:2*O+V] = OvlA[k,:O,O:]
        # va-oa
        Ovl[k,2*O:2*O+V,:O] = OvlA[k,O:,:O]
        # ob-vb
        Ovl[k,O:2*O,2*O+V:] = OvlA[k,:O,O:]
        # vb-ob
        Ovl[k,2*O+V:,O:2*O] = OvlA[k,O:,:O]
      del Ovl_r, Ovl_k_lt, OvlA
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
def getFock(mol_inp,O,V,NB,ipbc,basis,dk,dir_dk,MOCoef):
  # basis: "AO" or "MO"
  # dk: = F: regular MO(k) basis, = T: dS/dK in MO(k) basis
  #                               dir_dk: direction of k-derivative
  NOrb = O + V
  with open(f"{mol_inp}_txts/fock.txt","r") as reader:
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
      # Nkp, kp, l_list = fill_kl(ipbc)
      Nkp, _, _ = fill_kl(ipbc)
      # Nkp = len(kp)
      Fock_k_lt = fourier("Dir",ipbc,Fock_r,dk,dir_dk)
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
# Routine the read the Fock matrix with Nkp storage
########################################################
def getFock1k(mol_inp,O,V,NB,ipbc,basis,dk,dir_dk,MOCoef):
  # basis: "AO" or "MO"
  # dk: = F: regular MO(k) basis, = T: dS/dK in MO(k) basis
  #                               dir_dk: direction of k-derivative
  NOrb = O + V
  with open(f"{mol_inp}_txts/fock.txt","r") as reader:
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
      # Nkp, kp, l_list = fill_kl(ipbc)
      Nkp, _, _ = fill_kl(ipbc)
      # Nkp = len(kp)
      Fock_k_lt = fourier("Dir",ipbc,Fock_r,dk,dir_dk)
      FockA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Fock_k_lt)
      Fock = np.zeros((Nkp,NOrb*2,NOrb*2),dtype=complex)
      for k in range(Nkp):
        # Fill out the alpha and beta blocks
        # oa-oa
        Fock[k,:O,:O] = FockA[k,:O,:O]
        # ob-ob
        Fock[k,O:2*O,O:2*O] = FockA[k,:O,:O]
        # va-va
        Fock[k,2*O:2*O+V,2*O:2*O+V] = FockA[k,O:,O:]
        # vb-vb
        Fock[k,2*O+V:,2*O+V:] = FockA[k,O:,O:]
        # oa-va
        Fock[k,:O,2*O:2*O+V] = FockA[k,:O,O:]
        # va-oa
        Fock[k,2*O:2*O+V,:O] = FockA[k,O:,:O]
        # ob-vb
        Fock[k,O:2*O,2*O+V:] = FockA[k,:O,O:]
        # vb-ob
        Fock[k,2*O+V:,O:2*O] = FockA[k,O:,:O]
      del Fock_r, Fock_k_lt, FockA
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
def get2e(NB,ipbc,mol_out,eri_file,scratch,path_gauopen):
  # GAUOPEN path
  sys.path.insert(0,f"{path_gauopen}")
  # sys.path.insert(0, '/Volumes/gaussian/gdv_j30p/')
  from gauopen import QCBinAr as qcb
  from gauopen import BinArFile as bar
  # from gauopen import QCOpMat as qco
  import gauopen.qcmio as qcmio
  nmtpbc = 0
  NBX = NB
  #mol_int = sys.argv[2]
  if(ipbc):
    nmtpbc = ipbc[1]
    NBX = NB*nmtpbc
    NCMax = (nmtpbc-1)//2
  #  Nkp, kp, l_list = fill_kl(ipbc)
  # AOInt=np.zeros((NBX, NBX, NBX, NBX))
  # mol=sys.argv[1]
  # icount = 0
  # with open(f"{mol_inp}_txts/twoeint.txt", "r") as reader:
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
  #
  print(f"inside get2e 1")
  baf = qcb.QCBinAr(file=f"{eri_file}.baf")
  # baf = bar.BinArFile(debug=True,file=f"{eri_file}.baf")
  print(f"inside get2e 2")
  # ERI = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ERI-AO.npy",mode='w+',
  #                                 shape=(NBX,NBX,NBX,NBX)) 
  print(f"inside get2e 3")
  ERIsize = NBX**4
  tot_mem, avlb_mem = mem_check()
  print(f"inside get2e 4: {ERIsize*8/(1024**3)}")
  # if(avlb_mem*(1024**3) > 2*np.size(ERI)*8):
  if(avlb_mem*(1024**3) > 2*ERIsize*8):
    ERI = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ERI-AO.npy",mode='w+',
                                    shape=(NBX,NBX,NBX,NBX)) 
    ERI[:,:,:,:] = baf.matlist["REGULAR 2E INTEGRALS"].expand()
    if(ipbc):
      print(f"ERI: {ERI.shape}, {np.size(ERI)}")
      ERI = ERI.reshape((nmtpbc,NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB))
      ERIpbc = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ERI-AO1.npy",
                                         mode='w+',shape=(NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB)) 
      ERIpbc[:,:,:,:,:,:,:] = ERI[0,:,:,:,:,:,:,:]
      del ERIpbc
      os.system(f"mv {scratch}/{mol_out}-ERI-AO1.npy {scratch}/{mol_out}-ERI-AO.npy")
  else:
    with open(f"{mol_out}.txt","a") as writer:
      writer.write(f" Not enough memory to simply expand 2ERIs:\n")
      writer.write(f" AvlMem: {avlb_mem:.2f}GB vs 2ERI size: {8*ERIsize/(1024**3):.2f}GB\n")
    ERI0 = baf.matlist["REGULAR 2E INTEGRALS"]
    _,_,nr,_,ntot,_,_,_,n4,_,_ = ERI0.labpars
    if(n4 != NBX):
      print(f"ERIs size discrepancy: {n4}!={NBX}")
      exit()
    if(nr != 1):
      print(f"Cannot handle complex AO ERI")
      exit()
    print(F"ERI0: ntot: {ntot}, nr: {nr}, {ERI0.array.shape} {np.size(ERI0.array)*8/(1024**3)}")
    r = ERI0.array.reshape([ntot,nr])
    if(ipbc):
      ERI = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ERI-AO.npy",mode='w+',
                                      shape=(NB,NBX,NBX,NBX)) 
      for i in range(n4):
        for j in range(i+1):
          for k in range(i+1):
            if i == k: llim = j + 1
            else: llim = k + 1
            if(i>=NB and j>=NB and k>=NB): llim = min(llim,NB)
            for l in range(llim):
              ijkl,_ = qcmio.lind4(-n4,-n4,-n4,n4,i+1,j+1,k+1,l+1)
              doit = abs(r[ijkl,0]) >= 1.e-12 and (min(i,j,k,l)<NB)
              # if(ipbc): doit = doit and (i<NB or j<NB or k<NB or l<NB)
              if doit:
                if(i<NB):
                  ERI[i,j,k,l] = r[ijkl,0]
                  ERI[i,j,l,k] = r[ijkl,0]
                if(j<NB):
                  ERI[j,i,k,l] = r[ijkl,0]
                  ERI[j,i,l,k] = r[ijkl,0]
                if(k<NB):
                  ERI[k,l,i,j] = r[ijkl,0]
                  ERI[k,l,j,i] = r[ijkl,0]
                if(l<NB):
                  ERI[l,k,i,j] = r[ijkl,0]
                  ERI[l,k,j,i] = r[ijkl,0]
      ERI = ERI.reshape((NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB))
      ERIpbc = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ERI-AO1.npy",
                                         mode='w+',shape=(NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB)) 
      ERIpbc[:,:,:,:,:,:,:] = ERI[:,:,:,:,:,:,:]
      del ERIpbc
      os.system(f"mv {scratch}/{mol_out}-ERI-AO1.npy {scratch}/{mol_out}-ERI-AO.npy")
    else:
      ERI = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ERI-AO.npy",mode='w+',
                                      shape=(NBX,NBX,NBX,NBX)) 
      for i in range(n4):
        for j in range(i+1):
          for k in range(i+1):
            if i == k: llim = j + 1
            else: llim = k + 1
            for l in range(llim):
              ijkl,_ = qcmio.lind4(-n4,-n4,-n4,n4,i+1,j+1,k+1,l+1)
              doit = abs(r[ijkl,0]) >= 1.e-12
              if doit:
                ERI[i,j,k,l] = r[ijkl,0]
                ERI[j,i,k,l] = r[ijkl,0]
                ERI[i,j,l,k] = r[ijkl,0]
                ERI[j,i,l,k] = r[ijkl,0]
                ERI[k,l,i,j] = r[ijkl,0]
                ERI[l,k,i,j] = r[ijkl,0]
                ERI[k,l,j,i] = r[ijkl,0]
                ERI[l,k,j,i] = r[ijkl,0]
  # if(ipbc):
  #   ERIpbc = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ERI-AO1.npy",
  #                                      mode='w+',shape=(NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB)) 
  #   ERI = ERI.reshape((nmtpbc,NB,nmtpbc,NB,nmtpbc,NB,nmtpbc,NB))
  #   ERIpbc[:,:,:,:,:,:,:] = ERI[0,:,:,:,:,:,:,:]
  #   del ERIpbc
  #   os.system(f"mv {scratch}/{mol_out}-ERI-AO1.npy {scratch}/{mol_out}-ERI-AO.npy")
  del ERI
  return
  
#########################################################
####### AO -> MO Basis 2e Integral transformation########
#########################################################
def conMO(mol_out, scratch, O, V, NB, ipbc, MOCoef):
  # AOInt: single-bar 2ERI in AO, Mulliken notation [11|22]
  # MO: double-bar 2ERI in MO, physicist notation <12||12>
  O2 = O*2
  V2 = V*2
  NOrb = O + V
  AOInt = np.load(f"{scratch}/{mol_out}-ERI-AO.npy",mmap_mode='r')
  if(ipbc):
    #
    # PBC Fourier transform
    npdir = ipbc[0]
    nmtpbc = ipbc[1]
    ndimk = ipbc[12:15]
    Nkp, kp, _ = fill_kl(ipbc)
    map_kp = form_map_kp(npdir,ndimk)
    cof = fourier_coef(ipbc,False,0)
    # nmtpbc = ipbc[1]
    # Nkp, kp, l_list = fill_kl(ipbc)
    # # Nkp = len(kp)
    # co = np.einsum('k,l->kl',kp,l_list,optimize=True)
    # cof = np.cos(co) - 1j*np.sin(co)
    # # cof = np.cos(co) + 1j*np.sin(co)
    temp = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp.npy",
                                     mode='w+',shape=(Nkp,NB,NB,nmtpbc,NB,nmtpbc,NB),
                                     dtype=complex) 
    temp2 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp2.npy",
                                      mode='w+',shape=(Nkp,Nkp,NB,NB,NB,nmtpbc,NB),
                                      dtype=complex) 
    AOk = np.lib.format.open_memmap(f"{scratch}/{mol_out}-AOk.npy",
                                    mode='w+',shape=(Nkp,Nkp,Nkp,NB,NB,NB,NB),
                                    dtype=complex) 
    temp[:,:,:,:,:,:,:] = np.einsum('hl,albmcnd->habmcnd',cof,AOInt,optimize=True)
    del AOInt
    temp2[:,:,:,:,:,:,:] = np.einsum('km,habmcnd->hkabcnd',np.conjugate(cof),temp,optimize=True)
    AOk[:,:,:,:,:,:,:] = np.einsum('gn,hkabcnd->hkgabcd',cof,temp2,optimize=True)
    del temp, temp2
    os.system(f"rm {scratch}/{mol_out}-temp*.npy")
    #
    # AO(k,k')->MO(k,k') transformation
    temp = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp.npy",
                                     mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NB,NB),
                                     dtype=complex) 
    temp2 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp2.npy",
                                      mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NOrb,NB),
                                      dtype=complex) 
    temp3 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp3.npy",
                                      mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NOrb,NOrb),
                                      dtype=complex) 
    twoEk = np.lib.format.open_memmap(f"{scratch}/{mol_out}-twoEk.npy",
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
    os.system(f"rm {scratch}/{mol_out}-temp*.npy")
    os.system(f"rm {scratch}/{mol_out}-AOk.npy")
    #
    # Form double-bar integrals in physicist notation <12||12>
    MO = np.lib.format.open_memmap(f"{scratch}/{mol_out}-MO.npy",mode='w+',
                                   shape=(2*NOrb,2*NOrb,2*NOrb,2*NOrb),dtype=complex) 
    ABCDt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ABCDt.npy",mode='w+',
                                      shape=(Nkp,V2,Nkp,V2,Nkp,V2,Nkp,V2),dtype=complex) 
    IABCt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABCt.npy",mode='w+',
                                      shape=(Nkp,O2,Nkp,V2,Nkp,V2,Nkp,V2),dtype=complex) 
    IJABt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJABt.npy",mode='w+',
                                      shape=(Nkp,O2,Nkp,O2,Nkp,V2,Nkp,V2),dtype=complex) 
    IABJt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABJt.npy",mode='w+',
                                      shape=(Nkp,O2,Nkp,V2,Nkp,V2,Nkp,O2),dtype=complex) 
    IJKLt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKLt.npy",mode='w+',
                                      shape=(Nkp,O2,Nkp,O2,Nkp,O2,Nkp,O2),dtype=complex) 
    IJKAt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKAt.npy",mode='w+',
                                      shape=(Nkp,O2,Nkp,O2,Nkp,O2,Nkp,V2),dtype=complex)
    pi2 = round(2*np.pi,10)
    O2k = O2*Nkp
    V2k = V2*Nkp
    Nksum = 0
    for n in range(Nkp):
      for k in range(Nkp):
        for h in range(Nkp):
              gg = abs((n-h+k)%Nkp)
              g = momentum_cons(npdir,ndimk,map_kp,3,n,k,h,0)
              # if(gg!=g):
              #   print(f"this did not work {gg} vs {g}.")
              #   exit()
              
          # for g in range(Nkp):
          #   kn = kp[0,n]
          #   kh = kp[0,h]
          #   kk = kp[0,k]
          #   kg = kp[0,g]
          #   ktot = round(kn-kh+kk-kg,10)
          #   if(abs(ktot) < 1e-8 or abs(ktot%pi2) < 1e-8):
          #     gg = abs((n-h+k)%Nkp)
          #     print(f"n={n}, k={k}, h={h}, g={g}, gg={gg}")
          #     if(gg!=g):
          #       print(f"this did not work {gg} vs {g}.")
          #       exit()
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
              #
              #IJAB
              IJABt[n,:O,k,:O,h,:V,g,:V] = np.copy(MO[:O,:O,O:NOrb,O:NOrb])
              IJABt[n,O:,k,O:,h,V:,g,V:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
              IJABt[n,O:,k,:O,h,V:,g,:V] = np.copy(MO[NOrb:O+NOrb,:O,O+NOrb:2*NOrb,O:NOrb])
              IJABt[n,:O,k,O:,h,:V,g,V:] = np.copy(MO[:O,NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb])
              IJABt[n,O:,k,:O,h,:V,g,V:] = np.copy(MO[NOrb:O+NOrb,:O,O:NOrb,O+NOrb:2*NOrb])
              IJABt[n,:O,k,O:,h,V:,g,:V] = np.copy(MO[:O,NOrb:O+NOrb,O+NOrb:2*NOrb,O:NOrb])
              #
              # IJKL
              IJKLt[n,:O,k,:O,h,:O,g,:O] = np.copy(MO[:O,:O,:O,:O])
              IJKLt[n,O:,k,O:,h,O:,g,O:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb])
              IJKLt[n,O:,k,:O,h,O:,g,:O] = np.copy(MO[NOrb:O+NOrb,:O,NOrb:O+NOrb,:O])
              IJKLt[n,:O,k,O:,h,:O,g,O:] = np.copy(MO[:O,NOrb:O+NOrb,:O,NOrb:O+NOrb])
              IJKLt[n,O:,k,:O,h,:O,g,O:] = np.copy(MO[NOrb:O+NOrb,:O,:O,NOrb:O+NOrb])
              IJKLt[n,:O,k,O:,h,O:,g,:O] = np.copy(MO[:O,NOrb:O+NOrb,NOrb:O+NOrb,:O])
              #
              # IJKA
              IJKAt[n,:O,k,:O,h,:O,g,:V] = np.copy(MO[:O,:O,:O,O:NOrb])
              IJKAt[n,O:,k,O:,h,O:,g,V:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb,O+NOrb:2*NOrb])
              IJKAt[n,O:,k,:O,h,O:,g,:V] = np.copy(MO[NOrb:O+NOrb,:O,NOrb:O+NOrb,O:NOrb])
              IJKAt[n,:O,k,O:,h,:O,g,V:] = np.copy(MO[:O,NOrb:O+NOrb,:O,O+NOrb:2*NOrb])
              IJKAt[n,O:,k,:O,h,:O,g,V:] = np.copy(MO[NOrb:O+NOrb,:O,:O,O+NOrb:2*NOrb])
              IJKAt[n,:O,k,O:,h,O:,g,:V] = np.copy(MO[:O,NOrb:O+NOrb,NOrb:O+NOrb,O:NOrb])
              #
              # IABJ
              IABJt[n,:O,k,:V,h,:V,g,:O] = np.copy(MO[:O,O:NOrb,O:NOrb,:O])
              IABJt[n,O:,k,V:,h,V:,g,O:] = np.copy(MO[NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,NOrb:O+NOrb])
              IABJt[n,O:,k,:V,h,V:,g,:O] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb,:O])
              IABJt[n,:O,k,V:,h,:V,g,O:] = np.copy(MO[:O,O+NOrb:2*NOrb,O:NOrb,NOrb:O+NOrb])
              IABJt[n,O:,k,:V,h,:V,g,O:] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O:NOrb,NOrb:O+NOrb])
              IABJt[n,:O,k,V:,h,V:,g,:O] = np.copy(MO[:O,O+NOrb:2*NOrb,O+NOrb:2*NOrb,:O])
              #
              # IABC
              IABCt[n,:O,k,:V,h,:V,g,:V] = np.copy(MO[:O,O:NOrb,O:NOrb,O:NOrb])
              IABCt[n,O:,k,V:,h,V:,g,V:] = np.copy(MO[NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
              IABCt[n,O:,k,:V,h,V:,g,:V] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb,O:NOrb])
              IABCt[n,:O,k,V:,h,:V,g,V:] = np.copy(MO[:O,O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb])
              IABCt[n,O:,k,:V,h,:V,g,V:] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O:NOrb,O+NOrb:2*NOrb])
              IABCt[n,:O,k,V:,h,V:,g,:V] = np.copy(MO[:O,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O:NOrb])
              #
              # ABCD
              ABCDt[n,:V,k,:V,h,:V,g,:V] = np.copy(MO[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
              ABCDt[n,V:,k,V:,h,V:,g,V:] = np.copy(MO[O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
              ABCDt[n,V:,k,:V,h,V:,g,:V] = np.copy(MO[O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb,O:NOrb])
              ABCDt[n,:V,k,V:,h,:V,g,V:] = np.copy(MO[O:NOrb,O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb])
              ABCDt[n,V:,k,:V,h,:V,g,V:] = np.copy(MO[O+NOrb:2*NOrb,O:NOrb,O:NOrb,O+NOrb:2*NOrb])
              ABCDt[n,:V,k,V:,h,V:,g,:V] = np.copy(MO[O:NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O:NOrb])
    del MO, twoEk
    if(Nksum != Nkp**3):
      # Check that the number of 2ERIs is correct
      with open(f"{mol_out}.txt","a") as writer:
        writer.write(f"There are {NKsum} 2ERIs instead of {Nkp**3}, abort calculation.\n")
      exit()
    #
    os.system(f"rm {scratch}/{mol_out}-MO.npy")
    os.system(f"rm {scratch}/{mol_out}-twoEk.npy")
    ABCD = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ABCD.npy",
                                     mode='w+',shape=(V2k,V2k,V2k,V2k),
                                     dtype=complex) 
    ABCD[:,:,:,:] = ABCDt.reshape((V2k,V2k,V2k,V2k))
    del ABCD, ABCDt
    os.system(f"rm {scratch}/{mol_out}-ABCDt.npy")
    IJAB = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJAB.npy",
                                     mode='w+',shape=(O2k,O2k,V2k,V2k),
                                     dtype=complex) 
    IJAB[:,:,:,:] = IJABt.reshape((O2k,O2k,V2k,V2k))
    del IJAB, IJABt
    os.system(f"rm {scratch}/{mol_out}-IJABt.npy")
    IJKL = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKL.npy",
                                     mode='w+',shape=(O2k,O2k,O2k,O2k),
                                     dtype=complex) 
    IJKL[:,:,:,:] = IJKLt.reshape((O2k,O2k,O2k,O2k))
    del IJKL, IJKLt
    os.system(f"rm {scratch}/{mol_out}-IJKLt.npy")
    IJKA = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKA.npy",
                                     mode='w+',shape=(O2k,O2k,O2k,V2k),
                                     dtype=complex) 
    IJKA[:,:,:,:] = IJKAt.reshape((O2k,O2k,O2k,V2k))
    del IJKA, IJKAt
    os.system(f"rm {scratch}/{mol_out}-IJKAt.npy")
    IABJ = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABJ.npy",
                                     mode='w+',shape=(O2k,V2k,V2k,O2k),
                                     dtype=complex) 
    IABJ[:,:,:,:] = IABJt.reshape((O2k,V2k,V2k,O2k))
    del IABJ, IABJt
    os.system(f"rm {scratch}/{mol_out}-IABJt.npy")
    IABC = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABC.npy",
                                     mode='w+',shape=(O2k,V2k,V2k,V2k),
                                     dtype=complex) 
    IABC[:,:,:,:] = IABCt.reshape((O2k,V2k,V2k,V2k))
    del IABC, IABCt
    os.system(f"rm {scratch}/{mol_out}-IABCt.npy")
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
    ABCD = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ABCD.npy",
                                     mode='w+',shape=(V2,V2,V2,V2))
    ABCD[:V,:V,:V,:V] = np.copy(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
    ABCD[:V,:V,:V,:V] -= np.transpose(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb],axes=(0,1,3,2))
    ABCD[V:,V:,V:,V:] = np.copy(ABCD[:V,:V,:V,:V])
    ABCD[V:,:V,V:,:V] = np.copy(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
    ABCD[:V,V:,:V,V:] = np.copy(ABCD[V:,:V,V:,:V])
    ABCD[V:,:V,:V,V:] = -np.transpose(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb],axes=(0,1,3,2))
    ABCD[:V,V:,V:,:V] = np.copy(ABCD[V:,:V,:V,V:])
    del ABCD
    #
    # IJAB
    IJAB = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJAB.npy",
                                     mode='w+',shape=(O2,O2,V2,V2))
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
    IJKL = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKL.npy",
                                     mode='w+',shape=(O2,O2,O2,O2))
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
    IJKA = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKA.npy",
                                     mode='w+',shape=(O2,O2,O2,V2))
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
    IABJ = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABJ.npy",
                                     mode='w+',shape=(O2,V2,V2,O2))
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
    IABC = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABC.npy",
                                     mode='w+',shape=(O2,V2,V2,V2))
    IABC[:O,:V,:V,:V] = np.copy(twoE[:O,O:NOrb,O:NOrb,O:NOrb])
    IABC[:O,:V,:V,:V] -= np.transpose(twoE[:O,O:NOrb,O:NOrb,O:NOrb],axes=(0,1,3,2))
    IABC[O:,V:,V:,V:] = np.copy(IABC[:O,:V,:V,:V])
    IABC[O:,:V,V:,:V] = np.copy(twoE[:O,O:NOrb,O:NOrb,O:NOrb])
    IABC[:O,V:,:V,V:] = np.copy(IABC[O:,:V,V:,:V])
    IABC[O:,:V,:V,V:] = -np.transpose(twoE[:O,O:NOrb,O:NOrb,O:NOrb],axes=(0,1,3,2))
    IABC[:O,V:,V:,:V] = np.copy(IABC[O:,:V,:V,V:])
    del IABC
    del twoE
  return 

#########################################################
# AO -> CO basis 2e integral transformation for periodic systems with
# collective indices and 4Nkp storage
#########################################################
def TwoECO1(mol_out, scratch, O, V, NB, ipbc, MOCoef):
  # AOInt: single-bar 2ERI in AO, Mulliken notation [11|22]
  # MO: double-bar 2ERI in MO, physicist notation <12||12>
  O2 = O*2
  V2 = V*2
  NOrb = O + V
  AOInt = np.load(f"{scratch}/{mol_out}-ERI-AO.npy",mmap_mode='r')
  #
  # PBC Fourier transform
  npdir = ipbc[0]
  nmtpbc = ipbc[1]
  ndimk = ipbc[12:15]
  Nkp, kp, _ = fill_kl(ipbc)
  map_kp = form_map_kp(npdir,ndimk)
  cof = fourier_coef(ipbc,False,0)
  # nmtpbc = ipbc[1]
  # Nkp, kp, l_list = fill_kl(ipbc)
  # # Nkp = len(kp)
  # co = np.einsum('k,l->kl',kp,l_list,optimize=True)
  # cof = np.cos(co) - 1j*np.sin(co)
  # # cof = np.cos(co) + 1j*np.sin(co)
  temp = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp.npy",
                                   mode='w+',shape=(Nkp,NB,NB,nmtpbc,NB,nmtpbc,NB),
                                   dtype=complex) 
  temp2 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp2.npy",
                                    mode='w+',shape=(Nkp,Nkp,NB,NB,NB,nmtpbc,NB),
                                    dtype=complex) 
  AOk = np.lib.format.open_memmap(f"{scratch}/{mol_out}-AOk.npy",
                                  mode='w+',shape=(Nkp,Nkp,Nkp,NB,NB,NB,NB),
                                  dtype=complex) 
  temp[:,:,:,:,:,:,:] = np.einsum('hl,albmcnd->habmcnd',cof,AOInt,optimize=True)
  del AOInt
  temp2[:,:,:,:,:,:,:] = np.einsum('km,habmcnd->hkabcnd',np.conjugate(cof),temp,optimize=True)
  AOk[:,:,:,:,:,:,:] = np.einsum('gn,hkabcnd->hkgabcd',cof,temp2,optimize=True)
  del temp, temp2
  os.system(f"rm {scratch}/{mol_out}-temp*.npy")
  #
  # AO(k,k')->MO(k,k') transformation
  temp = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp.npy",
                                   mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NB,NB),
                                   dtype=complex) 
  temp2 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp2.npy",
                                    mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NOrb,NB),
                                    dtype=complex) 
  temp3 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp3.npy",
                                    mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NOrb,NOrb),
                                    dtype=complex) 
  twoEk = np.lib.format.open_memmap(f"{scratch}/{mol_out}-twoEk.npy",
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
  os.system(f"rm {scratch}/{mol_out}-temp*.npy")
  os.system(f"rm {scratch}/{mol_out}-AOk.npy")
  #
  # Form double-bar integrals in physicist notation <12||12>
  MO = np.lib.format.open_memmap(f"{scratch}/{mol_out}-MO.npy",mode='w+',
                                 shape=(2*NOrb,2*NOrb,2*NOrb,2*NOrb),dtype=complex) 
  ABCDt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ABCDt.npy",mode='w+',
                                    shape=(Nkp,V2,Nkp,V2,Nkp,V2,Nkp,V2),dtype=complex) 
  IABCt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABCt.npy",mode='w+',
                                    shape=(Nkp,O2,Nkp,V2,Nkp,V2,Nkp,V2),dtype=complex) 
  IJABt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJABt.npy",mode='w+',
                                    shape=(Nkp,O2,Nkp,O2,Nkp,V2,Nkp,V2),dtype=complex) 
  IABJt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABJt.npy",mode='w+',
                                    shape=(Nkp,O2,Nkp,V2,Nkp,V2,Nkp,O2),dtype=complex) 
  IJKLt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKLt.npy",mode='w+',
                                    shape=(Nkp,O2,Nkp,O2,Nkp,O2,Nkp,O2),dtype=complex) 
  IJKAt = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKAt.npy",mode='w+',
                                    shape=(Nkp,O2,Nkp,O2,Nkp,O2,Nkp,V2),dtype=complex)
  # pi2 = round(2*np.pi,10)
  O2k = O2*Nkp
  V2k = V2*Nkp
  # Nksum = 0
  for n in range(Nkp):
    for k in range(Nkp):
      for h in range(Nkp):
        g = momentum_cons(npdir,ndimk,map_kp,3,n,k,h,0)
        # Form double-bar integrals <12||12>. Spin blocks are stored as follows:
        # aaaa: Coulomb - Exchange
        # bbbb: Coulomb - Exchange
        # baba: Coulomb 
        # abab: Coulomb 
        # baab: - Exchange
        # abba: - Exchange
        #
        # Nksum += 1
        MO[:NOrb,:NOrb,:NOrb,:NOrb] = np.copy(twoEk[n,k,h,g,:,:,:,:])
        MO[:NOrb,:NOrb,:NOrb,:NOrb] -= np.transpose(twoEk[n,k,g,h,:,:,:,:],axes=(0,1,3,2))
        MO[NOrb:,NOrb:,NOrb:,NOrb:] = np.copy(MO[:NOrb,:NOrb,:NOrb,:NOrb])
        MO[NOrb:,:NOrb,NOrb:,:NOrb] = np.copy(twoEk[n,k,h,g,:,:,:,:])
        MO[:NOrb,NOrb:,:NOrb,NOrb:] = np.copy(MO[NOrb:,:NOrb,NOrb:,:NOrb])
        MO[NOrb:,:NOrb,:NOrb,NOrb:] = -np.transpose(twoEk[n,k,g,h,:,:,:,:],axes=(0,1,3,2))
        MO[:NOrb,NOrb:,NOrb:,:NOrb] = np.copy(MO[NOrb:,:NOrb,:NOrb,NOrb:])
        #
        #IJAB
        IJABt[n,:O,k,:O,h,:V,g,:V] = np.copy(MO[:O,:O,O:NOrb,O:NOrb])
        IJABt[n,O:,k,O:,h,V:,g,V:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
        IJABt[n,O:,k,:O,h,V:,g,:V] = np.copy(MO[NOrb:O+NOrb,:O,O+NOrb:2*NOrb,O:NOrb])
        IJABt[n,:O,k,O:,h,:V,g,V:] = np.copy(MO[:O,NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb])
        IJABt[n,O:,k,:O,h,:V,g,V:] = np.copy(MO[NOrb:O+NOrb,:O,O:NOrb,O+NOrb:2*NOrb])
        IJABt[n,:O,k,O:,h,V:,g,:V] = np.copy(MO[:O,NOrb:O+NOrb,O+NOrb:2*NOrb,O:NOrb])
        #
        # IJKL
        IJKLt[n,:O,k,:O,h,:O,g,:O] = np.copy(MO[:O,:O,:O,:O])
        IJKLt[n,O:,k,O:,h,O:,g,O:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb])
        IJKLt[n,O:,k,:O,h,O:,g,:O] = np.copy(MO[NOrb:O+NOrb,:O,NOrb:O+NOrb,:O])
        IJKLt[n,:O,k,O:,h,:O,g,O:] = np.copy(MO[:O,NOrb:O+NOrb,:O,NOrb:O+NOrb])
        IJKLt[n,O:,k,:O,h,:O,g,O:] = np.copy(MO[NOrb:O+NOrb,:O,:O,NOrb:O+NOrb])
        IJKLt[n,:O,k,O:,h,O:,g,:O] = np.copy(MO[:O,NOrb:O+NOrb,NOrb:O+NOrb,:O])
        #
        # IJKA
        IJKAt[n,:O,k,:O,h,:O,g,:V] = np.copy(MO[:O,:O,:O,O:NOrb])
        IJKAt[n,O:,k,O:,h,O:,g,V:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb,O+NOrb:2*NOrb])
        IJKAt[n,O:,k,:O,h,O:,g,:V] = np.copy(MO[NOrb:O+NOrb,:O,NOrb:O+NOrb,O:NOrb])
        IJKAt[n,:O,k,O:,h,:O,g,V:] = np.copy(MO[:O,NOrb:O+NOrb,:O,O+NOrb:2*NOrb])
        IJKAt[n,O:,k,:O,h,:O,g,V:] = np.copy(MO[NOrb:O+NOrb,:O,:O,O+NOrb:2*NOrb])
        IJKAt[n,:O,k,O:,h,O:,g,:V] = np.copy(MO[:O,NOrb:O+NOrb,NOrb:O+NOrb,O:NOrb])
        #
        # IABJ
        IABJt[n,:O,k,:V,h,:V,g,:O] = np.copy(MO[:O,O:NOrb,O:NOrb,:O])
        IABJt[n,O:,k,V:,h,V:,g,O:] = np.copy(MO[NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,NOrb:O+NOrb])
        IABJt[n,O:,k,:V,h,V:,g,:O] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb,:O])
        IABJt[n,:O,k,V:,h,:V,g,O:] = np.copy(MO[:O,O+NOrb:2*NOrb,O:NOrb,NOrb:O+NOrb])
        IABJt[n,O:,k,:V,h,:V,g,O:] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O:NOrb,NOrb:O+NOrb])
        IABJt[n,:O,k,V:,h,V:,g,:O] = np.copy(MO[:O,O+NOrb:2*NOrb,O+NOrb:2*NOrb,:O])
        #
        # IABC
        IABCt[n,:O,k,:V,h,:V,g,:V] = np.copy(MO[:O,O:NOrb,O:NOrb,O:NOrb])
        IABCt[n,O:,k,V:,h,V:,g,V:] = np.copy(MO[NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
        IABCt[n,O:,k,:V,h,V:,g,:V] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb,O:NOrb])
        IABCt[n,:O,k,V:,h,:V,g,V:] = np.copy(MO[:O,O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb])
        IABCt[n,O:,k,:V,h,:V,g,V:] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O:NOrb,O+NOrb:2*NOrb])
        IABCt[n,:O,k,V:,h,V:,g,:V] = np.copy(MO[:O,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O:NOrb])
        #
        # ABCD
        ABCDt[n,:V,k,:V,h,:V,g,:V] = np.copy(MO[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
        ABCDt[n,V:,k,V:,h,V:,g,V:] = np.copy(MO[O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
        ABCDt[n,V:,k,:V,h,V:,g,:V] = np.copy(MO[O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb,O:NOrb])
        ABCDt[n,:V,k,V:,h,:V,g,V:] = np.copy(MO[O:NOrb,O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb])
        ABCDt[n,V:,k,:V,h,:V,g,V:] = np.copy(MO[O+NOrb:2*NOrb,O:NOrb,O:NOrb,O+NOrb:2*NOrb])
        ABCDt[n,:V,k,V:,h,V:,g,:V] = np.copy(MO[O:NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O:NOrb])
  del MO, twoEk
  # if(Nksum != Nkp**3):
  #   # Check that the number of 2ERIs is correct
  #   with open(f"{mol_out}.txt","a") as writer:
  #     writer.write(f"There are {NKsum} 2ERIs instead of {Nkp**3}, abort calculation.\n")
  #   exit()
  # #
  os.system(f"rm {scratch}/{mol_out}-MO.npy")
  os.system(f"rm {scratch}/{mol_out}-twoEk.npy")
  ABCD = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ABCD.npy",
                                   mode='w+',shape=(V2k,V2k,V2k,V2k),
                                   dtype=complex) 
  ABCD[:,:,:,:] = ABCDt.reshape((V2k,V2k,V2k,V2k))
  del ABCD, ABCDt
  os.system(f"rm {scratch}/{mol_out}-ABCDt.npy")
  IJAB = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJAB.npy",
                                   mode='w+',shape=(O2k,O2k,V2k,V2k),
                                   dtype=complex) 
  IJAB[:,:,:,:] = IJABt.reshape((O2k,O2k,V2k,V2k))
  del IJAB, IJABt
  os.system(f"rm {scratch}/{mol_out}-IJABt.npy")
  IJKL = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKL.npy",
                                   mode='w+',shape=(O2k,O2k,O2k,O2k),
                                   dtype=complex) 
  IJKL[:,:,:,:] = IJKLt.reshape((O2k,O2k,O2k,O2k))
  del IJKL, IJKLt
  os.system(f"rm {scratch}/{mol_out}-IJKLt.npy")
  IJKA = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKA.npy",
                                   mode='w+',shape=(O2k,O2k,O2k,V2k),
                                   dtype=complex) 
  IJKA[:,:,:,:] = IJKAt.reshape((O2k,O2k,O2k,V2k))
  del IJKA, IJKAt
  os.system(f"rm {scratch}/{mol_out}-IJKAt.npy")
  IABJ = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABJ.npy",
                                   mode='w+',shape=(O2k,V2k,V2k,O2k),
                                   dtype=complex) 
  IABJ[:,:,:,:] = IABJt.reshape((O2k,V2k,V2k,O2k))
  del IABJ, IABJt
  os.system(f"rm {scratch}/{mol_out}-IABJt.npy")
  IABC = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABC.npy",
                                   mode='w+',shape=(O2k,V2k,V2k,V2k),
                                   dtype=complex) 
  IABC[:,:,:,:] = IABCt.reshape((O2k,V2k,V2k,V2k))
  del IABC, IABCt
  os.system(f"rm {scratch}/{mol_out}-IABCt.npy")
  return 

#########################################################
# AO -> CO basis 2e integral transformation for periodic systems with
# explicit k indices and 3Nkp storage
#########################################################
def TwoECO2(mol_out, scratch, O, V, NB, ipbc, MOCoef):
  # AOInt: single-bar 2ERI in AO, Mulliken notation [11|22]
  # MO: double-bar 2ERI in MO, physicist notation <12||12>
  O2 = O*2
  V2 = V*2
  NOrb = O + V
  AOInt = np.load(f"{scratch}/{mol_out}-ERI-AO.npy",mmap_mode='r')
  #
  # PBC Fourier transform
  npdir = ipbc[0]
  nmtpbc = ipbc[1]
  ndimk = ipbc[12:15]
  Nkp, kp, _ = fill_kl(ipbc)
  map_kp = form_map_kp(npdir,ndimk)
  cof = fourier_coef(ipbc,False,0)
  # nmtpbc = ipbc[1]
  # Nkp, kp, l_list = fill_kl(ipbc)
  # # Nkp = len(kp)
  # co = np.einsum('k,l->kl',kp,l_list,optimize=True)
  # cof = np.cos(co) - 1j*np.sin(co)
  # # cof = np.cos(co) + 1j*np.sin(co)
  temp = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp.npy",
                                   mode='w+',shape=(Nkp,NB,NB,nmtpbc,NB,nmtpbc,NB),
                                   dtype=complex) 
  temp2 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp2.npy",
                                    mode='w+',shape=(Nkp,Nkp,NB,NB,NB,nmtpbc,NB),
                                    dtype=complex) 
  AOk = np.lib.format.open_memmap(f"{scratch}/{mol_out}-AOk.npy",
                                  mode='w+',shape=(Nkp,Nkp,Nkp,NB,NB,NB,NB),
                                  dtype=complex) 
  temp[:,:,:,:,:,:,:] = np.einsum('hl,albmcnd->habmcnd',cof,AOInt,optimize=True)
  del AOInt
  temp2[:,:,:,:,:,:,:] = np.einsum('km,habmcnd->hkabcnd',np.conjugate(cof),temp,optimize=True)
  AOk[:,:,:,:,:,:,:] = np.einsum('gn,hkabcnd->hkgabcd',cof,temp2,optimize=True)
  del temp, temp2
  os.system(f"rm {scratch}/{mol_out}-temp*.npy")
  #
  # AO(k,k')->MO(k,k') transformation
  temp = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp.npy",
                                   mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NB,NB),
                                   dtype=complex) 
  temp2 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp2.npy",
                                    mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NOrb,NB),
                                    dtype=complex) 
  temp3 = np.lib.format.open_memmap(f"{scratch}/{mol_out}-temp3.npy",
                                    mode='w+',shape=(Nkp,Nkp,Nkp,NB,NOrb,NOrb,NOrb),
                                    dtype=complex) 
  twoEk = np.lib.format.open_memmap(f"{scratch}/{mol_out}-twoEk.npy",
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
  os.system(f"rm {scratch}/{mol_out}-temp*.npy")
  os.system(f"rm {scratch}/{mol_out}-AOk.npy")
  #
  # Form double-bar integrals in physicist notation <12||12>
  MO = np.lib.format.open_memmap(f"{scratch}/{mol_out}-MO.npy",mode='w+',
                                 shape=(2*NOrb,2*NOrb,2*NOrb,2*NOrb),dtype=complex) 
  ABCD = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ABCD.npy",mode='w+',
                                    shape=(Nkp,Nkp,Nkp,V2,V2,V2,V2),dtype=complex) 
  IABC = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABC.npy",mode='w+',
                                    shape=(Nkp,Nkp,Nkp,O2,V2,V2,V2),dtype=complex) 
  IJAB = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJAB.npy",mode='w+',
                                    shape=(Nkp,Nkp,Nkp,O2,O2,V2,V2),dtype=complex) 
  IABJ = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABJ.npy",mode='w+',
                                    shape=(Nkp,Nkp,Nkp,O2,V2,V2,O2),dtype=complex) 
  IJKL = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKL.npy",mode='w+',
                                    shape=(Nkp,Nkp,Nkp,O2,O2,O2,O2),dtype=complex) 
  IJKA = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKA.npy",mode='w+',
                                    shape=(Nkp,Nkp,Nkp,O2,O2,O2,V2),dtype=complex)
  # pi2 = round(2*np.pi,10)
  O2k = O2*Nkp
  V2k = V2*Nkp
  # Nksum = 0
  for n in range(Nkp):
    for k in range(Nkp):
      for h in range(Nkp):
        g = momentum_cons(npdir,ndimk,map_kp,3,n,k,h,0)
        # Form double-bar integrals <12||12>. Spin blocks are stored as follows:
        # aaaa: Coulomb - Exchange
        # bbbb: Coulomb - Exchange
        # baba: Coulomb 
        # abab: Coulomb 
        # baab: - Exchange
        # abba: - Exchange
        #
        # Nksum += 1
        MO[:NOrb,:NOrb,:NOrb,:NOrb] = np.copy(twoEk[n,k,h,g,:,:,:,:])
        MO[:NOrb,:NOrb,:NOrb,:NOrb] -= np.transpose(twoEk[n,k,g,h,:,:,:,:],axes=(0,1,3,2))
        MO[NOrb:,NOrb:,NOrb:,NOrb:] = np.copy(MO[:NOrb,:NOrb,:NOrb,:NOrb])
        MO[NOrb:,:NOrb,NOrb:,:NOrb] = np.copy(twoEk[n,k,h,g,:,:,:,:])
        MO[:NOrb,NOrb:,:NOrb,NOrb:] = np.copy(MO[NOrb:,:NOrb,NOrb:,:NOrb])
        MO[NOrb:,:NOrb,:NOrb,NOrb:] = -np.transpose(twoEk[n,k,g,h,:,:,:,:],axes=(0,1,3,2))
        MO[:NOrb,NOrb:,NOrb:,:NOrb] = np.copy(MO[NOrb:,:NOrb,:NOrb,NOrb:])
        #
        #IJAB
        IJAB[n,k,h,:O,:O,:V,:V] = np.copy(MO[:O,:O,O:NOrb,O:NOrb])
        IJAB[n,k,h,O:,O:,V:,V:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
        IJAB[n,k,h,O:,:O,V:,:V] = np.copy(MO[NOrb:O+NOrb,:O,O+NOrb:2*NOrb,O:NOrb])
        IJAB[n,k,h,:O,O:,:V,V:] = np.copy(MO[:O,NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb])
        IJAB[n,k,h,O:,:O,:V,V:] = np.copy(MO[NOrb:O+NOrb,:O,O:NOrb,O+NOrb:2*NOrb])
        IJAB[n,k,h,:O,O:,V:,:V] = np.copy(MO[:O,NOrb:O+NOrb,O+NOrb:2*NOrb,O:NOrb])
        #
        # IJKL
        IJKL[n,k,h,:O,:O,:O,:O] = np.copy(MO[:O,:O,:O,:O])
        IJKL[n,k,h,O:,O:,O:,O:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb])
        IJKL[n,k,h,O:,:O,O:,:O] = np.copy(MO[NOrb:O+NOrb,:O,NOrb:O+NOrb,:O])
        IJKL[n,k,h,:O,O:,:O,O:] = np.copy(MO[:O,NOrb:O+NOrb,:O,NOrb:O+NOrb])
        IJKL[n,k,h,O:,:O,:O,O:] = np.copy(MO[NOrb:O+NOrb,:O,:O,NOrb:O+NOrb])
        IJKL[n,k,h,:O,O:,O:,:O] = np.copy(MO[:O,NOrb:O+NOrb,NOrb:O+NOrb,:O])
        #
        # IJKA
        IJKA[n,k,h,:O,:O,:O,:V] = np.copy(MO[:O,:O,:O,O:NOrb])
        IJKA[n,k,h,O:,O:,O:,V:] = np.copy(MO[NOrb:O+NOrb,NOrb:O+NOrb,NOrb:O+NOrb,O+NOrb:2*NOrb])
        IJKA[n,k,h,O:,:O,O:,:V] = np.copy(MO[NOrb:O+NOrb,:O,NOrb:O+NOrb,O:NOrb])
        IJKA[n,k,h,:O,O:,:O,V:] = np.copy(MO[:O,NOrb:O+NOrb,:O,O+NOrb:2*NOrb])
        IJKA[n,k,h,O:,:O,:O,V:] = np.copy(MO[NOrb:O+NOrb,:O,:O,O+NOrb:2*NOrb])
        IJKA[n,k,h,:O,O:,O:,:V] = np.copy(MO[:O,NOrb:O+NOrb,NOrb:O+NOrb,O:NOrb])
        #
        # IABJ
        IABJ[n,k,h,:O,:V,:V,:O] = np.copy(MO[:O,O:NOrb,O:NOrb,:O])
        IABJ[n,k,h,O:,V:,V:,O:] = np.copy(MO[NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,NOrb:O+NOrb])
        IABJ[n,k,h,O:,:V,V:,:O] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb,:O])
        IABJ[n,k,h,:O,V:,:V,O:] = np.copy(MO[:O,O+NOrb:2*NOrb,O:NOrb,NOrb:O+NOrb])
        IABJ[n,k,h,O:,:V,:V,O:] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O:NOrb,NOrb:O+NOrb])
        IABJ[n,k,h,:O,V:,V:,:O] = np.copy(MO[:O,O+NOrb:2*NOrb,O+NOrb:2*NOrb,:O])
        #
        # IABC
        IABC[n,k,h,:O,:V,:V,:V] = np.copy(MO[:O,O:NOrb,O:NOrb,O:NOrb])
        IABC[n,k,h,O:,V:,V:,V:] = np.copy(MO[NOrb:O+NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
        IABC[n,k,h,O:,:V,V:,:V] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O+NOrb:2*NOrb,O:NOrb])
        IABC[n,k,h,:O,V:,:V,V:] = np.copy(MO[:O,O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb])
        IABC[n,k,h,O:,:V,:V,V:] = np.copy(MO[NOrb:O+NOrb,O:NOrb,O:NOrb,O+NOrb:2*NOrb])
        IABC[n,k,h,:O,V:,V:,:V] = np.copy(MO[:O,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O:NOrb])
        #
        # ABCD
        ABCD[n,k,h,:V,:V,:V,:V] = np.copy(MO[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
        ABCD[n,k,h,V:,V:,V:,V:] = np.copy(MO[O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb])
        ABCD[n,k,h,V:,:V,V:,:V] = np.copy(MO[O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb,O:NOrb])
        ABCD[n,k,h,:V,V:,:V,V:] = np.copy(MO[O:NOrb,O+NOrb:2*NOrb,O:NOrb,O+NOrb:2*NOrb])
        ABCD[n,k,h,V:,:V,:V,V:] = np.copy(MO[O+NOrb:2*NOrb,O:NOrb,O:NOrb,O+NOrb:2*NOrb])
        ABCD[n,k,h,:V,V:,V:,:V] = np.copy(MO[O:NOrb,O+NOrb:2*NOrb,O+NOrb:2*NOrb,O:NOrb])
  del MO, twoEk
  return 

#########################################################
# AO -> MO basis 2e integral transformation for molecules
#########################################################
def TwoEMO(mol_out, scratch, O, V, NB, MOCoef):
  # AOInt: single-bar 2ERI in AO, Mulliken notation [11|22]
  # MO: double-bar 2ERI in MO, physicist notation <12||12>
  O2 = O*2
  V2 = V*2
  NOrb = O + V
  AOInt = np.load(f"{scratch}/{mol_out}-ERI-AO.npy",mmap_mode='r')
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
  ABCD = np.lib.format.open_memmap(f"{scratch}/{mol_out}-ABCD.npy",
                                   mode='w+',shape=(V2,V2,V2,V2))
  ABCD[:V,:V,:V,:V] = np.copy(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
  ABCD[:V,:V,:V,:V] -= np.transpose(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb],
                                    axes=(0,1,3,2))
  ABCD[V:,V:,V:,V:] = np.copy(ABCD[:V,:V,:V,:V])
  ABCD[V:,:V,V:,:V] = np.copy(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb])
  ABCD[:V,V:,:V,V:] = np.copy(ABCD[V:,:V,V:,:V])
  ABCD[V:,:V,:V,V:] = -np.transpose(twoE[O:NOrb,O:NOrb,O:NOrb,O:NOrb],
                                    axes=(0,1,3,2))
  ABCD[:V,V:,V:,:V] = np.copy(ABCD[V:,:V,:V,V:])
  del ABCD
  #
  # IJAB
  IJAB = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJAB.npy",
                                   mode='w+',shape=(O2,O2,V2,V2))
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
  IJKL = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKL.npy",
                                   mode='w+',shape=(O2,O2,O2,O2))
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
  IJKA = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IJKA.npy",
                                   mode='w+',shape=(O2,O2,O2,V2))
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
  IABJ = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABJ.npy",
                                   mode='w+',shape=(O2,V2,V2,O2))
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
  IABC = np.lib.format.open_memmap(f"{scratch}/{mol_out}-IABC.npy",
                                   mode='w+',shape=(O2,V2,V2,V2))
  IABC[:O,:V,:V,:V] = np.copy(twoE[:O,O:NOrb,O:NOrb,O:NOrb])
  IABC[:O,:V,:V,:V] -= np.transpose(twoE[:O,O:NOrb,O:NOrb,O:NOrb],axes=(0,1,3,2))
  IABC[O:,V:,V:,V:] = np.copy(IABC[:O,:V,:V,:V])
  IABC[O:,:V,V:,:V] = np.copy(twoE[:O,O:NOrb,O:NOrb,O:NOrb])
  IABC[:O,V:,:V,V:] = np.copy(IABC[O:,:V,V:,:V])
  IABC[O:,:V,:V,V:] = -np.transpose(twoE[:O,O:NOrb,O:NOrb,O:NOrb],axes=(0,1,3,2))
  IABC[:O,V:,V:,:V] = np.copy(IABC[O:,:V,:V,V:])
  del IABC
  del twoE
  return 

#########################################################
# Get perturbation integrals in AO basis from file
#########################################################
def getPertAO(NB,ipbc,pert_type,mol_inp):
  NBX = NB
  ntt = NB*(NB+1)//2
  nttx = ntt
  NP4 = 0
  if(ipbc):
    nmtpbc = ipbc[1]
    NBX = NB*nmtpbc
    nttx = ntt*nmtpbc
  #
  if(pert_type == "DipE"):
    NP1 = 3
    NP2 = 3
    NP3 = 0
    NP = 3
    AOPert = np.zeros((NP*nttx))
    if(f"{mol_inp}_txts/dipole_r.txt"):
      with open(f"{mol_inp}_txts/dipole_r.txt","r") as reader:
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
    if(f"{mol_inp}_txts/dipole_v.txt"):
      with open(f"{mol_inp}_txts/dipole_v.txt","r") as reader:
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
    # OR MVG
    NP1 = 3
    NP2 = 3
    NP3 = 0
    NP = 6
    AOPert = np.zeros((NP*nttx))
    if(f"{mol_inp}_txts/dipole_v.txt"):
      with open(f"{mol_inp}_txts/dipole_v.txt","r") as reader:
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
    if(f"{mol_inp}_txts/magnetic.txt"):
      with open(f"{mol_inp}_txts/magnetic.txt","r") as reader:
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
  elif(pert_type == "OR_L"):
    # OR LG(OI)
    # Order of perturbations is: mu(L), m, mu(V)
    NP1 = 3
    NP2 = 3
    NP3 = 3
    NP = 9
    AOPert = np.zeros((NP*nttx))
    if(f"{mol_inp}_txts/dipole_r.txt"):
      with open(f"{mol_inp}_txts/dipole_r.txt","r") as reader:
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
      print(f" No electric dipole integrals found\n")
      exit()
    if(f"{mol_inp}_txts/magnetic.txt"):
      with open(f"{mol_inp}_txts/magnetic.txt","r") as reader:
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
    if(f"{mol_inp}_txts/dipole_v.txt"):
      with open(f"{mol_inp}_txts/dipole_v.txt","r") as reader:
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
  elif(pert_type == "FullOR_V"):
    # Full OR tensor MVG
    # Order of perturbations is: mu(V), m, Theta(V)
    NP1 = 3
    NP2 = 3
    NP3 = 6
    NP = 12
    AOPert = np.zeros((NP*nttx))
    # Read electric dipole V integrals (Del)
    if(f"{mol_inp}_txts/dipole_v.txt"):
      with open(f"{mol_inp}_txts/dipole_v.txt","r") as reader:
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
    if(f"{mol_inp}_txts/magnetic.txt"):
      with open(f"{mol_inp}_txts/magnetic.txt","r") as reader:
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
      print(f" No magnetic dipole integrals found\n")
      exit()
    # Read electric quadrupole integrals ((r Del + Del r)/2)
    if(f"{mol_inp}_txts/quadrupole_v.txt"):
      with open(f"{mol_inp}_txts/quadrupole_v.txt","r") as reader:
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
      print(f" No velocity electric quadrupole integrals found\n")
      exit()
    AOPert = AOPert.reshape(NP,nttx)
  elif(pert_type == "FullOR_L"):
    # Full OR tensor LG(OI)
    #
    # Order of perturbations is, initially: # mu(V), mu(L), m,
    # Theta(V), then we move them to mu(L), m, Theta(V), mu(V). The
    # reason is that for PBC, we need the mu(V) integrals to correct m
    # and Theta(V), but then the tensor evaluation expects mu(V) last.
    NP1 = 3
    NP2 = 3
    NP3 = 6
    NP4 = 3
    NP = 15
    AOPert = np.zeros((NP*nttx))
    # Read electric dipole V integrals (Del)
    if(f"{mol_inp}_txts/dipole_v.txt"):
      with open(f"{mol_inp}_txts/dipole_v.txt","r") as reader:
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
    # Read electric dipole L integrals (r)
    if(f"{mol_inp}_txts/dipole_r.txt"):
      with open(f"{mol_inp}_txts/dipole_r.txt","r") as reader:
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
      print(f" No electric dipole integrals found\n")
      exit()
    # Read magnetic dipole integrals (r x Del)
    if(f"{mol_inp}_txts/magnetic.txt"):
      with open(f"{mol_inp}_txts/magnetic.txt","r") as reader:
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
      print(f" No magnetic dipole integrals found\n")
      exit()
    # Read electric quadrupole integrals ((r Del + Del r)/2)
    if(f"{mol_inp}_txts/quadrupole_v.txt"):
      with open(f"{mol_inp}_txts/quadrupole_v.txt","r") as reader:
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
      print(f" No velocity electric quadrupole integrals found\n")
      exit()
    AOPert = AOPert.reshape(NP,nttx)
  else:
    print(f" Perturbation ",pert_type," is not available")
    exit()
  return NP, NP1, NP2, NP3, NP4, AOPert

#########################################################
# Get perturbation integrals and return them in MO basis
#########################################################
def getPert(O,V,NB,ipbc,tv,MOCoef,Fock,pert_type,mol_inp,mol_out):
  NBX = NB
  O2 = 2*O
  V2 = 2*V
  NOrb = O + V
  O2k = O2
  V2k = V2
  ntt = NB*(NB+1)//2
  nttx = ntt
  NP4 = 0
  if(ipbc):
    Nkp, _, _ = fill_kl(ipbc)
    # Nkp = len(kp)
    nmtpbc = ipbc[1]
    NBX = NB*nmtpbc
    O2k = O2*Nkp
    V2k = V2*Nkp
    nttx = ntt*nmtpbc
  #
  with open(f"{mol_out}.txt","a") as writer:
    writer.write(f"Reading perturbation {pert_type}\n")
  #
  # Read AO integrals
  NP, NP1, NP2, NP3, NP4, AOPert = getPertAO(NB,ipbc,pert_type,mol_inp)
  #
  # Now transform into MO/CO basis 
  if(ipbc):
    # PBC case
    #
    npdir = ipbc[0]
    UMat = []
    UMatS = []
    if(pert_type == "DipE" or pert_type == "FullOR_V" or pert_type == "FullOR_L"):
      # For the length gauge electric dipole, for the magnetic dipole
      # and electric quadrupole, we need the translation vector and to
      # form the U matrix
      #
      OrbE = np.diag(Fock.real)
      NOrb2k = NOrb*2*Nkp
      if(len(OrbE)!=NOrb2k):
        print(f"Mismatch in the number of orbital energies: {NOrb2k} != {len(OrbE)}")
        exit()
      DE = DEk(NOrb2k,OrbE)
      for dir_dk in range(npdir):
        # dF/dk = F'
        FockDk = getFock(mol_inp,O,V,NB,ipbc,"MO",True,dir_dk,MOCoef)
        #
        # dS/dk = S'
        OvlDk = getOvl(mol_inp,O,V,NB,ipbc,"MO",True,dir_dk,MOCoef)
        #
        # Form i(U + 1/2S') or iU or both
        UMat.append(FockDk - np.einsum('ij,j->ij',OvlDk,OrbE,optimize=True))
        UMat[dir_dk] /= -DE
        if(pert_type == "DipE"):
          UMat[dir_dk] += 0.5*OvlDk
          # The diagonal of this matrix is 0
          np.fill_diagonal(UMat[dir_dk],0)
        elif(pert_type == "FullOR_V"):
          # The diagonal of this matrix is -S'/2
          np.fill_diagonal(UMat[dir_dk],-np.diag(OvlDk)/2)
        elif(pert_type == "FullOR_L"):
          # We need both matrices
          UMatS.append(UMat[dir_dk] + 0.5*OvlDk)
          np.fill_diagonal(UMatS[dir_dk],0)
          UMatS[dir_dk] = UMatS[dir_dk]*1j
          np.fill_diagonal(UMat[dir_dk],-np.diag(OvlDk)/2)
        UMat[dir_dk] = UMat[dir_dk]*1j
      del FockDk, OvlDk, OrbE, DE
    # Now form the perturbation matrices in MO(k) basis
    X_ij = np.zeros((NP,Nkp,Nkp,O2,O2),dtype=complex)
    X_ia = np.zeros((NP,Nkp,Nkp,O2,V2),dtype=complex)
    X_ab = np.zeros((NP,Nkp,Nkp,V2,V2),dtype=complex)
    AOPert = AOPert.reshape((NP,nmtpbc,ntt))
    save_p = []
    if(pert_type == "FullOR_V" or pert_type == "FullOR_L"):
      save_p = np.zeros((NP1,Nkp*NOrb*2,Nkp*NOrb*2),dtype=complex)
    for n in range (NP):
      Pert_k_lt = fourier("Dir",ipbc,AOPert[n,:,:],False,0)
      if(pert_type == "DipE"):
        # Electric dipole length gauge
        PertA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Pert_k_lt)
      elif(pert_type == "DipEV" or pert_type == "FullOR_V"):
        # Electric dipole velocity gauge
        PertA = basis_tran("Dir",True,False,"AHer",NB,Nkp,MOCoef,Pert_k_lt)
      elif(pert_type == "FullOR_L"):
        if(n >= NP1 and n <NP1+NP2):
          # mu(L)
          PertA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Pert_k_lt)
        else:
          # m or Theta(V) or mu(V)
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
        for dir_dk in range(npdir):
          Pert -= UMat[dir_dk]*tv[dir_dk][n]
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
          for dir_dk in range(npdir):
            temp = -tv[dir_dk][nn2]*np.einsum('ij,jk->ik',save_p[nn1,:,:],
                                              UMat[dir_dk],optimize=True)
            temp += tv[dir_dk][nn1]*np.einsum('ij,jk->ik',save_p[nn2,:,:],
                                              UMat[dir_dk],optimize=True)
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
            for dir_dk in range(npdir):
              temp = 2*tv[dir_dk][nn2]*np.einsum('ij,jk->ik',save_p[nn1,:,:],
                                                 UMat[dir_dk],optimize=True)
              temp = (temp - np.conjugate(temp).T)/2
              Pert -= temp
              del temp
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
            for dir_dk in range(npdir):
              temp = tv[dir_dk][nn2]*np.einsum('ij,jk->ik',save_p[nn1,:,:],
                                               UMat[dir_dk],optimize=True)
              temp += tv[dir_dk][nn1]*np.einsum('ij,jk->ik',save_p[nn2,:,:],
                                                UMat[dir_dk],optimize=True)
              temp = (temp - np.conjugate(temp).T)/2
              Pert -= temp
              del temp
        Pert = Pert.reshape((Nkp,NOrb*2,Nkp,NOrb*2))
      elif(pert_type == "FullOR_L"):
        # Add UMat contributions to mu(L), m, and Theta(V)
        Pert = Pert.reshape((Nkp*NOrb*2,Nkp*NOrb*2))
        if(n < NP1):
          # Save electric dipole integrals in temporary array
          save_p[n,:,:] = np.copy(Pert)
        elif(n < NP1+NP2):
          # mu(L)
          for dir_dk in range(npdir):
            Pert -= UMatS[dir_dk]*tv[dir_dk][n-NP1]
        elif(n < NP1+NP2+NP4):
          # Correct magnetic dipole
          nn0 = n - (NP1+NP2)
          nn1 = (nn0+1)%3 
          nn2 = (nn0+2)%3
          for dir_dk in range(npdir):
            temp = -tv[dir_dk][nn2]*np.einsum('ij,jk->ik',save_p[nn1,:,:],
                                              UMat[dir_dk],optimize=True)
            temp += tv[dir_dk][nn1]*np.einsum('ij,jk->ik',save_p[nn2,:,:],
                                              UMat[dir_dk],optimize=True)
            temp = (temp - np.conjugate(temp).T)/2
            Pert -= temp
            del temp
        else:
          # Correct electric quadrupole
          nn0 = n - (NP1+NP2+NP4)
          if(nn0 < 3):
            nn1 = nn0
            nn2 = nn0
            for dir_dk in range(npdir):
              temp = 2*tv[dir_dk][nn2]*np.einsum('ij,jk->ik',save_p[nn1,:,:],
                                                 UMat[dir_dk],optimize=True)
              temp = (temp - np.conjugate(temp).T)/2
              Pert -= temp
              del temp
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
            for dir_dk in range(npdir):
              temp = tv[dir_dk][nn2]*np.einsum('ij,jk->ik',save_p[nn1,:,:],
                                               UMat[dir_dk],optimize=True)
              temp += tv[dir_dk][nn1]*np.einsum('ij,jk->ik',save_p[nn2,:,:],
                                                UMat[dir_dk],optimize=True)
              temp = (temp - np.conjugate(temp).T)/2
              Pert -= temp
              del temp
        Pert = Pert.reshape((Nkp,NOrb*2,Nkp,NOrb*2))
      Pert = np.transpose(Pert,axes=(0,2,1,3))
      for k in range(Nkp):
        prod = np.einsum('ij,ij',Pert[k,k],np.conjugate(Pert[k,k]),optimize=True)
        print(f"Kp:{k+1}, Pert-{n+1}: {prod}")
        X_ij[n,k,k,:,:] = Pert[k,k,:O2,:O2]
        X_ia[n,k,k,:,:] = Pert[k,k,:O2,O2:]
        X_ab[n,k,k,:,:] = Pert[k,k,O2:,O2:]
    del UMat, UMatS
    if(pert_type == "FullOR_L"):
      # First, reorder perturbations and put mu(V) last
      X_ij = np.roll(X_ij,-3,axis=0)      
      X_ia = np.roll(X_ia,-3,axis=0)      
      X_ab = np.roll(X_ab,-3,axis=0)      
    if(pert_type == "DipE"):
      X_ij = np.transpose(X_ij,axes=(0,1,3,2,4))
    elif(pert_type == "DipEV" or pert_type == "FullOR_V"):
      X_ij = np.transpose(X_ij,axes=(0,2,4,1,3))
    elif(pert_type == "FullOR_L"):
      X_ij0 = np.zeros((NP,Nkp,O2,Nkp,O2),dtype=X_ij.dtype)
      for n in range(NP):
        if(n<NP1):
          # mu(L)
          X_ij0[n,:,:,:,:] = np.transpose(X_ij[n,:,:,:,:],axes=(0,2,1,3))
        else:
          # m, Theta(V), mu(V)
          X_ij0[n,:,:,:,:] = np.transpose(X_ij[n,:,:,:,:],axes=(1,3,0,2))
      X_ij = X_ij0
      del X_ij0
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
    elif(pert_type == "FullOR_L"):
      X_ab0 = np.zeros((NP,Nkp,V2,Nkp,V2),dtype=X_ab.dtype)
      for n in range(NP):
        if(n<NP1):
          # mu(L)
          X_ab0[n,:,:,:,:] = np.transpose(X_ab[n,:,:,:,:],axes=(0,2,1,3))
        else:
          # m, Theta(V), mu(V)
          X_ab0[n,:,:,:,:] = np.transpose(X_ab[n,:,:,:,:],axes=(1,3,0,2))
      X_ab = X_ab0
      del X_ab0
    else:
      print(f"getPert is confused about X_ab")
      exit()
    X_ab = X_ab.reshape((NP,V2k,V2k))
    X_ia = np.conjugate(X_ia)
    if(pert_type == "DipE"):
      X_ij = np.conjugate(X_ij)
      X_ab = np.conjugate(X_ab)
    elif(pert_type == "FullOR_L"):
      for n in range(NP1):
        # mu(L)
        X_ij[n,:,:] = np.conjugate(X_ij[n,:,:])
        X_ab[n,:,:] = np.conjugate(X_ab[n,:,:])
    for n in range(NP):
      prodXij = np.einsum('ij,ij',X_ij[n],np.conjugate(X_ij[n]),optimize=True)
      prodXab = np.einsum('ij,ij',X_ab[n],np.conjugate(X_ab[n]),optimize=True)
      prodXia = np.einsum('ij,ij',X_ia[n],np.conjugate(X_ia[n]),optimize=True)
      print(f"Pert:{n+1}, Xij: {prodXij}, Xab: {prodXab}, Xia: {prodXia}")
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
    elif(pert_type == "OR_L" or pert_type == "FullOR_L"):
      if(pert_type == "FullOR_L"):
        # First, reorder perturbations and put mu(V) last
        AOPert[:,:] = np.roll(AOPert,-3,axis=0)      
      # Electric dipole length, electric dipole , magnetic dipole,
      # electric quadrupole velocity gauge
      for n in range (NP):
        if n < NP1:
          # mu L is symmetric
          PertSQ[n,:,:] = square_m(NB,True,"Sym",AOPert[n,:],PertSQ[n,:,:])
        else:
          # all other multipoles are antisymmetric
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
      for i in range(O):
        for a in range(V):
          X_ia[n,i,a] = temp[n,i,a+O]
          X_ia[n,i+O,a+V] = temp[n,i,a+O]
      for a in range(V):
        for b in range(V):
          X_ab[n,a,b] = temp[n,b+O,a+O]
          X_ab[n,a+V,b+V] = temp[n,b+O,a+O]
    del temp, AOPert, PertSQ
  return NP, NP1, NP2, NP3, NP4, X_ij, X_ia, X_ab

#########################################################
# Get perturbation integrals and return them in MO basis
# with Nkp storage
#########################################################
def getPert1k(O,V,NB,ipbc,tv,MOCoef,Fock,pert_type,mol_inp,mol_out):
  NBX = NB
  O2 = 2*O
  V2 = 2*V
  NOrb = O + V
  NOrb2 = NOrb*2
  ntt = NB*(NB+1)//2
  nttx = ntt
  NP4 = 0
  if(ipbc):
    Nkp, _, _ = fill_kl(ipbc)
    # Nkp = len(kp)
    nmtpbc = ipbc[1]
    NBX = NB*nmtpbc
    nttx = ntt*nmtpbc
  #
  with open(f"{mol_out}.txt","a") as writer:
    writer.write(f"Reading perturbation {pert_type}\n")
  #
  # Read AO integrals
  NP, NP1, NP2, NP3, NP4, AOPert = getPertAO(NB,ipbc,pert_type,mol_inp)
  #
  # Now transform into MO/CO basis 
  npdir = ipbc[0]
  UMat = []
  UMatS = []
  if(pert_type == "DipE" or pert_type == "FullOR_V" or pert_type == "FullOR_L"):
    # For the length gauge electric dipole, for the magnetic dipole
    # and electric quadrupole, we need the translation vector and to
    # form the U matrix
    #
    OrbE = []
    DE = []
    for k in range(Nkp):
      OrbE.append(np.diag(Fock[k].real))
      DE.append(DEk(NOrb2,OrbE[k]))
    for dir_dk in range(npdir):
      # dF/dk = F'
      FockDk = getFock1k(mol_inp,O,V,NB,ipbc,"MO",True,dir_dk,MOCoef)
      #
      # dS/dk = S'
      OvlDk = getOvl1k(mol_inp,O,V,NB,ipbc,"MO",True,dir_dk,MOCoef)
      #
      # Form i(U + 1/2S') or iU or both
      UMat.append(FockDk - np.einsum('kij,kj->kij',OvlDk,OrbE,optimize=True))
      UMat[dir_dk] /= -np.array(DE)
      if(pert_type == "DipE"):
        UMat[dir_dk] += 0.5*OvlDk
        # The diagonal of this matrix is 0
        ind = np.arange(NOrb2)
        UMat[dir_dk][:,ind,ind] = 0
      elif(pert_type == "FullOR_V"):
        # The diagonal of this matrix is -S'/2
        ind = np.arange(NOrb2)
        UMat[dir_dk][:,ind,ind] = -OvlDk[:,ind,ind]/2
      elif(pert_type == "FullOR_L"):
        # We need both matrices
        UMatS.append(UMat[dir_dk] + 0.5*OvlDk)
        ind = np.arange(NOrb2)
        UMatS[dir_dk][:,ind,ind] = 0
        UMatS[dir_dk] = UMatS[dir_dk]*1j
        UMat[dir_dk][:,ind,ind] = -OvlDk[:,ind,ind]/2
      UMat[dir_dk] = UMat[dir_dk]*1j
    del FockDk, OvlDk, OrbE, DE
  # Now form the perturbation matrices in MO(k) basis
  X_ij = np.zeros((NP,Nkp,O2,O2),dtype=complex)
  X_ia = np.zeros((NP,Nkp,O2,V2),dtype=complex)
  X_ab = np.zeros((NP,Nkp,V2,V2),dtype=complex)
  AOPert = AOPert.reshape((NP,nmtpbc,ntt))
  save_p = []
  if(pert_type == "FullOR_V" or pert_type == "FullOR_L"):
    save_p = np.zeros((NP1,Nkp,NOrb2,NOrb2),dtype=complex)
  for n in range (NP):
    Pert_k_lt = fourier("Dir",ipbc,AOPert[n,:,:],False,0)
    if(pert_type == "DipE"):
      # Electric dipole length gauge
      PertA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Pert_k_lt)
    elif(pert_type == "DipEV" or pert_type == "FullOR_V"):
      # Electric dipole velocity gauge
      PertA = basis_tran("Dir",True,False,"AHer",NB,Nkp,MOCoef,Pert_k_lt)
    elif(pert_type == "FullOR_L"):
      if(n >= NP1 and n <NP1+NP2):
        # mu(L)
        PertA = basis_tran("Dir",True,False,"Herm",NB,Nkp,MOCoef,Pert_k_lt)
      else:
        # m or Theta(V) or mu(V)
        PertA = basis_tran("Dir",True,False,"AHer",NB,Nkp,MOCoef,Pert_k_lt)
    Pert = np.zeros((Nkp,NOrb2,NOrb2),dtype=complex)
    for k in range(Nkp):
      # Fill out the alpha and beta blocks
      # oa-oa
      Pert[k,:O,:O] = PertA[k,:O,:O]
      # ob-ob
      Pert[k,O:2*O,O:2*O] = PertA[k,:O,:O]
      # va-va
      Pert[k,2*O:2*O+V,2*O:2*O+V] = PertA[k,O:,O:]
      # vb-vb
      Pert[k,2*O+V:,2*O+V:] = PertA[k,O:,O:]
      # oa-va
      Pert[k,:O,2*O:2*O+V] = PertA[k,:O,O:]
      # va-oa
      Pert[k,2*O:2*O+V,:O] = PertA[k,O:,:O]
      # ob-vb
      Pert[k,O:2*O,2*O+V:] = PertA[k,:O,O:]
      # vb-ob
      Pert[k,2*O+V:,O:2*O] = PertA[k,O:,:O]
    if(pert_type == "DipE"):
      # Electric dipole length gauge
      # Add UMat contribution
      for dir_dk in range(npdir):
        Pert -= UMat[dir_dk]*tv[dir_dk][n]
    elif(pert_type == "FullOR_V"):
      # Magnetic dipole and electric quadrupole
      # Add UMat contribution
      if(n < NP1):
        # Save electric dipole integrals in temporary array
        save_p[n] = np.copy(Pert)
      elif(n < NP1+NP2):
        # Correct magnetic dipole
        nn0 = n - NP1
        nn1 = (nn0+1)%3 
        nn2 = (nn0+2)%3
        for dir_dk in range(npdir):
          temp = -tv[dir_dk][nn2]*np.einsum('hij,hjk->hik',save_p[nn1],
                                            UMat[dir_dk],optimize=True)
          temp += tv[dir_dk][nn1]*np.einsum('hij,hjk->hik',save_p[nn2],
                                            UMat[dir_dk],optimize=True)
          temp = (temp - np.conjugate(np.transpose(temp,axes=(0,2,1))))/2
          # Pert = np.copy(temp)
          Pert -= temp
          del temp
      else:
        # Correct electric quadrupole
        nn0 = n - (NP1+NP2)
        if(nn0 < 3):
          nn1 = nn0
          nn2 = nn0
          for dir_dk in range(npdir):
            temp = 2*tv[dir_dk][nn2]*np.einsum('hij,hjk->hik',save_p[nn1],
                                               UMat[dir_dk],optimize=True)
            temp = (temp - np.conjugate(np.transpose(temp,axes=(0,2,1))))/2
            Pert -= temp
            del temp
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
          for dir_dk in range(npdir):
            temp = tv[dir_dk][nn2]*np.einsum('hij,hjk->hik',save_p[nn1],
                                             UMat[dir_dk],optimize=True)
            temp += tv[dir_dk][nn1]*np.einsum('hij,hjk->hik',save_p[nn2],
                                              UMat[dir_dk],optimize=True)
            temp = (temp - np.conjugate(np.transpose(temp,axes=(0,2,1))))/2
            Pert -= temp
            del temp
    elif(pert_type == "FullOR_L"):
      # Add UMat contributions to mu(L), m, and Theta(V)
      if(n < NP1):
        # Save electric dipole integrals in temporary array
        save_p[n] = np.copy(Pert)
      elif(n < NP1+NP2):
        # mu(L)
        for dir_dk in range(npdir):
          Pert -= UMatS[dir_dk]*tv[dir_dk][n-NP1]
      elif(n < NP1+NP2+NP4):
        # Correct magnetic dipole
        nn0 = n - (NP1+NP2)
        nn1 = (nn0+1)%3 
        nn2 = (nn0+2)%3
        for dir_dk in range(npdir):
          temp = -tv[dir_dk][nn2]*np.einsum('hij,hjk->hik',save_p[nn1],
                                            UMat[dir_dk],optimize=True)
          temp += tv[dir_dk][nn1]*np.einsum('hij,hjk->hik',save_p[nn2],
                                            UMat[dir_dk],optimize=True)
          temp = (temp - np.conjugate(np.transpose(temp,axes=(0,2,1))))/2
          Pert -= temp
          del temp
      else:
        # Correct electric quadrupole
        nn0 = n - (NP1+NP2+NP4)
        if(nn0 < 3):
          nn1 = nn0
          nn2 = nn0
          for dir_dk in range(npdir):
            temp = 2*tv[dir_dk][nn2]*np.einsum('hij,hjk->hik',save_p[nn1],
                                               UMat[dir_dk],optimize=True)
            temp = (temp - np.conjugate(np.transpose(temp,axes=(0,2,1))))/2
            Pert -= temp
            del temp
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
          for dir_dk in range(npdir):
            temp = tv[dir_dk][nn2]*np.einsum('hij,hjk->hik',save_p[nn1],
                                             UMat[dir_dk],optimize=True)
            temp += tv[dir_dk][nn1]*np.einsum('hij,hjk->hik',save_p[nn2],
                                              UMat[dir_dk],optimize=True)
            temp = (temp - np.conjugate(np.transpose(temp,axes=(0,2,1))))/2
            Pert -= temp
            del temp
    for k in range(Nkp):
      prod = np.einsum('ij,ij',Pert[k],np.conjugate(Pert[k]),optimize=True)
      print(f"Kp:{k+1}, Pert-{n+1}: {prod}")
      X_ij[n,k,:,:] = Pert[k,:O2,:O2]
      X_ia[n,k,:,:] = Pert[k,:O2,O2:]
      X_ab[n,k,:,:] = Pert[k,O2:,O2:]
  del UMat, UMatS
  if(pert_type == "FullOR_L"):
    # First, reorder perturbations and put mu(V) last
    X_ij = np.roll(X_ij,-3,axis=0)      
    X_ia = np.roll(X_ia,-3,axis=0)      
    X_ab = np.roll(X_ab,-3,axis=0)
    for n in range(NP1,NP):
      X_ij[n] = np.transpose(X_ij[n],axes=(0,2,1))
      X_ab[n] = np.transpose(X_ab[n],axes=(0,2,1))
  elif(pert_type == "DipEV" or pert_type == "FullOR_V"):
    X_ij = np.transpose(X_ij,axes=(0,1,3,2))
    X_ab = np.transpose(X_ab,axes=(0,1,3,2))
  # if(pert_type == "DipE"):
  #   X_ij = np.transpose(X_ij,axes=(0,1,3,2,4))
  # elif(pert_type == "DipEV" or pert_type == "FullOR_V"):
  #   X_ij = np.transpose(X_ij,axes=(0,2,4,1,3))
  # elif(pert_type == "FullOR_L"):
  #   X_ij0 = np.zeros((NP,Nkp,O2,Nkp,O2),dtype=X_ij.dtype)
  #   for n in range(NP):
  #     if(n<NP1):
  #       # mu(L)
  #       X_ij0[n,:,:,:,:] = np.transpose(X_ij[n,:,:,:,:],axes=(0,2,1,3))
  #     else:
  #       # m, Theta(V), mu(V)
  #       X_ij0[n,:,:,:,:] = np.transpose(X_ij[n,:,:,:,:],axes=(1,3,0,2))
  #   X_ij = X_ij0
  #   del X_ij0
  # else:
  #   print(f"getPert is confused about X_ij")
  #   exit()
  # X_ij = X_ij.reshape((NP,Nkp,O2,O2))
  # X_ia = np.transpose(X_ia,axes=(0,1,3,2,4))
  # X_ia = X_ia.reshape((NP,Nkp,O2,V2))
  # if(pert_type == "DipE"):
  #   X_ab = np.transpose(X_ab,axes=(0,1,3,2,4))
  # elif(pert_type == "DipEV" or pert_type == "FullOR_V"):
  #   X_ab = np.transpose(X_ab,axes=(0,2,4,1,3))
  # elif(pert_type == "FullOR_L"):
  #   X_ab0 = np.zeros((NP,Nkp,V2,Nkp,V2),dtype=X_ab.dtype)
  #   for n in range(NP):
  #     if(n<NP1):
  #       # mu(L)
  #       X_ab0[n,:,:,:,:] = np.transpose(X_ab[n,:,:,:,:],axes=(0,2,1,3))
  #     else:
  #       # m, Theta(V), mu(V)
  #       X_ab0[n,:,:,:,:] = np.transpose(X_ab[n,:,:,:,:],axes=(1,3,0,2))
  #   X_ab = X_ab0
  #   del X_ab0
  # else:
  #   print(f"getPert is confused about X_ab")
  #   exit()
  # X_ab = X_ab.reshape((NP,V2k,V2k))
  X_ia = np.conjugate(X_ia)
  if(pert_type == "DipE"):
    X_ij = np.conjugate(X_ij)
    X_ab = np.conjugate(X_ab)
  elif(pert_type == "FullOR_L"):
    for n in range(NP1):
      # mu(L)
      X_ij[n] = np.conjugate(X_ij[n])
      X_ab[n] = np.conjugate(X_ab[n])
  for n in range(NP):
    prodXij = np.einsum('Iij,Iij',X_ij[n],np.conjugate(X_ij[n]),optimize=True)
    prodXab = np.einsum('Iij,Iij',X_ab[n],np.conjugate(X_ab[n]),optimize=True)
    prodXia = np.einsum('Iij,Iij',X_ia[n],np.conjugate(X_ia[n]),optimize=True)
    print(f"Pert:{n+1}, Xij: {prodXij}, Xab: {prodXab}, Xia: {prodXia}")
  return NP, NP1, NP2, NP3, NP4, X_ij, X_ia, X_ab
