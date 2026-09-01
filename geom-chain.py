import sys
import os
import numpy as np
from scipy.constants import angstrom, physical_constants, h, c
import importlib
from pathlib import Path
np.set_printoptions(precision=16,threshold=sys.maxsize,floatmode='fixed')
# geom_unit = [[-0.6,0.0,0.0],[0.6,0.0,0.0],[-1.1,0.5,-0.5],[1.1,0.5,0.5]]
# atom_list = ["O","O","H","H"]
# Tv = [0.0,0.0,4.0]
# units = 14
# increment = 0.5
# max_incr = 17
input_file = sys.argv[1]
file_name = importlib.import_module(input_file)
mol_inp = file_name.molecule
files = Path(f"{mol_inp}_txts")
Tv = file_name.tv
# Gauopen path
path_gauopen = file_name.path_gauopen
sys.path.insert(0,f"{path_gauopen}")
# from gauopen import QCOpMat as qco
from gauopen import QCBinAr as qcb
baf = qcb.QCBinAr(file=f"{mol_inp}.baf")
atom_list = baf.ian
geometry = baf.c.reshape((len(atom_list),3))
bohr_radius = physical_constants["Bohr radius"][0]
toAng = bohr_radius/angstrom 
geometry *= toAng
ipbc = baf.matlist["File 733 Integers"].array
npdir = ipbc[0]
nmtpbc = ipbc[1]
nrecip = ipbc[9]
# Read cell list from ipbc
l_listall = np.array(ipbc[21:]).reshape((nmtpbc,3))
os.system(f"rm {mol_inp}_geom")
geom = ""
print(f"geometry: \n {geometry}\n")
for a in range(len(atom_list)):
  print(f"{atom_list[a]} {geometry[a]}\n")
if(npdir == 1):
  l_list = l_listall[:,0].reshape((1,nmtpbc))
  for n in range(nmtpbc):
    addline = l_list[0][n]*np.array(Tv[0]) 
    for a in range(len(atom_list)):
      line = np.array(geometry[a]) + addline
      geom += (f"{atom_list[a]} {line[0]:10.8f} {line[1]:10.8f} {line[2]:10.8f}\n")
elif(npdir == 2):
  l_list = l_listall[:,:2].reshape((nmtpbc,2))
  l_list = np.transpose(l_list,axes=(1,0))
  for n in range(nmtpbc):
    addline = l_list[0][n]*np.array(Tv[0]) + l_list[1][n]*np.array(Tv[1])
    print(f"Mat={n}, lists: {l_list[0][n]} - {l_list[1][n]}")
    print(f"Tv: {Tv[0]} - {Tv[1]}, addline : {addline}")
    for a in range(len(atom_list)):
      line = np.array(geometry[a]) + addline
      geom += (f"{atom_list[a]} {line[0]:10.8f} {line[1]:10.8f} {line[2]:10.8f}\n")
else:
  l_list = np.transpose(l_listall,axes=(1,0))
  for n in range(nmtpbc):
    addline = l_list[0][n]*np.array(Tv[0]) + l_list[1][n]*np.array(Tv[1]) + l_list[2][n]*np.array(Tv[2])
    for a in range(len(atom_list)):
      line = np.array(geometry[a]) + addline
      geom += (f"{atom_list[a]} {line[0]:10.8f} {line[1]:10.8f} {line[2]:10.8f}\n")
with open(f"{mol_inp}_geom","a") as writer: writer.write(f"{geom}")
    
# for i in range(max_incr):
#   TvNew = [0.0,0.0,0.0]
#   for j in range(3):
#     if(Tv[j] != 0): TvNew[j] = Tv[j] + i*increment
#   geom = ""
#   for line in range(len(atom_list)):
#     line_0 = np.array(geom_unit[line])
#     geom += (f"{atom_list[line]} {line_0[0]:10.8f} {line_0[1]:10.8f} {line_0[2]:10.8f}\n")
#   for cell in range(units):
#     geom_p = ""
#     geom_m = ""
#     for line in range(len(atom_list)):
#       line_p = np.array(geom_unit[line]) + (cell+1)*np.array(TvNew)
#       line_m = np.array(geom_unit[line]) - (cell+1)*np.array(TvNew)
#       geom_p += (f"{atom_list[line]} {line_p[0]:10.8f} {line_p[1]:10.8f} {line_p[2]:10.8f}\n")
#       geom_m += (f"{atom_list[line]} {line_m[0]:10.8f} {line_m[1]:10.8f} {line_m[2]:10.8f}\n")
#     geom += geom_p
#     geom += geom_m
#   # print(f"{geom}\n\n")
#   os.system(f"rm H2O2-{i}-z")
#   with open(f"H2O2-{i}-z","a") as writer: writer.write(f"{geom}")

