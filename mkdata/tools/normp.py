#encoding: utf-8

import sys
import h5py
import numpy

def handle(src,avg,rs):
	h5f=h5py.File(avg,"r")
	avgm=h5f["avg"][:]
	stdm=h5f["std"][:]
	h5f.close()
	avgm=avgm.astype("float32")
	stdm=stdm.astype("float32")
	srcf=h5py.File(src,"r")
	rsf=h5py.File(rs,"w")
	for lu in srcf.keys():
		rsf[lu]=(srcf[lu][:].astype("float32")-avgm)/stdm
	srcf.close()
	rsf.close()

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"))
