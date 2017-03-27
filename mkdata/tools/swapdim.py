#encoding: utf-8

import sys
import numpy
import h5py

def handle(src,rs):
	srcf=h5py.File(src,"r")
	rsf=h5py.File(rs,"w")
	for lu in srcf.keys():
		rsf[lu]=numpy.moveaxis(srcf[lu],-1,0)
	rsf.close()
	srcf.close()

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"))
