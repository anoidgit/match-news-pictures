#encoding: utf-8

import sys
import h5py
import numpy

def handle(src,rs):
	sumat=numpy.zeros((3,224,224))
	h5f=h5py.File(src,"r")
	inds=h5f.keys()
	for ind in inds:
		sumat+=h5f[ind]
	sumat/=float(len(inds))
	h5f.close()
	h5f=h5py.File(rs,"w")
	h5f["avg"]=sumat
	h5f.close()

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"))
