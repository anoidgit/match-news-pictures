#encoding: utf-8

import sys
import h5py
import cv2
import numpy

import os

def ldpl(fname):
	rs=[]
	with open(fname) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				rs.append(tmp.decode("utf-8"))
	return rs

def ldpic(fname):
	try:
		img=cv2.imread(fname)
	except:
		return False,False
	if img is None:
		return False,False
	else:
		return cv2.resize(img,(224,224)),True

def handle(src,rsf,srcp):
	pl=ldpl(src)
	h5f=h5py.File(rsf,"w")
	curid=1
	fl=[]
	for pu in pl:
		tmp,flag=ldpic(srcp+pu)
		if flag:
			h5f[str(curid)]=numpy.moveaxis(tmp,-1,0)
			curid+=1
		else:
			fl.append(pu)
	h5f.close()
	if fl:
		print(len(fl)),
		print("files does not exist")
		with open("errf.txt","w") as fwrt:
			fwrt.write("\n".join(fl).encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"))
