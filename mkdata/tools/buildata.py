#encoding: utf-8

import sys
import h5py,numpy

def padmat(l2,padv):
	slen=0
	for lu in l2:
		tmp=len(lu)
		if tmp>slen:
			slen=tmp
	for lu in l2:
		tmp=slen-len(lu)
		if tmp>0:
			for i in xrange(tmp):
				lu.append(padv)
	return l2

def handle(srcpf,rspf,srctf,rstf,splf,padv):
	spf=h5py.File(srcpf,"r")
	rpf=h5py.File(rspf,"w")
	rtf=h5py.File(rstf,"w")
	curpid=1
	cuwid=1
	with open(srctf) as frd:
		with open(splf) as spl:
			for line in spl:
				tmp=line.strip()
				if tmp:
					tmp=int(tmp.decode("utf-8"))
					rp=numpy.zeros((tmp,3,224,224),dtype=numpy.float32)
					for i in xrange(tmp):
						rp[i]=spf[str(curpid)][:]
						curpid+=1
					wrtkey=str(cuwid)
					rpf[wrtkey]=rp
					td=[]
					for i in xrange(tmp):
						lind=frd.readline().strip().decode("utf-8").split(" ")
						td.append([int(linu) for linu in lind])
					td=numpy.array(padmat(td,padv),dtype=long).T
					rtf[wrtkey]=td
					cuwid+=1
	rpf.close()
	rtf.close()
	spf.close()

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"),sys.argv[4].decode("utf-8"),sys.argv[5].decode("utf-8"),int(sys.argv[6].decode("utf-8")))
