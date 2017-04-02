#encoding: utf-8

import sys
import h5py,numpy

from random import shuffle

def padmat(l2,padv):
	slen=0
	rs=[]
	for lu in l2:
		tmp=len(lu)
		if tmp>slen:
			slen=tmp
	for lu in l2:
		tmp=slen-len(lu)
		if tmp>0:
			tmpl=[padv for i in xrange(tmp)]
			tmpl.extend(lu)
			rs.append(tmpl)
			#lu.extend([padv for i in xrange(tmp)])
		else:
			rs.append(lu)
	return rs

def shufflepair(srcm1, srcm2):
	bsize=srcm1.shape[0]
	if bsize == 1:
		return srcm1, srcm2
	else:
		rind = [i for i in xrange(bsize)]
		shuffle(rind)
		rsm1 = srcm1.copy()
		rsm2 = srcm2.copy()
		curid = 0
		for ru in rind:
			rsm1[curid] = srcm1[ru]
			rsm2[curid] = srcm2[ru]
			curid+=1
		return rsm1, rsm2

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
					td=[]
					for i in xrange(tmp):
						lind=frd.readline().strip().decode("utf-8").split(" ")
						td.append([int(linu) for linu in lind])
					td=numpy.array(padmat(td,padv),dtype=long).T
					rp, td=shufflepair(rp,td)
					wrtkey=str(cuwid)
					rpf[wrtkey]=rp
					rtf[wrtkey]=td
					cuwid+=1
	rpf.close()
	rtf.close()
	spf.close()

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"),sys.argv[4].decode("utf-8"),sys.argv[5].decode("utf-8"),int(sys.argv[6].decode("utf-8")))
