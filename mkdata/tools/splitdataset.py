#encoding: utf-8

import sys
from random import sample
import h5py

def handle(srcp,srct,tp,tt,dp,dt):
	spf=h5py.File(srcp,"r")
	tpf=h5py.File(tp,"w")
	dpf=h5py.File(dp,"w")
	t=len(spf.keys())
	dinds=set(sample([i for i in xrange(1,t+1)],t/20))
	curid=1
	curt=1
	curd=1
	with open(srct) as frd:
		with open(tt,"w") as fwrt:
			with open(dt,"w") as fwrd:
				for line in frd:
					ckey=str(curid)
					if curid in dinds:
						fwrd.write(line)
						dpf[str(curd)]=spf[ckey][:]
						curd+=1
					else:
						fwrt.write(line)
						tpf[str(curt)]=spf[ckey][:]
						curt+=1
					curid+=1
	spf.close()
	tpf.close()
	dpf.close()
	print("total:"+str(curid-1)+",train:"+str(curt-1)+",dev:"+str(curd-1))

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"),sys.argv[4].decode("utf-8"),sys.argv[5].decode("utf-8"),sys.argv[6].decode("utf-8"))
