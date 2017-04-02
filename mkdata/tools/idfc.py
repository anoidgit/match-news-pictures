#encoding: utf-8

import sys

from math import log

def ldsrc(fname):
	rs=[]
	sum=0
	with open(fname) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8")
				ind=tmp.find(" ")
				frq=int(tmp[:ind])
				if frq>5:
					wds=tmp[ind+1:]
					sum+=frq*len(wds.split(" "))
					rs.append((frq,wds))
	return rs,sum

def handle(srcfile,rsfile):
	fd, sum=ldsrc(srcfile)
	sum=float(sum)
	rs=[]
	for f, ws in fd:
		rs.append(" ".join((str(log(sum/f)),ws)))
	rs="\n".join(rs)
	with open(rsfile,"w") as fwrt:
		fwrt.write(rs.encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"))
