#encoding: utf-8

import sys

def ldwd(fname):
	rs=set()
	with open(fname) as frd:
		for line in frd:
			tmp=line.strip().decode("utf-8")
			if tmp and not tmp in rs:
				rs.add(tmp)
	return rs

def filine(l,f):
	rs=[]
	for lu in l:
		if lu and lu in f:
			rs.append(lu)
	return " ".join(rs)

def handle(srcf,rsf,wdf):
	wd=ldwd(wdf)
	with open(rsf,"w") as fwrt:
		with open(srcf) as frd:
			for line in frd:
				tmp=line.strip()
				if tmp:
					tmp=filine(tmp.decode("utf-8").split(" "),wd)
					fwrt.write(tmp.encode("utf-8"))
					fwrt.write("\n".encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"))
