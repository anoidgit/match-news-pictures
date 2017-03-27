#encoding: utf-8

import sys

def ldfbd(fname):
	rs=set()
	with open(fname) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8")
				if not tmp in rs:
					rs.add(tmp)
	return rs

def handle(srcp,srcd,fbd,rs):
	drop=ldfbd(fbd)
	with open(rs,"w") as fwrt:
		with open(srcp) as frdp:
			with open(srcd) as frdd:
				for p,d in zip(frdp,frdd):
					p=p.strip().decode("utf-8")
					if not p in drop:
						fwrt.write(d)

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"),sys.argv[4].decode("utf-8"))
