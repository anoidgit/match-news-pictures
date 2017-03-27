#encoding: utf-8

import sys

def handle(src,rsf):
	rsd={}
	with open(src) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8").split(" ")
				for tmpu in tmp:
					tt=tmpu.strip()
					if tt:
						rsd[tt]=rsd.get(tt,0)+1
	rs=[]
	for k,v in rsd.iteritems():
		if v>1:
			rs.append(k)
	tmp="\n".join(rs)
	with open(rsf,"w") as fwrt:
		fwrt.write(tmp.encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"))
