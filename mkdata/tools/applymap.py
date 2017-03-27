#encoding: utf-8

import sys

def ldmap(mapf):
	rsd={}
	with open(mapf) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8")
				pic,txt=tmp.split("	")
				rsd[txt]=pic
	return rsd

def handle(srcf,rsf,mapf):
	mapd=ldmap(mapf)
	with open(srcf) as frd:
		with open(rsf,"w") as fwrt:
			for line in frd:
				tmp=line.strip()
				if tmp:
					tmp=mapd[tmp.decode("utf-8")]
					fwrt.write(tmp.encode("utf-8"))
					fwrt.write("\n".encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"))
