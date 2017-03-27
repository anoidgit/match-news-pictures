#encoding: utf-8

import sys

def ldmap(fname):
	rsd={}
	curid=1
	with open(fname) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8").split(" ")[1:]
				for tmpu in tmp:
					if tmpu and not tmpu in rsd:
						rsd[tmpu]=str(curid)
						curid+=1
	return rsd

def reportnwd(fname):
	rsd=set()
	count=0
	with open(fname) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8").split(" ")[1:]
				for tmpu in tmp:
					if tmpu and not tmpu in rsd:
						count+=1
	return count

def applymap(sl,mapd):
	rs=[]
	for su in sl:
		if su in mapd:
			rs.append(mapd[su])
		else:
			print("Warning:word "+su+" not contained")
	return " ".join(rs)

def handle(srcf,mapf,rsf):
	mapd=ldmap(mapf)
	with open(rsf,"w") as fwrt:
		with open(srcf) as frd:
			for line in frd:
				tmp=line.strip()
				if tmp:
					tmp=applymap(tmp.decode("utf-8").split(" "),mapd)
					fwrt.write(tmp.encode("utf-8"))
					fwrt.write("\n".encode("utf-8"))

if __name__=="__main__":
	if len(sys.argv)>3:
		handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"))
	else:
		print(reportnwd(sys.argv[1].decode("utf-8")))
