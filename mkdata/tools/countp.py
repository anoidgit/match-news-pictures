#encoding: utf-8

import sys

import os

def countp(paths):
	fdone=set()
	rsd={}
	addflag=True
	for pth in paths:
		for root,dirs,files in os.walk(pth):
			for file in files:
				curfile=os.path.join(root,file)
				if not curfile in fdone:
					fdone.add(curfile)
					with open(curfile) as frd:
						for line in frd:
							tmp=line.strip()
							if tmp:
								tmp=tmp.decode("utf-8").split(" ")
								for tmpu in tmp:
									if addflag:
										rsd[tmpu]=rsd.get(tmpu,0)+1
									else:
										if tmpu in rsd:
											rsd[tmpu]+=1
		addflag=False
	return rsd

def trand(din):
	rs={}
	for k,v in din.iteritems():
		if v in rs:
			rs[v].append(k)
		else:
			rs[v]=[k]
	return rs

def saved(din,fname):
	f=din.keys()
	f.sort(reverse=True)
	rs=[]
	for ku in f:
		tmp=din[ku]
		tmp.insert(0,str(ku))
		rs.append(tmp)
	tmp="\n".join(" ".join(tmpu) for tmpu in rs)
	with open(fname,"w") as fwrt:
		fwrt.write(tmp.encode("utf-8"))

def tranarg(argv):
	rs=[]
	for argu in argv:
		rs.append(argu.decode("utf-8"))
	return rs

if __name__=="__main__":
	saved(trand(countp(tranarg(sys.argv[2:]))), sys.argv[1].decode("utf-8"))
