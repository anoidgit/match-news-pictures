#encoding: utf-8

import sys

def ldidf(fname):
	rs={}
	with open(fname) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8").split(" ")
				weight=float(tmp[0])
				for wu in tmp[1:]:
					rs[wu]=weight
	return rs

def ldsrc(fname,dkeep):
	rs={}
	with open(fname) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8").split(" ")
				for tmpu in tmp:
					if tmpu in dkeep:
						rs[tmpu]=rs.get(tmpu,0)+1
	return rs

def mktfidf(count,idf):
	rs={}
	sum=0
	for v in count.values():
		sum+=v
	for k, c in count.iteritems():
		rs[k]=float(c)/sum*idf[k]
	return rs

def handle(srcfile,rsfile):
	w=ldidf("tools/ref_weight/idf.txt")
	w=mktfidf(ldsrc(srcfile,w),w)
	rs={}
	for k,v in w.iteritems():
		if v in rs:
			rs[v].append(k)
		else:
			rs[v]=[k]
	w=rs.keys()
	w.sort()
	rsl=[]
	for wu in w:
		rsl.extend(rs[wu])
	rs=" ".join(rsl)
	with open(rsfile,"w") as fwrt:
		fwrt.write(rs.encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"))
