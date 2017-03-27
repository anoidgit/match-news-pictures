#encoding: utf-8

import sys

def splitwt(strin):
	ind=strin.rfind("/")
	if ind>0:
		return strin[:ind],strin[ind+1:]
	else:
		return strin,""

def linefilter(strin,keepset):
	rs=[]
	tmp=strin.split(" ")
	for tmpu in tmp:
		wd,tag=splitwt(tmpu)
		if tag.startswith("n") or tag.startswith("s") or tag.startswith("b") or tag in keepset:
			rs.append(wd)
	return rs

def filefilter(srcfile,rsfile):
	keepset=set(["v","vd","vn","vi","vg","a","ad","an","ag"])
	rs=[]
	with open(srcfile,"rb") as f:
		for line in f:
			tmp=line.strip()
			if tmp:
				tmp=linefilter(tmp.decode("utf-8","ignore"),keepset)
				if tmp:
					rs.extend(tmp)
	tmp=" ".join(rs).strip()
	with open(rsfile,"wb") as f:
		f.write(tmp.encode("utf-8"))

if __name__=="__main__":
	filefilter(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"))
