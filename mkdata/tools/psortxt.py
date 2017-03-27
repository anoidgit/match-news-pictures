#encoding: utf-8

import sys

import os

def ldfile(fname):
	rs=[]
	with open(fname) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8").split(" ")
				rs.extend(tmp)
	return " ".join(rs), len(rs)

def handle(srcp,rsf,rsm,distf):
	rsd={}
	nfile=0
	for root,dirs,files in os.walk(srcp):
		for file in files:
			content,l=ldfile(os.path.join(root,file))
			if l in rsd:
				rsd[l].append((file,content))
			else:
				rsd[l]=[(file,content)]
			nfile+=1
	l=rsd.keys()
	l.sort(reverse=True)
	nline=0
	with open(rsf,"w") as fwrtc:
		with open(rsm,"w") as fwrtm:
			with open(distf,"w") as fwrtd:
				for lu in l:
					curl=rsd[lu]
					ncont=len(curl)
					nline+=ncont
					tmp=" ".join([str(lu),str(ncont)])
					fwrtd.write(tmp.encode("utf-8"))
					fwrtd.write("\n".encode("utf-8"))
					for fname,content in curl:
						fwrtc.write(content.encode("utf-8"))
						fwrtc.write("\n".encode("utf-8"))
						fwrtm.write(fname.encode("utf-8"))
						fwrtm.write("\n".encode("utf-8"))
	if nline==nfile:
		print("Check passed")
	else:
		print("Check fail")

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"),sys.argv[4].decode("utf-8"))
