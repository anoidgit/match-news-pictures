#encoding: utf-8

import sys
import subprocess
import threading
from time import sleep

import os

nthread=8
sleeptime=1
plock=threading.Lock()
printlock=threading.Lock()

def mulprint(strp):
	global printlock
	with printlock:
		print(strp)

def createPool(srcp,rsp):
	srcl=[]
	rsl=[]
	for root, dirs, files in os.walk(srcp):
		for file in files:
			srcf=os.path.join(root,file)
			srcl.append(srcf)
			rsl.append(srcf.replace(srcp,rsp))
	return srcl, rsl

def get():
	global plock,srcp,rsp
	rstate=False
	with plock:
		if len(srcp)>0:
			rstate=True
			rsrc=srcp.pop()
			rs=rsp.pop()
	if not rstate:
		rsrc=""
		rs=""
	return rsrc,rs,rstate

def oneseg(srcf,rsf):
	mulprint("handle:"+srcf+",to:"+rsf)
	subprocess.call(" ".join(('python tools/weightfilter.py',srcf,rsf)),shell=True)

def core():
	global nthread
	state=True
	while state:
		srcf,rsf,state=get()
		if state:
			oneseg(srcf,rsf)
		else:
			nthread-=1

if __name__=="__main__":
	srcp,rsp=createPool(sys.argv[1].decode("utf-8"),sys.argv[-1].decode("utf-8"))
	tpool=[]
	for i in xrange(nthread):
		t=threading.Thread(target=core)
		t.setDaemon(True)
		t.start()
		tpool.append(t)
	while nthread>0:
		sleep(sleeptime)
	del tpool
	del srcp
	del rsp
