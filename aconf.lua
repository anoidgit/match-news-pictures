starterate=math.huge--warning:only used as init erate, not asigned to criterion

require "dset"

ieps=1
warmcycle=4
expdecaycycle=4
gtraincycle=64

modlr=1/8192

earlystop=math.ceil(gtraincycle/2)

csave=3

lrdecaycycle=4

recyclemem=0.05
