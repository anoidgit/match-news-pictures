require "hdf5"

traint=hdf5.open("datasrc/traint.hdf5","r")
trainp=hdf5.open("datasrc/trainp.hdf5","r")

devt=hdf5.open("datasrc/devt.hdf5","r")
devp=hdf5.open("datasrc/devp.hdf5","r")
