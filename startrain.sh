#!/bin/bash

logfile=170331_0.5lstd.log

logdir=logs

th train.lua 2>&1 | tee $logdir/$logfile
