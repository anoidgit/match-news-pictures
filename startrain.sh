#!/bin/bash

logfile=170331_0.5l.log

logdir=logs

th train.lua 2>&1 | tee $logdir/$logfile
