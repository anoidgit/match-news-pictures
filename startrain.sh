#!/bin/bash

logfile=170402_0.5lanw.log

logdir=logs

th train.lua 2>&1 | tee $logdir/$logfile
