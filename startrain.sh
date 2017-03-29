#!/bin/bash

logfile=170329_0.5l.log

logdir=logs

th train.lua 2>&1 | tee $logdir/$logfile
