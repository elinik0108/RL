#!/bin/bash

# Example script for the Othello assignment. Don't forget to make it executable (chmod +x othello)
# Change the last line (java Othello ...) to suit your needs
#
# usage: bash othello <position> <time_limit> <do_compile>
# 
# Author: Ola Ringdahl

position=$1
time_limit=$2
do_compile=$3 

if [ "$#" -ne 3 ]; then
	# do_compile not set (not enough input arguments)
	do_compile=0
fi

# Change directory to the location of your program 
# $(dirname "$0") is the path to where this script is located (don't change this) 
cd "$(dirname "$0")"

if [ $do_compile -eq 1 ]; then
	# Compile the code:
	javac *.java
else
	# Call your Java program with a position and time limit:
	java Othello $position $time_limit
fi
