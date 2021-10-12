#!/bin/bash

# Example script for the Othello assignment. Don't forget to make it executable (chmod +x othello)
# Change the last line (python3 Othello.py ...) if you are using python2 
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

# only run if <do_compile> is not set (we don't need to compile Python code, but the automated testing needs this)
if [ $do_compile -ne 1 ]; then
	# Call your Python program with a position and time limit
	python3 Othello.py $position $time_limit
fi
