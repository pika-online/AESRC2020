#!/bin/bash
set -e

src=$1
form=$2
mode=$3

if [ -z $mode ];then
	mode="lst"
fi

dir=`cd $src;pwd`
if [ $mode == "lst" ];then
	find $dir -name "*.$form" 
elif [ $mode == "scp" ]; then
	find $dir -name "*.$form" | awk -F '/' '{a=$NF;sub(/\..*$/,"",a);printf("%s %s\n",a,$0)}' 
else
	echo "ERROR:Specify lst/scp Please"
	exit 1
fi
exit 0