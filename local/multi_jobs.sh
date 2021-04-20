#!/bin/bash
set -e

lst=$1
exec=$2
prefix=$3
nj=$4


time=`date +'%Y-%m-%d-%H-%M-%S'`
mj=tmp_multi_jobs-$time

[ ! -s $lst ] && echo "$0 empth file: $lst" && exit 1
[ ! -s $exec ] && echo "$0 empth file: $exec" && exit 1
[ -z $nj ] && nj=1

[ -d $mj ] &&  rm -rf $mj 
[ ! -d $mj ] &&  mkdir -p $mj 

n=`cat $lst | wc -l`
l=`expr $n / $nj`
split -l $l $lst $mj/ 

for job in $mj/*;do
	{
		cat $job | while read line
		do
			$prefix $exec $line || continue
		done
	}&
done

wait && rm -r $mj && exit 0
