set -e

file=$1

trans=`cat $file`
accent=`echo $file | cut -d"/" -f5`
fn=`basename $file .txt`
echo ${accent}_${fn} $trans

