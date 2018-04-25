out=imglist.txt
(rm $out)
(touch $out)
for file in *.jpg; do
    p="$(pwd)/$file"
    (echo "$p" >> $out)
done
