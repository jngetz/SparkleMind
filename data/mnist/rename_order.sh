k=1
for i in *.jpg; do
    new=$(printf "%06d.jpg" "$k")
    mv -i -- "$i" "$new"
    let k=k+1
done
