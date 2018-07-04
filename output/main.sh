for name in `ls *png`
do
  num=${name##*_}
  num=${num%.*}
  num=${num##*0}
  mv $name `printf "output_%04d.png" $num`
done
