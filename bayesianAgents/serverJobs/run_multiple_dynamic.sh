for n_participants in 100
do
    for temp in 0.5 1.0 3.0 5.0
    do
        echo "$n_participants|$temp"
        sbatch ./runjob_dynamic.sh $n_participants $temp
    done
done