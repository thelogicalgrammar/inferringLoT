for n_participants in 1 10 30 60 120 250 500
do
    for temp in 0.5 1 3
    do
        echo "$n_participants|$temp"
        sbatch ./runjob.sh $n_participants $temp
    done
done
