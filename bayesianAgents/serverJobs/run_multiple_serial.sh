for n_participants in 300
do
    for temp in 0.5 1.0 3.0 5.0
    do
        echo "$n_participants|$temp"
        sbatch ./runjob_serial.sh $n_participants $temp
    done
done
