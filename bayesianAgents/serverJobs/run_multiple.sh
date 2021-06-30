# Already calculated for 10 and 15
for datasize in 0 1 5
do
    # Already calculated for 1
    for n_participants in 5 10 30 60 120 250 500 1000
    do
        for temp in 0.5 1 3 6
        do
            echo "$datasize|$n_participants|$temp"
            sbatch ./runjob.sh $datasize $n_participants $temp
        done
    done
done
