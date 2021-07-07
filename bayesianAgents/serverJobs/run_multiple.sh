# for now don't run 0 and 1. They probably take more than 100 hours atm
for datasize in 5 10 15
do
    for n_participants in 1 10 30 60 120 250 500
    do
        for temp in 0.5 1 3
        do
            echo "$datasize|$n_participants|$temp"
            sbatch ./runjob.sh $datasize $n_participants $temp
        done
    done
done
