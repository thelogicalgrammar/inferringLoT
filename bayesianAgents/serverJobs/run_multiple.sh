# Already ran 5 10 15.
# the limit for each simulation is 100 hours
# and there are 350 runs within each simulation
# therefore as long as each run takes less than 1028 seconds
# (i.e. about 17 minutes)
# it should have enough time to finish
for datasize in 0 1
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
