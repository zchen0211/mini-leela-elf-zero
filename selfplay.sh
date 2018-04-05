for i in $(seq 1 50)$
do
    echo "Submit job $i."
    sbatch selfplay_single.sh &
done
