for i in $(seq 1 10)$
do
    echo "Submit job $i."
    sh selfplay_single.sh
done
