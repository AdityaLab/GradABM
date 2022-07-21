set -e
declare -a weeks=('202014' '202016' '202018' '202020' '202022' '202024' '202026' '202028' '202030')

for w in "${weeks[@]}"
do
    python -u main.py -st MA -j -d 0 1 2 3 -ew "$w" --seed 1234 -m GradABM-time-varying -di COVID
done 

declare -a weeks=('201746' '201748' '201750' '201752' '201802' '201804' '201806' '201808' '201810' '201812' '201814' '201816' '201818')

for w in "${weeks[@]}"
do
    python -u main.py -st MA -j -d 0 1 2 3 -ew "$w" --seed 1234 -m GradABM-time-varying -di Flu
done 
