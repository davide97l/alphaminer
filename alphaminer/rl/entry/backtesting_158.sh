#!/bin/bash

#python eval.py --env-type 158 --top-n 50 --load-path /Users/weiyuhui/Desktop/projects/alphaminer/exp/weights/alpha158/avg_best_at_1415000.pth.tar --exp-name alpha158 --start-time 2019-01-01 --end-time 2022-06-30 -cs 1600 -sl 0
python eval.py --env-type 158 --top-n 50 --load-path /Users/weiyuhui/Desktop/projects/alphaminer/exp/weights/alpha158_wc_2/avg_best_at_5000.pth.tar --exp-name alpha158_wc --start-time 2019-01-01 --end-time 2022-06-30 -cs 1600 -sl 0