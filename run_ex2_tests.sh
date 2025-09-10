#!/bin/bash
sbatch -A ICT25_ESP_0 --partition=boost_usr_prod --qos=normal --ntasks=4 ./test_ex2.sh
sleep 1 # pause to be kind to the scheduler
sbatch -A ICT25_ESP_0 --partition=boost_usr_prod --ntasks=16 ./test_ex2.sh
sleep 1 # pause to be kind to the scheduler
sbatch -A ICT25_ESP_0 --partition=boost_usr_prod --ntasks=32 ./test_ex2.sh
sleep 1 # pause to be kind to the scheduler
sbatch -A ICT25_ESP_0 --partition=boost_usr_prod --ntasks=64 ./test_ex2.sh
sleep 1 # pause to be kind to the scheduler
sbatch -A ICT25_ESP_0 --partition=boost_usr_prod --ntasks=128 ./test_ex2.sh
sleep 1 # pause to be kind to the scheduler