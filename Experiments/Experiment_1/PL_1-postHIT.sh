#!/usr/bin/env sh
pushd /Documents/GRADUATE_SCHOOL/Turk/aws-mturk-clt-1.3.0/bin
./loadHITs.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 -label /Documents/GRADUATE_SCHOOL/Projects/pragLearn/Experiment_1//PL_1 -input /Documents/GRADUATE_SCHOOL/Projects/pragLearn/Experiment_1//PL_1.input -question /Documents/GRADUATE_SCHOOL/Projects/pragLearn/Experiment_1//PL_1.question -properties /Documents/GRADUATE_SCHOOL/Projects/pragLearn/Experiment_1//PL_1.properties -maxhits 1
popd