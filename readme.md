# Contents and Reproduction Instructions
The project contains the companion code to the submission of paper
"How to Tamper with a Parliament: Strategic Campaigns in Apportionment
Elections".

The files contain:
 - directory `DATA` containing the demonstration of outcomes (already computed)
   for experiments 1 and 2
 - `Code/data` containing the data we used in the experiments
 - directory `Code` containing
   - bash scripts `run_single_experiment_1.sh`, `run_single_avg_experiment_1.sh`,
     `run_multi_experiment_1.sh`, and `run_multi_avg_experiment_1.sh` to run
     parts of experiments 1, according to the names for single district and
     multiple districts, and either computing strategies for all parties (`avg`
     versions) or for certain ones (best, worst, etc.) as described in the article.
   - bash script `run_all_experiment_2.sh` to run the second experiment from the
     paper
   - python script `Experiment3/experiment_runner.py` to run parts of the third
     experiment. To get all results one needs slightly modify the source code of 
     the script. There are two places which needs to be
     ammended depending on whether one wants to compute the descructive or
     constructive veriants (see the comments in the source file). The party
     for which the computation is done is specified by variable `preferredParty`
     (line 217) whose respective values for the results presented in the paper
     are 3, 16, and 1 for the election in Poland, Portugal, and Argentina
     respectively. Further, one needs to set correct values of the
     for-loop control variable `mergesCount` (line 240) to compute the results for appropriate
     numbers of merges presented in the article.
