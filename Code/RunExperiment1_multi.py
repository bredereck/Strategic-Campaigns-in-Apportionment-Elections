#!/usr/bin/env python3
import random
import os
import sys
import csv
import math
import numpy as np
from pathlib import Path
import datetime
from multiprocessing import Process

from Election import *
from Apportionment import *
from Campaigns1_multi import *

if len(sys.argv) < 5:
    print("Analysis.py <Dataset> <Seats> <target party (best, median, worst, secondbest, thirdbest, worst_pos)> <Threshold>")
    exit()


cur_time = datetime.datetime.now()
target_dir = "../DATA/%s/%s/%s/" % ('multi', sys.argv[3], sys.argv[1].rsplit("/", 1)[1])
Path(target_dir).mkdir(parents=True, exist_ok=True)


################################## loading data ###################################

def read_elections_from_file(filename):
    elections = {
        "districts": []
    }
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if reader.line_num == 1:
                labels = row[2:]
                continue
            district = {
                "number": int(row[0]),
                "seats": int(row[1]),
                "votes": [int(i) for i in row[2:]]
            }
            elections["districts"].append(district)
    elections["parties"] = len(elections["districts"][0]["votes"])
    assert len(labels) == elections["parties"]
    elections["labels"] = labels
    return elections


def get_elections(dataset):

 def datasets_files(dataset):
     if dataset == "Datasets/polishelection2023.csv":
        return {"polishelection2023.csv": "polish23"}
     elif dataset == "Datasets/argentinaelection2021.csv":
         return {"argentinaelection2021.csv": "argentina21"}
     elif dataset == "Datasets/portugalelection2024.csv":
         return {"portugalelection2024.csv": "portugal24"}
     elif dataset == "Datasets/polishelection2019.csv":
        return {"polishelection2019.csv": "polish19"}
 def get_dataset_path(datasetFile):
    return os.path.join("..", "Code", "Datasets", datasetFile)

 electionsList = []
 for dataset in datasets_files(dataset).items():
    electionsList.append((read_elections_from_file(get_dataset_path(dataset[0])), dataset[1]))
 return electionsList



t = float(sys.argv[4])
election = get_elections(sys.argv[1])[0] # the whole election
votes_in_districts = [] # votes for each party in each district
T = [] # thresholds in districts
seats = [] # number of seats in districts
for district in election[0]["districts"]:
    seats.append(district["seats"])
    votes_in_districts.append(district["votes"])
    T.append(np.ceil(t*sum(district["votes"])))

number_of_districts = len(T)
parties = election[0]['labels']
print(parties)
no_manipulation = [] # seats before bribery
for i  in range(0,len(parties)):
    no_manipulation.append(get_count_party_all_seats(i, election[0], T))
print(no_manipulation)

#################### chosen party ##############################################

sum_of_votes = np.array(votes_in_districts[0])
for i in range(1,number_of_districts):
    sum_of_votes += np.array(votes_in_districts[i])

sum_of_votes = list(sum_of_votes)




target_party = sys.argv[3]
if target_party == "best":
    ind = sum_of_votes.index(max(sum_of_votes))
    P = parties[ind]
elif target_party == "median":
    sorted_votes = sorted(sum_of_votes)
    m = sorted_votes[int(np.ceil((len(sum_of_votes)-1)/2))]
    ind = sum_of_votes.index(m)
    P = parties[ind]
elif target_party == "worst":
    ind = sum_of_votes.index(min(sum_of_votes))
    P = parties[ind]



#################### manipulation ##############################################

# contructive bribery - optimal, from the strongest party(ies), from the weakest party(ies)
def get_max_additional_seats_dhondt(nr):
    added_votes = open(target_dir  + P + "-" + str(nr) + "-" + "-DHondt-AddedVotes-optimal.dat", "w")
    added_votes_strongest = open(target_dir  + P + "-" + str(nr) + "-" + "-DHondt-AddedVotes-strongest.dat", "w")
    added_votes_weakest = open(target_dir  + P + "-" + str(nr) + "-" + "-DHondt-AddedVotes-weakest.dat", "w")

    E_initial = election_from_file_multi(parties, votes_in_districts[nr])
    goal = dhondt_allocation(E_initial, T[nr], seats[nr], prefer=P)[P]
    no_manipulation_in_district = int(dhondt_allocation(E_initial, T[nr], seats[nr], prefer=P)[P])
    con = [0,P,no_manipulation_in_district, goal]

    F = E_initial.remove(P)
    rest_votes = int(F.num_votes())
    i = no_manipulation_in_district + 1

    print("%.5f %s %f %f %f %f" % (0, P, sum_of_votes[ind], E_initial[P], no_manipulation[ind], goal), file=added_votes)
    print("%.5f %s %f %f %f %f" % (0, P, sum_of_votes[ind], E_initial[P], no_manipulation[ind], goal), file=added_votes_strongest)
    print("%.5f %s %f %f %f %f" % (0, P, sum_of_votes[ind], E_initial[P], no_manipulation[ind], goal), file=added_votes_weakest)

# optimal
    for B in range(0,rest_votes+1):
        if goal >= no_manipulation_in_district+1:
            break
        E_optimal_con = election_from_file_multi(parties, votes_in_districts[nr])
        cbb = constructive_bribery(E_optimal_con, T[nr], seats[nr], P, i, B)
        if cbb[0] not in [True, False]:
            print("%s" % con, file=added_votes)
            print("%.5f %s %f %f %f" % (B, P, no_manipulation_in_district, cbb[1], cbb[0][P]), file=added_votes)
            for p in cbb[0].keys():
                E_optimal_con[p] -= cbb[0][p]
            print("%s" % E_optimal_con.votealloc, file=added_votes)
            x_opt_con = float(B)
            print("%.5f" % x_opt_con, file=added_votes)
            goal = cbb[1]
        if cbb[0] in [True, False]:
            con = [B,P,goal]
    print("%s" % con, file=added_votes)
    added_votes.close()

# from the strongest party(ies)
    goal = no_manipulation_in_district
    con_str = [0, P, goal]
    for B in range(0, rest_votes + 1):
        if goal >= no_manipulation_in_district + 1:
            break
        E_strongest_con = election_from_file_multi(parties, votes_in_districts[nr])
        cbb_str = constructive_bribery_from_strongest(E_strongest_con, T[nr], seats[nr], P, i, B)
        if cbb_str[0] not in [True, False]:
            print("%s" % con_str, file=added_votes_strongest)
            print("%.5f %s %f %f %f" % (B, P, no_manipulation_in_district, cbb_str[1], cbb_str[0][P] - E_initial[P]),
                  file=added_votes_strongest)
            print("%s" % E_strongest_con.votealloc,
                  file=added_votes_strongest)
            x_str_con = float(B)
            goal = cbb_str[1]
        if cbb_str[0] in [True, False]:
            con_str = [B,P,cbb_str[1]]
    print("%s" % con_str, file=added_votes_strongest)
    added_votes_strongest.close()

# from the weakest party(ies)
    goal = no_manipulation_in_district
    con_wea = [0, P, goal]
    for B in range(0, rest_votes + 1):
        if goal >= no_manipulation_in_district + 1:
            break
        E_weakest_con = election_from_file_multi(parties, votes_in_districts[nr])
        cbb_wea = constructive_bribery_from_weakest(E_weakest_con, T[nr], seats[nr], P, i, B)
        if cbb_wea[0] not in [True, False]:
            print("%s" % con_wea, file=added_votes_weakest)
            print("%.5f %s %f %f %f" % (B, P, no_manipulation_in_district, cbb_wea[1], cbb_wea[0][P] - E_initial[P]),
                      file=added_votes_weakest)
            print("%s" % E_weakest_con.votealloc,
                  file=added_votes_weakest)
            x_wea_con = float(B)
            goal = cbb_wea[1]
        if cbb_wea[0] in [True, False]:
            con_wea = [B, P, cbb_wea[1]]
    print("%s" % con_wea, file=added_votes_weakest)
    added_votes_weakest.close()

    return [x_opt_con, x_str_con, x_wea_con]


# destructive bribery - optimal, to the strongest party, to the weakest party
def get_max_prevented_seats_dhondt(nr):
    removed_votes = open(target_dir + P + "-" + str(nr) + "-" +"-DHondt-RemovedVotes-optimal.dat", "w")
    removed_votes_str = open(target_dir  + P + "-" + str(nr) + "-" + "-DHondt-RemovedVotes_strongest.dat", "w")
    removed_votes_wea = open(target_dir  + P + "-" + str(nr) + "-" + "-DHondt-RemovedVotes_weakest.dat", "w")

    E_initial = election_from_file_multi(parties, votes_in_districts[nr])
    goal = dhondt_allocation(E_initial, T[nr], seats[nr], prefer=P)[P]
    no_manipulation_in_district = dhondt_allocation(E_initial, T[nr], seats[nr], prefer=P)[P]
    con = [0, P, no_manipulation_in_district, goal]

    print("%.5f %s %f %f %f %f" % (0, P, sum_of_votes[ind], E_initial[P], no_manipulation[ind], goal), file=removed_votes)
    print("%.5f %s %f %f %f %f" % (0, P, sum_of_votes[ind], E_initial[P], no_manipulation[ind], goal), file=removed_votes_str)
    print("%.5f %s %f %f %f %f" % (0, P, sum_of_votes[ind], E_initial[P], no_manipulation[ind], goal), file=removed_votes_wea)

    if no_manipulation_in_district == 0:
        print("%.5f %s %f %s" % (0, P, no_manipulation[ind], "no bribery possible"), file=removed_votes)
        print("%.5f %s %f %s" % (0, P, no_manipulation[ind], "no bribery possible"), file=removed_votes_str)
        print("%.5f %s %f %s" % (0, P, no_manipulation[ind], "no bribery possible"), file=removed_votes_wea)
    else:
# optimal
        i = no_manipulation_in_district - 1
        for B in range(0,E_initial[P]+1):
            if goal <= no_manipulation_in_district - 1:
                break
            E_optimal_des = election_from_file_multi(parties, votes_in_districts[nr])
            dbb = destructive_bribery(E_optimal_des, T[nr], seats[nr], P, i, B)
            if dbb[0] not in [True, False]:
                print("%s" % con, file=removed_votes)
                print("%.5f %s %f %f %f" % (B, P, no_manipulation_in_district, dbb[1], dbb[0][P]), file=removed_votes)
                for p in dbb[0].keys():
                    E_optimal_des[p] -= dbb[0][p]
                print("%s" % E_optimal_des.votealloc,
                      file=removed_votes)
                goal = dbb[1]
                x_opt_des = float(B)
            if dbb[0] in [True,False]:
                con = [B, P, goal]
        print("%s" % con, file=removed_votes)

# to the strongest party
        goal = no_manipulation_in_district
        con_str = [0, P, goal]
        for B in range(0, E_initial[P] + 1):
            if goal <= no_manipulation_in_district - 1:
                break
            E_strongest_des = election_from_file_multi(parties, votes_in_districts[nr])
            dbb_str = destructive_bribery_to_strongest(E_strongest_des, T[nr], seats[nr], P, i, B)
            if dbb_str[0] not in [True, False]:
                print("%s" % con_str, file=removed_votes_str)
                print("%.5f %s %f %f %f" % (B, P, no_manipulation_in_district, dbb_str[1], E_initial[P] - dbb_str[0][P]),
                      file=removed_votes_str)
                print("%s" % E_strongest_des.votealloc,
                      file=removed_votes_str)
                goal = dbb_str[1]
                x_str_des = float(B)
            if dbb_str[0] in [True, False]:
                con_str = [B, P, goal]
        print("%s" % con_str, file=removed_votes_str)

# to the weakest party
        goal = no_manipulation_in_district
        con_wea = [0, P, goal]
        for B in range(0, E_initial[P] + 1):
            if goal <= no_manipulation_in_district - 1:
                break
            E_weakest_des = election_from_file_multi(parties, votes_in_districts[nr])
            dbb_wea = destructive_bribery_to_weakest(E_weakest_des, T[nr], seats[nr], P, i, B)
            if dbb_wea[0] not in [True, False]:
                print("%s" % con_wea, file=removed_votes_wea)
                print("%.5f %s %f %f %f" % (B, P, no_manipulation_in_district, dbb_wea[1], E_initial[P] - dbb_wea[0][P]),
                      file=removed_votes_wea)
                print("%s" % E_weakest_des.votealloc,
                      file=removed_votes_wea)
                goal = dbb_wea[1]
                x_wea_des = float(B)
            if dbb_wea[0] in [True, False]:
                con_wea = [B, P, goal]
        print("%s" % con_wea, file=removed_votes_wea)

    removed_votes.close()
    removed_votes_str.close()
    removed_votes_wea.close()

    return [x_opt_des,x_str_des,x_wea_des]


# three functions for constructive balanced bribery
def experiment(nr,parties_in, votes_in, seats_in, k, gainvals, identifier="no identifier"):
    divisors = range(1, 200)
    origresult = seats_in
    results = []
    for gainseats in gainvals:
        target = origresult[0] + gainseats
        if target > k:
            continue

        # Balanced Bribery
        balbrbres = findbalancedbribery(votes_in, k, target, divisors, T[nr])
        #balbrbres = findbalancedbribery_con(votes, k, target, divisors, T)
        balbrb = balbrbres[0]
        corr = balbrbres[1]

        results.append((identifier, len(votes_in), votes_in, k,
                            gainseats,
                            parties_in[0], balbrb,
                            # brb, strbrb, weakbrb,
                            #abstainers,
                            balbrb[0]
                            # brb[0],
                            # abstainers * 1. / brb[0],
                            #abstainers * 1. / balbrb[0]
                            # abstainers * 1. / strbrb[0],
                            # abstainers * 1. / weakbrb[0]
                            ))
        # print("bribery gain = {}".format(abstainers / brb[0]))
        # print(results..1])
        #print(results[-1])
    return [results,corr]


def constructive_exp_single(gainvals,nr):
    E = election_from_file_multi(parties, votes_in_districts[nr])
    year = sys.argv[1][-4:]
    results = []
    votes_in = [E[P]]
    parties_in = [P]
    votes_above = []
    parties_above = []
    seats_in = [dhondt_allocation(E, T[nr], seats[nr], prefer=P)[P]]
    seats_above = []
    F = E.remove(P)
    for p in F.parties():
        votes_in.append(F[p])
        parties_in.append(p)
        seats_in.append(dhondt_allocation(E, T[nr], seats[nr], prefer=P)[p])
    for i in range(0,len(votes_in)):
        if votes_in[i] >= T[nr]:
            votes_above.append(votes_in[i])
            parties_above.append(parties_in[i])
            seats_above.append(seats_in[i])


    results_exp = experiment(nr,parties_in, votes_in, seats_in, seats[nr], gainvals, identifier=year)
    results += results_exp[0]

    w = np.array(votes_in)
    u = np.array(results[0][-2])
    votes_brb = w + u

    for i in range(0,len(votes_in)):
        E[parties_in[i]] = votes_brb[i]
    #print(E[P])
    #print(dhondt_allocation(E, T, K, prefer=P)[P])
    return [E, results, dhondt_allocation(E, T[nr], seats[nr], prefer=P)[P], results_exp[1]]


def balanced_additional_dhondt(nr):
    E_initial = election_from_file_multi(parties, votes_in_districts[nr])
    no_manipulation_in_district = dhondt_allocation(E_initial, T[nr], seats[nr], prefer=P)[P]
    added_votes_bal = open(target_dir + P + "-" + str(nr) + "-" + "-DHondt-AddedVotes-balanced.dat", "w")

    under_threshold = T[nr] - E_initial[P]

    print("%.5f %s %f %f" % (0, P, E_initial[P], no_manipulation[ind]), file=added_votes_bal)

    gainvals = list(range(1, 2))
    results = constructive_exp_single(gainvals,nr)
    #print(no_manipulation)
    #print(results)
    E_balanced_con = results[0]

    print("%.5f %s %f %f %f %s" % ( results[1][0][-1], P, no_manipulation_in_district, results[2], results[1][0][-1], E_balanced_con.votealloc),
          file=added_votes_bal)


    F = E_balanced_con.remove(P)
    res = dhondt_allocation(E_initial, T[nr], seats[nr], prefer=P)[P]
    R = F.get_x_best_party(1)
    results_con = dhondt_allocation(E_balanced_con, T[nr], seats[nr], prefer=P)[P]
    while results_con == res + 1:
        E_balanced_con[P] -= 1
        E_balanced_con[R] += 1
        results_con = dhondt_allocation(E_balanced_con, T[nr], seats[nr], prefer=P)[P]

        print("%s %f %.5f %s %f %f" % (results[-1], res, E_balanced_con[P] - E_initial[P], P, results_con, under_threshold),
              file=added_votes_bal)

    print("%s %f %.5f %s %f %f" % (results[-1], res, E_initial[P], P, results_con, under_threshold),
          file=added_votes_bal)

    added_votes_bal.close()
    return float(results[1][0][-1])


# three functions for destructive balanced bribery
def experiment_des(nr, parties_in, votes_in, seats_in, k, lostvals, identifier="no identifier"):
    divisors = range(1, 200)
    origresult = seats_in
    results = []
    for lostseats in lostvals:
        target = origresult[0] - lostseats
        if target > k:
            continue


        # Balanced Bribery
        balbrbres = findbalancedbribery_des(votes_in, k, target, divisors, T[nr])
        balbrb = balbrbres[0]
        corr = balbrbres[1]

        results.append((identifier, len(votes_in), votes_in, k,
                            lostseats,
                            parties_in[0], balbrb,
                            # brb, strbrb, weakbrb,
                            #abstainers,
                            balbrb[0]
                            # brb[0],
                            # abstainers * 1. / brb[0],
                            #abstainers * 1. / balbrb[0]
                            # abstainers * 1. / strbrb[0],
                            # abstainers * 1. / weakbrb[0]
                            ))
        # print("bribery gain = {}".format(abstainers / brb[0]))
        # print(results..1])
        #print(results[-1])
    return [results,corr]



def destructive_exp_single(lostvals,nr):
    E = election_from_file_multi(parties, votes_in_districts[nr])
    year = sys.argv[1][-4:]
    results = []
    F = E.remove(P)
    votes_in = [E[P]]
    parties_in = [P]
    votes_above = []
    parties_above = []
    seats_in = [dhondt_allocation(E, T[nr], seats[nr], prefer=P)[P]]
    seats_above = []
    for p in F.parties():
        votes_in.append(F[p])
        parties_in.append(p)
        seats_in.append(dhondt_allocation(E, T[nr], seats[nr], prefer=P)[p])
    for i in range(0,len(votes_in)):
        if votes_in[i] >= T[nr]:
            votes_above.append(votes_in[i])
            parties_above.append(parties_in[i])
            seats_above.append(seats_in[i])

    results_exp = experiment_des(nr, parties_in, votes_in, seats_in, seats[nr], lostvals, identifier=year)
    results += results_exp[0]

    w = np.array(votes_in)
    u = np.array(results[0][-2])
    votes_brb = w + u
    #print(votes_brb)
    #print(parties)
    for i in range(0,len(votes_in)):
        E[parties_in[i]] = votes_brb[i]
    #print(E)
    return [E, results, dhondt_allocation(E, T[nr], seats[nr], prefer=P)[P], results_exp[1]]



def balanced_prevented_dhondt(nr):
    E_initial = election_from_file_multi(parties, votes_in_districts[nr])
    no_manipulation_in_district = dhondt_allocation(E_initial, T[nr], seats[nr], prefer=P)[P]
    removed_votes_bal = open(target_dir + P + "-" + str(nr) + "-" + "-DHondt-RemovedVotes-balanced.dat", "w")

    under_threshold = T[nr] - E_initial[P]

    print("%.5f %s %f %f" % (0, P, E_initial[P], no_manipulation[ind]), file=removed_votes_bal)

    lostvals = list(range(1, 2))

    if no_manipulation_in_district == 0:
        print("%s %f %s" % ( P, no_manipulation[ind], "bribery impossible"),
            file=removed_votes_bal)
    else:
        results = destructive_exp_single(lostvals,nr)
        E_balanced_des = results[0]
        print("%.5f %s %f %f %f %s" % (results[1][0][-1], P, no_manipulation_in_district, results[2], results[1][0][-1], E_balanced_des.votealloc),
          file=removed_votes_bal)

        res = dhondt_allocation(E_initial, T[nr], seats[nr], prefer=P)[P]
        F = E_balanced_des.remove(P)
        R = F.get_x_best_party(1)
        results_con = dhondt_allocation(E_balanced_des, T[nr], seats[nr], prefer=P)[P]
        while results_con == res - 1:
            E_balanced_des[P] += 1
            E_balanced_des[R] -= 1
            results_con = dhondt_allocation(E_balanced_des, T[nr], seats[nr], prefer=P)[P]

            print("%s %f %.5f %s %f %f" % (results[-1], res, E_balanced_des[P], P, results_con, under_threshold),
                  file=removed_votes_bal)

    removed_votes_bal.close()
    return -float(results[1][0][-1])


print("multi-district")
print("File: ", sys.argv[1])
print(sys.argv[3])

#election_example = election_from_file_multi(parties,votes_in_districts[0])

eff_in_districts_optimal_con = []
eff_in_districts_strongest_con = []
eff_in_districts_weakest_con = []
eff_in_districts_balanced_con = []
eff_votes_in_con = []

eff_in_districts_optimal_des = []
eff_in_districts_strongest_des = []
eff_in_districts_weakest_des = []
eff_in_districts_balanced_des = []
eff_votes_in_des = []


for nr_district in range(0,number_of_districts):
    E = election_from_file_multi(parties, votes_in_districts[nr_district])
    #print(dhondt_allocation(E, T[nr_district], seats[nr_district], prefer=P))
    print("district: ", nr_district)

    if dhondt_allocation(E, T[nr_district], seats[nr_district], prefer=P)[P] < seats[nr_district]:
        x = get_max_additional_seats_dhondt(nr_district)
        eff_in_districts_optimal_con.append(x[0])
        eff_in_districts_strongest_con.append(x[1])
        eff_in_districts_weakest_con.append(x[2])
        eff_votes_in_con.append(sum(votes_in_districts[nr_district])-votes_in_districts[nr_district][ind])
        print("#1 done")

        eff_in_districts_balanced_con.append(balanced_additional_dhondt(nr_district))
        print("#3 done")

    if dhondt_allocation(E, T[nr_district], seats[nr_district], prefer=P)[P] > 0:
        y = get_max_prevented_seats_dhondt(nr_district)
        eff_in_districts_optimal_des.append(y[0])
        eff_in_districts_strongest_des.append(y[1])
        eff_in_districts_weakest_des.append(y[2])
        eff_votes_in_des.append(votes_in_districts[nr_district][ind])
        print("#2 done")

        eff_in_districts_balanced_des.append(balanced_prevented_dhondt(nr_district))
        print("#4 done")

if eff_in_districts_optimal_des == []:
    eff_in_districts_optimal_des.append(-1)
    eff_votes_in_des.append(1)

if eff_in_districts_strongest_des == []:
    eff_in_districts_strongest_des.append(-1)

if eff_in_districts_weakest_des == []:
    eff_in_districts_weakest_des.append(-1)

if eff_in_districts_balanced_des == []:
    eff_in_districts_balanced_des.append(-1)

min_optimal_con = min(eff_in_districts_optimal_con)
min_optimal_des = min(eff_in_districts_optimal_des)

final = open(target_dir + P + "-final.dat", "w")
print("%s %i %i %i" % (sum_of_votes, sum(sum_of_votes), sum_of_votes[ind], sum(sum_of_votes)-sum_of_votes[ind]),file=final)

print("%s %f %i %.10f " % ("constructive-optimal", min_optimal_con, eff_in_districts_optimal_con.index(min_optimal_con), min_optimal_con/min_optimal_con),file=final)
print("%s" % eff_in_districts_optimal_con,file=final)
print("%s %f %i %.10f " % ("constructive-strongest", min(eff_in_districts_strongest_con), eff_in_districts_strongest_con.index(min(eff_in_districts_strongest_con)),  min(eff_in_districts_strongest_con)/min_optimal_con),file=final)
print("%s " % eff_in_districts_strongest_con,file=final)
print("%s %f %i %.10f " % ("constructive-weakest", min(eff_in_districts_weakest_con), eff_in_districts_weakest_con.index(min(eff_in_districts_weakest_con)), min(eff_in_districts_weakest_con)/min_optimal_con),file=final)
print("%s " % eff_in_districts_weakest_con,file=final)
print("%s %f %i %.10f " % ("constructive-balanced", min(eff_in_districts_balanced_con), eff_in_districts_balanced_con.index(min(eff_in_districts_balanced_con)), min(eff_in_districts_balanced_con)/min_optimal_con),file=final)
print("%s " % eff_in_districts_balanced_con,file=final)

print("%s %f %i %.10f " % ("destructive-optimal", min_optimal_des, eff_in_districts_optimal_des.index(min_optimal_des), min_optimal_des/min_optimal_des),file=final)
print("%s" % eff_in_districts_optimal_des,file=final)
print("%s %f %i %.10f " % ("destructive-strongest", min(eff_in_districts_strongest_des), eff_in_districts_strongest_des.index(min(eff_in_districts_strongest_des)), min(eff_in_districts_strongest_des)/min_optimal_des),file=final)
print("%s" % eff_in_districts_strongest_des,file=final)
print("%s %f %i %.10f " % ("destructive-weakest", min(eff_in_districts_weakest_des), eff_in_districts_weakest_des.index(min(eff_in_districts_weakest_des)), min(eff_in_districts_weakest_des)/min_optimal_des),file=final)
print("%s" % eff_in_districts_weakest_des,file=final)
print("%s %f %i %.10f " % ("destructive-balanced", min(eff_in_districts_balanced_des), eff_in_districts_balanced_des.index(min(eff_in_districts_balanced_des)), min(eff_in_districts_balanced_des)/min_optimal_des),file=final)
print("%s" % eff_in_districts_balanced_des,file=final)
final.close()

print("Finished")
