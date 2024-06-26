#!/usr/bin/env python3
import random
import os
import sys
import math
import numpy as np
from pathlib import Path
import datetime
from multiprocessing import Process

from Election import *
from Apportionment import *
from Campaigns1 import *

if len(sys.argv) < 5:
    print("Analysis.py <Dataset> <Seats> <target party (best, median, worst, secondbest, thirdbest, worst_pos)> <Threshold>")
    exit()



E = election_from_file(sys.argv[1])
E_optimal_con = election_from_file(sys.argv[1])
E_optimal_des = election_from_file(sys.argv[1])
E_strongest_con = election_from_file(sys.argv[1])
E_strongest_des = election_from_file(sys.argv[1])
E_weakest_con = election_from_file(sys.argv[1])
E_weakest_des = election_from_file(sys.argv[1])
E_balanced_con = election_from_file(sys.argv[1])
E_balanced_des = election_from_file(sys.argv[1])
K = int(sys.argv[2])
t = float(sys.argv[4])
T = math.ceil(t*E.num_votes())
target_party = sys.argv[3]
if target_party == "best":
    P = E.get_x_best_party(1)
elif target_party == "median":
    P = E.get_x_best_party(E.num_parties() // 2)
elif target_party == "worst":
    P = E.get_worst_party()
elif target_party == "secondbest":
    P = E.get_x_best_party(2)
elif target_party == "thirdbest":
    P = E.get_x_best_party(3)
elif target_party == "worst_pos":
    possible_parties = []
    for p in E.parties():
        if dhondt_allocation(E, T, K, prefer=p)[p] >= 1:
            possible_parties.append(p)
    vot = E[possible_parties[0]].copy()
    P = possible_parties[0].copy()
    for p in possible_parties:
        if E[p] < vot:
            vot = E[p].copy()
            P = p.copy()
else:
    print("Unknown option: %s"%target_party)
    exit()
cur_time = datetime.datetime.now()
target_dir = "../DATA/%s/%s/%s/" % ('1', target_party, sys.argv[1].rsplit("/", 1)[1])
Path(target_dir).mkdir(parents=True, exist_ok=True)

no_manipulation = dhondt_allocation(E, T, K, prefer=P)[P]
F = E.remove(P)


# contructive bribery - optimal, from the strongest party(ies), from the weakest party(ies)
def get_max_additional_seats_dhondt():
    added_votes = open(target_dir  + P + "-DHondt-AddedVotes-optimal.dat", "w")
    added_votes_strongest = open(target_dir  + P + "-DHondt-AddedVotes-strongest.dat", "w")
    added_votes_weakest = open(target_dir  + P + "-DHondt-AddedVotes-weakest.dat", "w")

    goal = no_manipulation
    con = [0,P,no_manipulation]

    rest_votes = int(F.num_votes())
    i = no_manipulation + 1

    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=added_votes)
    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=added_votes_strongest)
    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=added_votes_weakest)

# optimal
    for B in range(0,rest_votes+1):
        if goal >= no_manipulation+1:
            break
        E_optimal_con = election_from_file(sys.argv[1])
        cbb = constructive_bribery(E_optimal_con, T, K, P, i, B)
        if cbb[0] not in [True, False]:
            print("%s" % con, file=added_votes)
            print("%.5f %s %f %f %f" % (B,P,no_manipulation,cbb[1],cbb[0][P]), file=added_votes)
            for p in cbb[0].keys():
                E_optimal_con[p] -= cbb[0][p]
            print("%s" % E_optimal_con.votealloc, file=added_votes)
            x1 = B/(rest_votes+E[P])
            x2 = 1-x1
            x3 = B/rest_votes
            x4 = 1-x3
            print("%.5f %.5f %.5f %.5f" % (x1,x2,x3,x4), file=added_votes)
            goal = cbb[1]
        if cbb[0] in [True, False]:
            con = [B,P,goal]
    print("%s" % con, file=added_votes)
    added_votes.close()

# from the strongest party(ies)
    goal = no_manipulation
    con_str = [0, P, goal]
    for B in range(0, rest_votes + 1):
        if goal >= no_manipulation + 1:
            break
        E_strongest_con = election_from_file(sys.argv[1])
        cbb_str = constructive_bribery_from_strongest(E_strongest_con, T, K, P, i, B)
        if cbb_str[0] not in [True, False]:
            print("%s" % con_str, file=added_votes_strongest)
            print("%.5f %s %f %f %f" % (B,P,no_manipulation,cbb_str[1],cbb_str[0][P]-E[P]), file=added_votes_strongest)
            print("%s" % E_strongest_con.votealloc,
                  file=added_votes_strongest)
            x1 = B/(rest_votes+E[P])
            x2 = 1-x1
            x3 = B/rest_votes
            x4 = 1-x3
            print("%.5f %.5f %.5f %.5f" % (x1,x2,x3,x4), file=added_votes_strongest)
            goal = cbb_str[1]
        if cbb_str[0] in [True, False]:
            con_str = [B,P,cbb_str[1]]
    print("%s" % con_str, file=added_votes_strongest)
    added_votes_strongest.close()

# from the weakest party(ies)
    goal = no_manipulation
    con_wea = [0, P, goal]
    for B in range(0, rest_votes + 1):
        if goal >= no_manipulation + 1:
            break
        E_weakest_con = election_from_file(sys.argv[1])
        cbb_wea = constructive_bribery_from_weakest(E_weakest_con, T, K, P, i, B)
        if cbb_wea[0] not in [True, False]:
            print("%s" % con_wea, file=added_votes_weakest)
            print("%.5f %s %f %f %f" % (B, P, no_manipulation, cbb_wea[1], cbb_wea[0][P] - E[P]),
                      file=added_votes_weakest)
            print("%s" % E_weakest_con.votealloc,
                  file=added_votes_weakest)
            x1 = B/(rest_votes+E[P])
            x2 = 1-x1
            x3 = B/rest_votes
            x4 = 1-x3
            print("%.5f %.5f %.5f %.5f" % (x1,x2,x3,x4), file=added_votes_weakest)
            goal = cbb_wea[1]
        if cbb_wea[0] in [True, False]:
            con_wea = [B, P, cbb_wea[1]]
    print("%s" % con_wea, file=added_votes_weakest)
    added_votes_weakest.close()


# destructive bribery - optimal, to the strongest party, to the weakest party
def get_max_prevented_seats_dhondt():
    removed_votes = open(target_dir + P +"-DHondt-RemovedVotes-optimal.dat", "w")
    removed_votes_str = open(target_dir  + P + "-DHondt-RemovedVotes_strongest.dat", "w")
    removed_votes_wea = open(target_dir  + P + "-DHondt-RemovedVotes_weakest.dat", "w")

    goal = no_manipulation
    con = [0, P, goal]

    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=removed_votes)
    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=removed_votes_str)
    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=removed_votes_wea)

    if no_manipulation == 0:
        print("%.5f %s %f %s" % (0, P, no_manipulation, "no bribery possible"), file=removed_votes)
        print("%.5f %s %f %s" % (0, P, no_manipulation, "no bribery possible"), file=removed_votes_str)
        print("%.5f %s %f %s" % (0, P, no_manipulation, "no bribery possible"), file=removed_votes_wea)
    else:
# optimal
        i = no_manipulation - 1
        for B in range(0,E[P]+1):
            if goal <= no_manipulation-1:
                break
            E_optimal_des = election_from_file(sys.argv[1])
            dbb = destructive_bribery(E_optimal_des, T, K, P, i, B)
            if dbb[0] not in [True, False]:
                print("%s" % con, file=removed_votes)
                print("%.5f %s %f %f %f" % (B, P,no_manipulation, dbb[1], dbb[0][P]), file=removed_votes)
                for p in dbb[0].keys():
                    E_optimal_des[p] -= dbb[0][p]
                print("%s" % E_optimal_des.votealloc,
                      file=removed_votes)
                if B != 0:
                    x1 = B/(E.num_votes())
                    x2 = 1-x1
                    x3 = B/E[P]
                    x4 = 1-x3
                    print("%.5f %.5f %.5f %.5f" % (x1,x2,x3,x4), file=removed_votes)
                else:
                    print("%.5f" % 0, file=removed_votes)
                goal = dbb[1]
            if dbb[0] in [True,False]:
                con = [B, P, goal]
        print("%s" % con, file=removed_votes)

# to the strongest party
        goal = no_manipulation
        con_str = [0, P, goal]
        for B in range(0, E[P] + 1):
            if goal <= no_manipulation - 1:
                break
            E_strongest_des = election_from_file(sys.argv[1])
            dbb_str = destructive_bribery_to_strongest(E_strongest_des, T, K, P, i, B)
            if dbb_str[0] not in [True, False]:
                print("%s" % con_str, file=removed_votes_str)
                print("%.5f %s %f %f %f" % (B, P, no_manipulation, dbb_str[1], E[P]-dbb_str[0][P]),
                          file=removed_votes_str)
                print("%s" % E_strongest_des.votealloc,
                      file=removed_votes_str)
                if B != 0:
                    x1 = B/(E.num_votes())
                    x2 = 1-x1
                    x3 = B/E[P]
                    x4 = 1-x3
                    print("%.5f %.5f %.5f %.5f" % (x1,x2,x3,x4), file=removed_votes_str)
                else:
                    print("%.5f" % 0, file=removed_votes_str)
                goal = dbb_str[1]
            if dbb_str[0] in [True, False]:
                con_str = [B, P, dbb_str[1]]
        print("%s" % con_str, file=removed_votes_str)

# to the weakest party
        goal = no_manipulation
        con_wea = [0, P, goal]
        for B in range(0, E[P] + 1):
            if goal <= no_manipulation - 1:
                break
            E_weakest_des = election_from_file(sys.argv[1])
            dbb_wea = destructive_bribery_to_weakest(E_weakest_des, T, K, P, i, B)
            if dbb_wea[0] not in [True, False]:
                print("%s" % con_wea, file=removed_votes_wea)
                print("%.5f %s %f %f %f" % (B, P, no_manipulation, dbb_wea[1], E[P]-dbb_wea[0][P]),
                          file=removed_votes_wea)
                print("%s" % E_weakest_des.votealloc,
                          file=removed_votes_wea)
                if B != 0:
                    x1 = B/(E.num_votes())
                    x2 = 1-x1
                    x3 = B/E[P]
                    x4 = 1-x3
                    print("%.5f %.5f %.5f %.5f" % (x1,x2,x3,x4), file=removed_votes_wea)
                else:
                    print("%.5f" % 0, file=removed_votes_wea)
                goal = dbb_wea[1]
            if dbb_wea[0] in [True, False]:
                con_wea = [B, P, dbb_wea[1]]
        print("%s" % con_wea, file=removed_votes_wea)

    removed_votes.close()
    removed_votes_str.close()
    removed_votes_wea.close()


# three functions for constructive balanced bribery
def experiment(parties, votes, seats, k, gainvals, identifier="no identifier"):
    divisors = range(1, 200)
    origresult = seats
    results = []
    for gainseats in gainvals:
        target = origresult[0] + gainseats
        if target > k:
            continue

        # Balanced Bribery
        balbrbres = findbalancedbribery(votes, k, target, divisors, T)
        #balbrbres = findbalancedbribery_con(votes, k, target, divisors, T)
        balbrb = balbrbres[0]
        corr = balbrbres[1]

        results.append((identifier, len(votes), votes, k,
                            gainseats,
                            parties[0], balbrb,
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


def constructive_exp_single(gainvals):
    year = sys.argv[1][-4:]
    results = []
    votes = [E[P]]
    parties = [P]
    votes_above = []
    parties_above = []
    seats = [no_manipulation]
    seats_above = []
    for p in F.parties():
        votes.append(F[p])
        parties.append(p)
        seats.append(dhondt_allocation(E, T, K, prefer=P)[p])
    for i in range(0,len(votes)):
        if votes[i] >= T:
            votes_above.append(votes[i])
            parties_above.append(parties[i])
            seats_above.append(seats[i])

    results_exp = experiment(parties, votes, seats, K, gainvals, identifier=year)
    results += results_exp[0]

    w = np.array(votes)
    u = np.array(results[0][-2])
    votes_brb = w + u

    for i in range(0,len(votes)):
        E_balanced_con[parties[i]] = votes_brb[i]

    return [E_balanced_con, results, dhondt_allocation(E_balanced_con, T, K, prefer=P)[P], results_exp[1]]


def balanced_additional_dhondt():
    added_votes_bal = open(target_dir + P + "-DHondt-AddedVotes-balanced.dat", "w")

    under_threshold = T - E[P]

    print("%.5f %s %f %f" % (0, P, E[P], no_manipulation), file=added_votes_bal)

    gainvals = list(range(1, 2))
    results = constructive_exp_single(gainvals)
    E_balanced_con = results[0]

    print("%.5f %s %f %f %f" % ( results[1][0][-1], P, no_manipulation, results[2], results[1][0][-1]),
          file=added_votes_bal)
    print("%s" % E_balanced_con.votealloc, file=added_votes_bal)

    res = dhondt_allocation(E, T, K, prefer=P)[P]
    R = F.get_x_best_party(1)
    results_con = dhondt_allocation(E_balanced_con, T, K, prefer=P)[P]
    while results_con == res + 1:
        E_balanced_con[P] -= 1
        E_balanced_con[R] += 1
        results_con = dhondt_allocation(E_balanced_con, T, K, prefer=P)[P]

        print("%s %f %.5f %s %f %f" % (results[-1], res, E_balanced_con[P]-E[P], P, results_con, under_threshold),
              file=added_votes_bal)

    B = results[1][0][-1]
    if B != 0:
        x1 = B/(E.num_votes())
        x2 = 1-x1
        x3 = B/(E.num_votes()-E[P])
        x4 = 1-x3
        print("%.5f %.5f %.5f %.5f" % (x1,x2,x3,x4), file=added_votes_bal)
    else:
        print("%.5f" % 0, file=added_votes_bal)

    added_votes_bal.close()


# three functions for destructive balanced bribery
def experiment_des(parties, votes, seats, k, lostvals, identifier="no identifier"):
    divisors = range(1, 200)
    origresult = seats
    results = []
    for lostseats in lostvals:
        target = origresult[0] - lostseats
        if target > k:
            continue


        # Balanced Bribery
        balbrbres = findbalancedbribery_des(votes, k, target, divisors, T)
        balbrb = balbrbres[0]
        corr = balbrbres[1]

        results.append((identifier, len(votes), votes, k,
                            lostseats,
                            parties[0], balbrb,
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



def destructive_exp_single(lostvals):
    year = sys.argv[1][-4:]
    results = []
    #F = E.remove(P)
    votes = [E[P]]
    parties = [P]
    votes_above = []
    parties_above = []
    seats = [no_manipulation]
    seats_above = []
    for p in F.parties():
        votes.append(F[p])
        parties.append(p)
        seats.append(dhondt_allocation(E, T, K, prefer=P)[p])
    for i in range(0,len(votes)):
        if votes[i] >= T:
            votes_above.append(votes[i])
            parties_above.append(parties[i])
            seats_above.append(seats[i])

    results_exp = experiment_des(parties, votes, seats, K, lostvals, identifier=year)
    results += results_exp[0]

    w = np.array(votes)
    u = np.array(results[0][-2])
    votes_brb = w + u
    #print(votes_brb)
    #print(parties)
    for i in range(0,len(votes)):
        E_balanced_des[parties[i]] = votes_brb[i]
    #print(E)
    return [E_balanced_des, results, dhondt_allocation(E_balanced_des, T, K, prefer=P)[P], results_exp[1]]



def balanced_prevented_dhondt():
    removed_votes_bal = open(target_dir + P + "-DHondt-RemovedVotes-balanced.dat", "w")

    under_threshold = T - E[P]

    print("%.5f %s %f %f" % (0, P, E[P], no_manipulation), file=removed_votes_bal)

    lostvals = list(range(1, 2))

    if no_manipulation == 0:
        print("%s %f %s" % ( P, no_manipulation, "bribery impossible"),
            file=removed_votes_bal)
    else:
        results = destructive_exp_single(lostvals)
        E_balanced_des = results[0]
        print("%.5f %s %f %f %f" % (results[1][0][-1], P, no_manipulation, results[2], results[1][0][-1]),
          file=removed_votes_bal)
        print("%s" % E_balanced_des.votealloc, file=removed_votes_bal)

        res = dhondt_allocation(E, T, K, prefer=P)[P]
        R = F.get_x_best_party(1)
        results_con = dhondt_allocation(E_balanced_des, T, K, prefer=P)[P]
        while results_con == res - 1:
            E_balanced_des[P] += 1
            E_balanced_des[R] -= 1
            results_con = dhondt_allocation(E_balanced_des, T, K, prefer=P)[P]

            print("%s %f %.5f %s %f %f" % (results[-1], res, E_balanced_des[P], P, results_con, under_threshold),
                    file=removed_votes_bal)

        B = -results[1][0][-1]
        if B != 0:
            x1 = B/(E.num_votes())
            x2 = 1-x1
            x3 = B/E[P]
            x4 = 1-x3
            print("%.5f %.5f %.5f %.5f" % (x1,x2,x3,x4), file=removed_votes_bal)
        else:
            print("%.5f" % 0, file=removed_votes_bal)

    removed_votes_bal.close()


print("File: ", sys.argv[1])
print(sys.argv[3])

get_max_additional_seats_dhondt()
print("#1 done")

get_max_prevented_seats_dhondt()
print("#2 done")

balanced_additional_dhondt()
print("#3 done")

balanced_prevented_dhondt()
print("#4 done")




print("Finished")
