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
    print("Analysis.py <Dataset> <Seats> <target party (all)> <Threshold>")
    exit()


E = election_from_file(sys.argv[1])
K = int(sys.argv[2])
t = float(sys.argv[4])
T = math.ceil(t*E.num_votes())
target_party = sys.argv[3]
if target_party != "all":
    print("Unknown option: %s"%target_party)
    exit()
cur_time = datetime.datetime.now()
target_dir = "../DATA/%s/%s/" % (target_party.replace("/","_"), sys.argv[1].rsplit("/", 1)[1])
Path(target_dir).mkdir(parents=True, exist_ok=True)


# contructive bribery - optimal, from the strongest rival(s), from the weakest rival(s)
def get_max_additional_seats_dhondt(P):
    added_votes = open(target_dir  + P.replace("/","_") + "-DHondt-AddedVotes-optimal.dat", "w")
    added_votes_strongest = open(target_dir  + P.replace("/","_") + "-DHondt-AddedVotes-strongest.dat", "w")
    added_votes_weakest = open(target_dir  + P.replace("/","_") + "-DHondt-AddedVotes-weakest.dat", "w")


    E = election_from_file(sys.argv[1])
    no_manipulation = dhondt_allocation(E, T, K, prefer=P)[P]
    goal = no_manipulation
    con = [0,P,no_manipulation]

    F = E.remove(P)
    rest_votes = int(F.num_votes())
    i = no_manipulation + 1

    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=added_votes)
    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=added_votes_strongest)
    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=added_votes_weakest)


    for B in range(0,rest_votes+1):
        if goal>=no_manipulation+1:
            break
        E_optimal_con = election_from_file(sys.argv[1])
        cbb = constructive_bribery(E_optimal_con, T, K, P, i, B)
        if cbb[0] not in [True, False]:
            print("%s" % con, file=added_votes)
            print("%.5f %s %f %f %f %s" % (B,P,no_manipulation,cbb[1],cbb[0][P], cbb[0]), file=added_votes)
            goal = cbb[1]
            x_opt_con = float(B)
        if cbb[0] in [True, False]:
            con = [B,P,goal]
    print("%s" % con, file=added_votes)
    print("optimal done")
    added_votes.close()

    goal = no_manipulation
    con_str = [0, P, goal]
    for B in range(0, rest_votes + 1):
        if goal >= no_manipulation + 1:
            break
        E_strongest_con = election_from_file(sys.argv[1])
        cbb_str = constructive_bribery_from_strongest(E_strongest_con, T, K, P, i, B)
        if cbb_str[0] not in [True, False]:
            print("%s" % con_str, file=added_votes_strongest)
            print("%.5f %s %f %f %f %s" % (B,P,no_manipulation,cbb_str[1],cbb_str[0][P]-E[P], cbb_str[0]), file=added_votes_strongest)
            goal = cbb_str[1]
            x_str_con = float(B)
        if cbb_str[0] in [True, False]:
            con_str = [B,P,cbb_str[1]]
    print("%s" % con_str, file=added_votes_strongest)
    print("strongest done")
    added_votes_strongest.close()

    goal = no_manipulation
    con_wea = [0, P, goal]
    for B in range(0, rest_votes + 1):
        if goal >= no_manipulation + 1:
            break
        E_weakest_con = election_from_file(sys.argv[1])
        cbb_wea = constructive_bribery_from_weakest(E_weakest_con, T, K, P, i, B)
        if cbb_wea[0] not in [True, False]:
            print("%s" % con_wea, file=added_votes_weakest)
            print("%.5f %s %f %f %f %s" % (B, P, no_manipulation, cbb_wea[1], cbb_wea[0][P] - E[P], cbb_wea[0]),
                      file=added_votes_weakest)
            goal = cbb_wea[1]
            x_wea_con = float(B)
        if cbb_wea[0] in [True, False]:
            con_wea = [B, P, cbb_wea[1]]
    print("%s" % con_wea, file=added_votes_weakest)
    added_votes_weakest.close()
    print("weakest done")

    return [x_opt_con,x_str_con,x_wea_con]


# destructive bribery - optimal, to the strongest candidate, to the weakest candidate
def get_max_prevented_seats_dhondt(P):
    removed_votes = open(target_dir + P.replace("/","_") +"-DHondt-RemovedVotes-optimal.dat", "w")
    removed_votes_str = open(target_dir  + P.replace("/","_") + "-DHondt-RemovedVotes_strongest.dat", "w")
    removed_votes_wea = open(target_dir  + P.replace("/","_") + "-DHondt-RemovedVotes_weakest.dat", "w")

    E = election_from_file(sys.argv[1])
    no_manipulation = dhondt_allocation(E, T, K, prefer=P)[P]
    goal = no_manipulation
    con = [0, P, goal]

    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=removed_votes)
    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=removed_votes_str)
    print("%.5f %s %f %f %f" % (0, P, E[P], no_manipulation, goal), file=removed_votes_wea)

    if no_manipulation == 0:
        print("%.5f %s %f %s" % (0, P, no_manipulation, "no bribery possible"), file=removed_votes)
        print("%.5f %s %f %s" % (0, P, no_manipulation, "no bribery possible"), file=removed_votes_str)
        print("%.5f %s %f %s" % (0, P, no_manipulation, "no bribery possible"), file=removed_votes_wea)
        x_opt_des = 10000
        x_str_des = 10000
        x_wea_des = 10000
    else:
        i = no_manipulation - 1
        for B in range(0,E[P]+1):
            if goal <= no_manipulation-1:
                break
            E_optimal_des = election_from_file(sys.argv[1])
            dbb = destructive_bribery(E_optimal_des, T, K, P, i, B)
            if dbb[0] not in [True, False]:
                print("%s" % con, file=removed_votes)
                print("%.5f %s %f %f %f %s" % (B, P,no_manipulation, dbb[1], dbb[0][P], dbb[0]), file=removed_votes)
                goal = dbb[1]
                x_opt_des = float(B)
            if dbb[0] in [True,False]:
                con = [B, P, goal]
        print("%s" % con, file=removed_votes)

        goal = no_manipulation
        con_str = [0, P, goal]
        for B in range(0, E[P] + 1):
            if goal <= no_manipulation - 1:
                break
            E_strongest_des = election_from_file(sys.argv[1])
            dbb_str = destructive_bribery_to_strongest(E_strongest_des, T, K, P, i, B)
            if dbb_str[0] not in [True, False]:
                print("%s" % con_str, file=removed_votes_str)
                print("%.5f %s %f %f %f %s" % (B, P, no_manipulation, dbb_str[1], E[P]-dbb_str[0][P], dbb_str[0]),
                          file=removed_votes_str)
                goal = dbb_str[1]
                x_str_des = float(B)
            if dbb_str[0] in [True, False]:
                con_str = [B, P, dbb_str[1]]
        print("%s" % con_str, file=removed_votes_str)

        goal = no_manipulation
        con_wea = [0, P, goal]
        for B in range(0, E[P] + 1):
            if goal <= no_manipulation - 1:
                break
            E_weakest_des = election_from_file(sys.argv[1])
            dbb_wea = destructive_bribery_to_weakest(E_weakest_des, T, K, P, i, B)
            if dbb_wea[0] not in [True, False]:
                print("%s" % con_wea, file=removed_votes_wea)
                print("%.5f %s %f %f %f %s" % (B, P, no_manipulation, dbb_wea[1], E[P]-dbb_wea[0][P], dbb_wea[0]),
                          file=removed_votes_wea)
                goal = dbb_wea[1]
                x_wea_des = float(B)
            if dbb_wea[0] in [True, False]:
                con_wea = [B, P, dbb_wea[1]]
        print("%s" % con_wea, file=removed_votes_wea)

    removed_votes.close()
    removed_votes_str.close()
    removed_votes_wea.close()

    return [x_opt_des,x_str_des,x_wea_des]


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


def constructive_exp_single(gainvals, P):
    E = election_from_file(sys.argv[1])
    year = sys.argv[1][-4:]
    results = []
    F = E.remove(P)
    votes = [E[P]]
    parties = [P]
    votes_above = []
    parties_above = []
    seats = [dhondt_allocation(E, T, K, prefer=P)[P]]
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
    #print(parties)
    #print(votes)
    #print(parties_above)
    #print(votes_above)
    #print(seats)
    #print(divisor_app(votes, divisors, K, T))

    results_exp = experiment(parties, votes, seats, K, gainvals, identifier=year)
    results += results_exp[0]

    w = np.array(votes)
    u = np.array(results[0][-2])
    votes_brb = w + u
    #print(votes_brb)
    #print(parties)
    for i in range(0,len(votes)):
        E[parties[i]] = votes_brb[i]
    return [E, results, dhondt_allocation(E, T, K, prefer=P)[P], results_exp[1]]


def balanced_additional_dhondt(P):
    added_votes_bal = open(target_dir + P.replace("/","_") + "-DHondt-AddedVotes-balanced.dat", "w")

    E_balanced_con = election_from_file(sys.argv[1])
    no_manipulation = dhondt_allocation(E_balanced_con, T, K, prefer=P)[P]
    under_threshold = T - E_balanced_con[P]

    print("%.5f %s %f %f" % (0, P, E_balanced_con[P], no_manipulation), file=added_votes_bal)

    gainvals = list(range(1, 2))
    results = constructive_exp_single(gainvals, P)
    E_balanced_con = results[0]

    print("%.5f %s %f %f %f %s" % ( results[1][0][-1], P, no_manipulation, results[2], results[1][0][-1], E_balanced_con.votealloc),
          file=added_votes_bal)

    E = election_from_file(sys.argv[1])
    x_bal_con = float(results[1][0][-1])

    F = E_balanced_con.remove(P)
    res = dhondt_allocation(E, T, K, prefer=P)[P]
    R = F.get_x_best_party(1)
    results_con = dhondt_allocation(E_balanced_con, T, K, prefer=P)[P]
    while results_con == res + 1:
        E_balanced_con[P] -= 1
        E_balanced_con[R] += 1
        results_con = dhondt_allocation(E_balanced_con, T, K, prefer=P)[P]

        print("%s %f %.5f %s %f %f" % (results[-1], res, E_balanced_con[P] - E[P], P, results_con, under_threshold),
              file=added_votes_bal)

    print("%s %f %.5f %s %f %f" % (results[-1], res, E_balanced_con[P], P, results_con, under_threshold),
          file=added_votes_bal)

    added_votes_bal.close()
    return x_bal_con


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



def destructive_exp_single(lostvals,P):
    E = election_from_file(sys.argv[1])
    year = sys.argv[1][-4:]
    results = []
    F = E.remove(P)
    votes = [E[P]]
    parties = [P]
    votes_above = []
    parties_above = []
    seats = [dhondt_allocation(E, T, K, prefer=P)[P]]
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
        E[parties[i]] = votes_brb[i]
    #print(E)
    return [E, results, dhondt_allocation(E, T, K, prefer=P)[P], results_exp[1]]


def balanced_prevented_dhondt(P):
    removed_votes_bal = open(target_dir + P.replace("/","_") + "-DHondt-RemovedVotes-balanced.dat", "w")

    E_balanced_des = election_from_file(sys.argv[1])
    no_manipulation = dhondt_allocation(E_balanced_des, T, K, prefer=P)[P]
    under_threshold = T - E_balanced_des[P]
    E = election_from_file(sys.argv[1])

    print("%.5f %s %f %f" % (0, P, E_balanced_des[P], no_manipulation), file=removed_votes_bal)

    lostvals = list(range(1, 2))

    if no_manipulation == 0:
        print("%s %f %s" % ( P, no_manipulation, "bribery impossible"),
            file=removed_votes_bal)
        x_bal_des = 10000
    else:
        results = destructive_exp_single(lostvals,P)
        E_balanced_des = results[0]
        print("%.5f %s %f %f %f %s" % (results[1][0][-1], P, no_manipulation, results[2], results[1][0][-1], E_balanced_des.votealloc),
          file=removed_votes_bal)

        res = dhondt_allocation(E, T, K, prefer=P)[P]
        F = E_balanced_des.remove(P)
        R = F.get_x_best_party(1)
        results_con = dhondt_allocation(E_balanced_des, T, K, prefer=P)[P]
        while results_con == res - 1:
            E_balanced_des[P] += 1
            E_balanced_des[R] -= 1
            results_con = dhondt_allocation(E_balanced_des, T, K, prefer=P)[P]

            print("%s %f %.5f %s %f %f" % (results[-1], res, E_balanced_des[P], P, results_con, under_threshold),
                  file=removed_votes_bal)

        print("%s %f %.5f %s %f %f" % (results[-1], res, E_balanced_des[P], P, results_con, under_threshold),
          file=removed_votes_bal)


        x_bal_des = -float(results[1][0][-1])

    removed_votes_bal.close()
    return x_bal_des



print("File: ", sys.argv[1])
print(sys.argv[3])

eff_optimal_con = []
eff_strongest_con = []
eff_weakest_con = []
eff_balanced_con = []

eff_optimal_des = []
eff_strongest_des = []
eff_weakest_des = []
eff_balanced_des = []

E_contr = election_from_file(sys.argv[1])

for p in E_contr.parties():
    x = get_max_additional_seats_dhondt(p)
    eff_optimal_con.append(x[0])
    eff_strongest_con.append(x[1])
    eff_weakest_con.append(x[2])
    print(p, "#1 done")

    eff_balanced_con.append(balanced_additional_dhondt(p))
    print(p, "#3 done")

    if dhondt_allocation(E_contr, T, K, prefer=p)[p] > 0:  # We take into account only the parties where the bribery is possible.
        y = get_max_prevented_seats_dhondt(p)
        eff_optimal_des.append(y[0])
        eff_strongest_des.append(y[1])
        eff_weakest_des.append(y[2])
        print(p, "#2 done")

        eff_balanced_des.append(balanced_prevented_dhondt(p))
        print(p, "#4 done")



print(eff_optimal_con)
print(eff_strongest_con)
print(eff_weakest_con)
print(eff_balanced_con)

print(eff_optimal_des)
print(eff_strongest_des)
print(eff_weakest_des)
print(eff_balanced_des)

print("constructive")
print("optimal: ", np.mean(eff_optimal_con))
print("strongest: ", np.mean(eff_strongest_con))
print("weakest: ", np.mean(eff_weakest_con))
print("balanced: ", np.mean(eff_balanced_con))

print("destructive")
print("optimal: ", np.mean(eff_optimal_des))
print("strongest: ", np.mean(eff_strongest_des))
print("weakest: ", np.mean(eff_weakest_des))
print("balanced: ", np.mean(eff_balanced_des))


final = open(target_dir + "final.dat", "w")
print("%s %.5f" % ("constructive-optimal", np.mean(eff_optimal_con)), file=final)
print("%s" % eff_optimal_con, file=final)

print("%s %.5f" % ("constructive-strongest", np.mean(eff_strongest_con)), file=final)
print("%s" % eff_strongest_con, file=final)

print("%s %.5f" % ("constructive-weakest", np.mean(eff_weakest_con)), file=final)
print("%s" % eff_weakest_con, file=final)

print("%s %.5f" % ("constructive-balanced", np.mean(eff_balanced_con)), file=final)
print("%s" % eff_balanced_con, file=final)


print("%s %.5f" % ("destructive-optimal", np.mean(eff_optimal_des)), file=final)
print("%s" % eff_optimal_des, file=final)

print("%s %.5f" % ("destructive-strongest", np.mean(eff_strongest_des)), file=final)
print("%s" % eff_strongest_des, file=final)

print("%s %.5f" % ("destructive-weakest", np.mean(eff_weakest_des)), file=final)
print("%s" % eff_weakest_des, file=final)

print("%s %.5f" % ("destructive-balanced", np.mean(eff_balanced_des)), file=final)
print("%s" % eff_balanced_des, file=final)
final.close()

print("Finished")
