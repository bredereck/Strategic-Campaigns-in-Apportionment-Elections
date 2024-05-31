#!/usr/bin/env python3
import random
import os
import sys
import math
from pathlib import Path
import datetime
from multiprocessing import Process

from Election import *
from Apportionment import *
from Campaigns1 import *

if len(sys.argv) < 5:
    print("Analysis.py <Dataset> <Seats> <target party (best, median, worst, secondbest, thirdbest)> <Threshold>")
    exit()



E = election_from_file(sys.argv[1])
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
else:
    print("Unknown option: %s"%target_party)
    exit()
print(str(E))
cur_time = datetime.datetime.now()
target_dir = "../DATA/%s/%s/%s/" % ('1', target_party, sys.argv[1].rsplit("/", 1)[1])
Path(target_dir).mkdir(parents=True, exist_ok=True)


def get_max_additional_seats_dhondt():
    added_votes = open(target_dir + "DHondt-AddedVotes-optimal.dat", "w")
    added_votes_strongest = open(target_dir + "DHondt-AddedVotes-strongest.dat", "w")
    added_votes_weakest = open(target_dir + "DHondt-AddedVotes-weakest.dat", "w")

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
        E = election_from_file(sys.argv[1])
        cbb = constructive_bribery(E, T, K, P, i, B)
        if cbb[0] not in [True, False]:
            print("%s" % con, file=added_votes)
            print("%.5f %s %f %f %f %s" % (B,P,no_manipulation,cbb[1],cbb[0][P], cbb[0]), file=added_votes)
            goal = cbb[1]
        if cbb[0] in [True, False]:
            con = [B,P,goal]
    print("%s" % con, file=added_votes)

    goal = no_manipulation
    con_str = [0, P, goal]
    for B in range(0, rest_votes + 1):
        if goal >= no_manipulation + 1:
            break
        E = election_from_file(sys.argv[1])
        v = E[P]
        cbb_str = constructive_bribery_from_strongest(E, T, K, P, i, B)
        if cbb_str[0] not in [True, False]:
            print("%s" % con_str, file=added_votes_strongest)
            print("%.5f %s %f %f %f %s" % (B,P,no_manipulation,cbb_str[1],cbb_str[0][P]-v, cbb_str[0]), file=added_votes_strongest)
            goal = cbb_str[1]
        if cbb_str[0] in [True, False]:
            con_str = [B,P,goal]
    print("%s" % con_str, file=added_votes_strongest)

    goal = no_manipulation
    con_wea = [0, P, goal]
    for B in range(0, rest_votes + 1):
        if goal >= no_manipulation + 1:
            break
        E = election_from_file(sys.argv[1])
        v = E[P]
        cbb_wea = constructive_bribery_from_weakest(E, T, K, P, i, B)
        if cbb_wea[0] not in [True, False]:
            print("%s" % con_wea, file=added_votes_weakest)
            print("%.5f %s %f %f %f %s" % (B, P, no_manipulation, cbb_wea[1], cbb_wea[0][P] - v, cbb_wea[0]),
                      file=added_votes_weakest)
            goal = cbb_wea[1]
        if cbb_wea[0] in [True, False]:
            con_wea = [B, P, goal]
    print("%s" % con_wea, file=added_votes_weakest)

    added_votes.close()
    added_votes_strongest.close()
    added_votes_weakest.close()


def get_max_prevented_seats_dhondt():
    removed_votes = open(target_dir + "DHondt-RemovedVotes-optimal.dat", "w")
    removed_votes_str = open(target_dir + "DHondt-RemovedVotes_strongest.dat", "w")
    removed_votes_wea = open(target_dir + "DHondt-RemovedVotes_weakest.dat", "w")

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
    else:
        i = no_manipulation - 1
        for B in range(0,E[P]+1):
            if goal <= no_manipulation-1:
                break
            E = election_from_file(sys.argv[1])
            dbb = destructive_bribery(E, T, K, P, i, B)
            if dbb[0] not in [True, False]:
                print("%s" % con, file=removed_votes)
                print("%.5f %s %f %f %f %s" % (B, P,no_manipulation, dbb[1], dbb[0][P], dbb[0]), file=removed_votes)
                goal = dbb[1]
            if dbb[0] in [True,False]:
                con = [B, P, goal]
        print("%s" % con, file=removed_votes)

        goal = no_manipulation
        con_str = [0, P, goal]
        for B in range(0, E[P] + 1):
            if goal <= no_manipulation - 1:
                break
            E = election_from_file(sys.argv[1])
            v = E[P]
            dbb_str = destructive_bribery_to_strongest(E, T, K, P, i, B)
            if dbb_str[0] not in [True, False]:
                print("%s" % con_str, file=removed_votes_str)
                print("%.5f %s %f %f %f %s" % (B, P, no_manipulation, dbb_str[1], v-dbb_str[0][P], dbb_str[0]),
                          file=removed_votes_str)
                goal = dbb_str[1]
            if dbb_str[0] in [True, False]:
                con_str = [B, P, goal]
        print("%s" % con_str, file=removed_votes_str)

        goal = no_manipulation
        con_wea = [0, P, goal]
        for B in range(0, E[P] + 1):
            if goal <= no_manipulation - 1:
                break
            E = election_from_file(sys.argv[1])
            v = E[P]
            dbb_wea = destructive_bribery_to_weakest(E, T, K, P, i, B)
            if dbb_wea[0] not in [True, False]:
                print("%s" % con_wea, file=removed_votes_wea)
                print("%.5f %s %f %f %f %s" % (B, P, no_manipulation, dbb_wea[1], v-dbb_wea[0][P], dbb_wea[0]),
                          file=removed_votes_wea)
                goal = dbb_wea[1]
            if dbb_wea[0] in [True, False]:
                con_wea = [B, P, goal]
        print("%s" % con_wea, file=removed_votes_wea)

    removed_votes.close()
    removed_votes_str.close()
    removed_votes_wea.close()


get_max_additional_seats_dhondt()
get_max_prevented_seats_dhondt()

#no_manipulation = dhondt_allocation(E, T, K, prefer=P)[P]
#goal = no_manipulation
#print(int(E.num_votes()/5))
#print(P)
#print(E[P])
#print(dhondt_allocation(E, T, K, prefer=P)[P])
#x = destructive_bribery_to_strongest(E, T, K, P, no_manipulation + 1, int(E.num_votes()/5))
#print(x[0])
#print(x[1])


print("Finished")
