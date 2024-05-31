#!/usr/bin/env python3
import random
import os
import sys
from pathlib import Path
import datetime
from multiprocessing import Process

from Election import *
from Apportionment import *
from Campaigns2 import *

if len(sys.argv) < 5:
    print("Analysis.py <Dataset> <Seats> <target party (best, median, worst, secondbest, thirdbest)> <Budget>")
    exit()

E = election_from_file(sys.argv[1])
K = int(sys.argv[2])
BUDGET = float(sys.argv[4])
B = int(BUDGET*E.num_votes())
target_party = sys.argv[3]
if target_party == "best":
    P = E.get_x_best_party(1)
elif target_party == "median":
    P = E.get_x_median_party(E.num_parties//2)
elif target_party == "worst":
    P = E.get_worst_party()
elif target_party == "secondbest":
    P = E.get_x_best_party(2)
elif target_party == "thirdbest":
    P = E.get_x_best_party(3)
else:
    print("Unknown option: %s"%target_party)
    exit()
#print(str(E))
cur_time = datetime.datetime.now()
target_dir = "../DATA/%s/%s/%s/" % (sys.argv[4], target_party, sys.argv[1].rsplit("/", 1)[1])
Path(target_dir).mkdir(parents=True, exist_ok=True)









def get_max_additional_seats_dhondt():
    normal_seats = open(target_dir + "DHondt-NormalSeats.dat", "w")
    bribery = open(target_dir + "DHondt-Bribery-Max.dat", "w")
    diff = open(target_dir + "DHondt-Bribery-Max-Diff.dat", "w")
    for t in (i*0.0005 for i in range(250)):
        T = int(t*E.num_votes())
        no_manipulation = dhondt_allocation(E, T, K, prefer=P)[P]
        print("%.5f %d"%(t*100, no_manipulation), file=normal_seats)

        for i in range(no_manipulation+1, K+1):
            if not constructive_bribery(E, T, K, P, i, B):
                print("%.5f %d"%(t*100, i-1), file=bribery)
                print("%.5f %d"%(t*100, i-1-no_manipulation), file=diff)
                break
    normal_seats.close()
    bribery.close()
    diff.close()

def get_max_prevented_seats_dhondt():
    bribery = open(target_dir + "DHondt-Bribery-Min.dat", "w")
    diff = open(target_dir + "DHondt-Bribery-Min-Diff.dat", "w")
    for t in (i*0.0005 for i in range(250)):
        T = int(t*E.num_votes())
        no_manipulation = dhondt_allocation(E, T, K, prefer=P)[P]

        for i in range(no_manipulation, -1, -1):
            if not destructive_bribery(E, T, K, P, i, B):
                print("%.5f %d"%(t*100, i+1), file=bribery)
                print("%.5f %d"%(t*100, no_manipulation - i-1), file=diff)
                break
            elif i == 0:
                print("%.5f %d"%(t*100, 0), file=bribery)
                print("%.5f %d"%(t*100, no_manipulation), file=diff)
    bribery.close()
    diff.close()

def get_max_additional_seats_sainte_lague():
    normal_seats = open(target_dir + "SainteLague-NormalSeats.dat", "w")
    bribery = open(target_dir + "SainteLague-Bribery-Max.dat", "w")
    diff = open(target_dir + "SainteLague-Bribery-Max-Diff.dat", "w")
    for t in (i*0.0005 for i in range(250)):
        T = int(t*E.num_votes())
        no_manipulation = sainte_lague_allocation(E, T, K, prefer=P)[P]
        print("%.5f %d"%(t*100, no_manipulation), file=normal_seats)

        for i in range(no_manipulation+1, K+1):
            if not constructive_bribery(E, T, K, P, i, B, allocation_method=sainte_lague_allocation):
                print("%.5f %d"%(t*100, i-1), file=bribery)
                print("%.5f %d"%(t*100, i-1-no_manipulation), file=diff)
                break
    normal_seats.close()
    bribery.close()
    diff.close()

def get_max_prevented_seats_sainte_lague():
    bribery = open(target_dir + "SainteLague-Bribery-Min.dat", "w")
    diff = open(target_dir + "SainteLague-Bribery-Min-Diff.dat", "w")
    for t in (i*0.0005 for i in range(250)):
        T = int(t*E.num_votes())
        no_manipulation = sainte_lague_allocation(E, T, K, prefer=P)[P]

        for i in range(no_manipulation, -1, -1):
            if not destructive_bribery(E, T, K, P, i, B, allocation_method=sainte_lague_allocation):
                print("%.5f %d"%(t*100, i+1), file=bribery)
                print("%.5f %d"%(t*100, no_manipulation - i-1), file=diff)
                break
            elif i == 0:
                print("%.5f %d"%(t*100, 0), file=bribery)
                print("%.5f %d"%(t*100, no_manipulation), file=diff)
    bribery.close()
    diff.close()


get_max_additional_seats_dhondt()
get_max_prevented_seats_dhondt()

get_max_additional_seats_sainte_lague()
get_max_prevented_seats_sainte_lague()

print("Finished")
