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
import bribery

from Election import *
from Apportionment import *
from Campaigns3_multi import *


# contructive bribery - optimal
def get_max_additional_seats_dhondt(election,P,t,seats_to_gain):
    votes_in_districts = []  # votes for each party in each district
    T = []  # thresholds in districts
    seats = []  # number of seats in districts
    for district in election["districts"]:
        seats.append(district["seats"])
        votes_in_districts.append(district["votes"])
        T.append(np.ceil(t * sum(district["votes"])))

    print(T)

    number_of_districts = len(T)
    parties = election['labels']


    knapsackInput = dict()

    for i in range(0,number_of_districts):
        print("district",i)
        minimal_budget_for_seats = [(0,0)]

        E_initial = election_from_file_multi(parties, votes_in_districts[i])
        #print(E_initial)
        initial_seats = int(dhondt_allocation(E_initial, T[i], seats[i], prefer=P)[P])
        current_seats = int(dhondt_allocation(E_initial, T[i], seats[i], prefer=P)[P])

        F = E_initial.remove(P)
        rest_votes = int(F.num_votes())

        possible_additional_seats = min(seats[i] - initial_seats,seats_to_gain)

        for j in range(0,possible_additional_seats+1):

            for B in range(0,rest_votes+1):
                if current_seats >= initial_seats+j:
                    break
                E_optimal_con = election_from_file_multi(parties, votes_in_districts[i])
                cbb = constructive_bribery(E_optimal_con, T[i], seats[i], P, initial_seats+j, B)
                if cbb[0] not in [True, False]:
                    minimal_budget_for_seats.append((j,-cbb[0][P]))
                    current_seats = cbb[1]
                    current_changes = cbb[0]

                    #print(j, -cbb[0][P])
                    E_new = election_from_file_multi(parties, votes_in_districts[i])
                    v = []
                    pn = []
                    for pa in E_new.parties():
                        E_new[pa] -= current_changes[pa]
                        v.append(E_new[pa])
                        pn.append(pa)
                    #print(E_new)
                    #print(v)
                    #print(pn)
                    #print(bribery.dhondt_app(v,seats[i]))
                    #print(dhondt_allocation(E_new, T[i], seats[i], prefer=P))

        #print(minimal_budget_for_seats)
        knapsackInput[i+1] = minimal_budget_for_seats.copy()

    return knapsackInput


# destructive bribery - optimal
def get_max_prevented_seats_dhondt(election,P,t,seats_to_lose):
    votes_in_districts = []  # votes for each party in each district
    T = []  # thresholds in districts
    seats = []  # number of seats in districts
    for district in election["districts"]:
        seats.append(district["seats"])
        votes_in_districts.append(district["votes"])
        T.append(np.ceil(t * sum(district["votes"])))

    print(T)

    number_of_districts = len(T)
    parties = election['labels']

    knapsackInput = dict()

    for i in range(0, number_of_districts):
        print("district", i)
        minimal_budget_for_seats = [(0, 0)]

        E_initial = election_from_file_multi(parties, votes_in_districts[i])
        print(E_initial)
        initial_seats = int(dhondt_allocation(E_initial, T[i], seats[i], prefer=P)[P])
        current_seats = int(dhondt_allocation(E_initial, T[i], seats[i], prefer=P)[P])

        if initial_seats > 0:

            votes_to_lose = int(E_initial[P])

            start_votes_no = 0

            possible_lost_seats = min(initial_seats, seats_to_lose)

            for j in range(0, possible_lost_seats+1):

                for B in range(start_votes_no, E_initial[P] + 1):
                    if current_seats <= initial_seats - j:
                        break
                    E_optimal_des = election_from_file_multi(parties, votes_in_districts[i])
                    dbb = destructive_bribery(E_optimal_des, T[i], seats[i], P, initial_seats-j, B)
                    if dbb[0] not in [True, False]:
                        start_votes_no = B
                        minimal_budget_for_seats.append((j, dbb[0][P]))
                        current_seats = dbb[1]
                        current_changes = dbb[0]

                        print(j, dbb[0][P])
                        E_new = election_from_file_multi(parties, votes_in_districts[i])
                        v = []
                        pn = []
                        for pa in E_new.parties():
                            E_new[pa] -= current_changes[pa]
                            v.append(E_new[pa])
                            pn.append(pa)
                        

        print(minimal_budget_for_seats)
        knapsackInput[i + 1] = minimal_budget_for_seats.copy()

    return knapsackInput


