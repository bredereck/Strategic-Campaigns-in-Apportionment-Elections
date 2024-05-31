#!/usr/bin/python
from __future__ import division
import os
import elections_utils as elutils 
import copy
import random
import dhondt_ilp as dILP
import knapsack_ilp as kILP
import time
import bribery
import RunExperiment3_multi
import statistics
import numpy as np


def experiment(votes, k):
    divisors = range(1, 200)
    origresult = bribery.dhondt_app(votes, k)
    
    for brbparty in range(len(votes)):
        # permute votes so that party 0 bribes
        permvotes = ([votes[brbparty]] + votes[:brbparty]
                     + votes[brbparty+1:])

        for gainseats in range(1, 11):
            target = origresult[brbparty]+gainseats

            # Abstainers
            abstainers = bribery.findoptimalabstainers(
                permvotes, k, target, divisors)
            # print("Party {} can gain {} seats by convincing {} abstainers".format(
            #    brbparty, gainseats, abstainers))

            # Bribery
            brb = bribery.findoptimalbribery(
                permvotes, k, target, divisors, ilp=True)
            # print("Party {} can gain {} seats with {} bribes".format(
            #     brbparty, gainseats, brb[0]), end="")
            # print(" (bribery  = {})".format(brb))
            PP = [permvotes[i]+brb[i] for i in range(len(permvotes))]
            AA = bribery.dhondt_app(PP, k)
            if AA[0] - origresult[brbparty] < gainseats:
                print(target, AA, origresult)
                print(PP)
                raise Exception

            # print("bribery gain = {}".format(abstainers / brb[0]))

            print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(
                len(votes), votes, k, gainseats, brbparty, brb, abstainers,
                brb[0], abstainers / brb[0]))


def get_elections():

 def datasets_files():
  return {
   #"polish_election_2011_filtered.csv": "polish11"
   #,"polish_election_2015_filtered.csv": "polish15"
   #,"polish_election_2019_filtered.csv": "polish19"
      # ,
       #"polishelection2023.csv": "polish23",
       "portugalelection2024.csv": "portugal24"#,
       #"argentinaelection2021.csv": "argentina21"
  }
 def get_dataset_path(datasetFile):
  return os.path.join("..", "data", datasetFile)

 electionsList = []
 for dataset in datasets_files().items():
   electionsList.append((elutils.read_elections_from_file(get_dataset_path(dataset[0])), dataset[1]))
 return electionsList


def get_merged_election(mergesCount, initElection):

  def merge_districts(fromElection):
    election = copy.copy(fromElection)
    districtsCount = len(election["districts"])
    districts = [district["number"] for district in fromElection["districts"]]
    toMerge = random.sample(population = districts, k=2)
    for district in election["districts"]:
      if district["number"] == toMerge[0]:
        districtA = district
      if district["number"] == toMerge[1]:
        districtB = district
    election["districts"].remove(districtA)
    election["districts"].remove(districtB)
    newDistrict = {
        "number" : min(districtA["number"], districtB["number"])
        ,"seats" : districtA["seats"] + districtB["seats"]
        ,"votes" : list(map(sum, zip(districtA["votes"], districtB["votes"])))
        }
    election["districts"].append(newDistrict)
    return election

  currentElection = copy.deepcopy(initElection)
  for _ in range(mergesCount):
    currentElection = merge_districts(currentElection)
  return currentElection


def get_count_party_seats_per_district(party, election,t):
  seatsPerDistrict = dict()
  for district in election["districts"]:
    seatsPerDistrict[district["number"]] = elutils.dhondt_single_district(
        district["votes"], district["seats"],t)[party]
  return seatsPerDistrict


def get_count_party_all_seats(party, election,t):
  return sum( districtSeats for districtSeats in
      get_count_party_seats_per_district(party, election,t).values())

def get_total_seats(election):
  return sum([district["seats"] for district in election["districts"]])


def compute_election_statistics(party, election, maxBudget=None, maxSeats=None):
  seatPrices = dict()
  seatsTotal = get_total_seats(election) 
  partySeatsPerDistrict = get_count_party_seats_per_district(party, election,t)
  maxBudgetAchieved = False
  maxSeatsAchieved = False
  for district in election["districts"]:
    seatPrices[district["number"]]=[0]+[-1]*(seatsTotal-1)
  for district in election["districts"]:
    initialSeats = partySeatsPerDistrict[district["number"]]
    for desiredSeats in range(initialSeats + 1, district["seats"] + 1):
      try:
        price = dILP.solve(district["votes"], district["seats"],
           party, desiredSeats, range(1, seatsTotal+1))[0]
      except Exception:
        break
      else:
        seatPrices[district["number"]][desiredSeats-initialSeats]=price
        #if there is one max given, stop computing when one max reached
        #if there are two maxes, stop computing only when both maxes reached
        if maxSeatsAchieved != False and desiredSeats >= maxSeats:
          maxSeatsAchieved = True
        if maxBudgetAchieved != False and price >= maxBudget:
          maxBudgetAchieved = True
        if (maxBudgetAchieved and maxSeatsAchieved) \
         or (maxBudgetAchieved and maxSeatsAchieved == None) \
         or (maxSeatsAchieved == None and maxBudgetAchieved):
          break
  return seatPrices

def gain_prices_to_knapsack(pricesOfGain):
  knapsackInput = dict()
  for districtNr in pricesOfGain.keys():
    prices = list(filter(lambda x: x!=-1, pricesOfGain[districtNr]))
    gains = range(0,len(prices))
    knapsackInput[districtNr]=list(zip(gains, prices))
  return knapsackInput


def pretty_knapsack_data(knapsackInput):
  def print_chunk(datachunks):
    districtsNrList = list(set([chunk[0] for chunk in datachunks]))
    districtsNrList.sort()
    print("     " + ("{: ^11}"*len(districtsNrList)).format(*districtsNrList))
    #print datachunks
    maxGain = max([len(chunk[1]) for chunk in datachunks])
    for currentGain in range(maxGain):
      gainPricesString=""
      for chunk in datachunks:
        try:
          currentPrice = chunk[1][currentGain][1]
        except IndexError:
          currentPrice = "---"
        gainPricesString += "{: >11}".format(currentPrice)
      print(("{: <3}: "+ gainPricesString).format(currentGain))

  counter = 0
  datachunks = []
  chunkSize = 8
  for districtNrAndCharacteristic in knapsackInput.items():
    counter = counter+1
    if counter <= chunkSize:
      datachunks.append(districtNrAndCharacteristic)
    if counter == chunkSize:
      print_chunk(datachunks)
      datachunks = []
      counter = 0
  if datachunks != []:
    print_chunk(datachunks)

def build_header(trialsNumber):
  constantPart = "{},{},{},{},{},{},{}".format(
     "Election Name"
     ,"Districts"
     ,"Preferred Party"
     ,"Merges Count"
     ,"Seats to Majority"
     ,"Average Time [seconds]"
     ,"Average Budget to Majority"
    )
  variablePart = ",{}"*trialsNumber
  variablePart = variablePart.format(*["Budget to Majority-Trial {}".format(i)
    for i in range(1, trialsNumber+1)])
  return constantPart+variablePart


def build_line(electionName, districtsCount, prefParty, mergesCount, 
    seatsToMajority, avgTime, trialScores):
  constantPart = "{},{},{},{},{},{}".format(
     electionName, districtsCount, prefParty, mergesCount, seatsToMajority, avgTime)
  variablePart = ",{}"*(len(trialScores)+1)
  avgScore = float(sum(filter(lambda x: x != None, trialScores))) / float(len(trialScores))
  variablePart = variablePart.format(avgScore, *trialScores)
  return constantPart+variablePart


if __name__ == "__main__":
  budget=400000
  preferredParty=0
  trialsCount= 10
  print(build_header(trialsCount))
  finalScores = []
  all_election_none_numbers = []
  for electionDataset in get_elections():
    all_none_number = []
    totalScore = []
    election = electionDataset[0]
    name = electionDataset[1]

    if name=='polish23':
        t=0.05
    elif name=='portugal24':
        t=0
    elif name=='argentina21':
        t=0.03
    else:
        t=0

    districtsCount = len(election["districts"])
    majorityQuota = int(get_total_seats(election)/2) + 1

    for mergesCount in range(districtsCount):
      trialScores = []
      trial_none_number = 0
      for _ in range(0,trialsCount):
            currentElection = get_merged_election(mergesCount, election)
            party = election["labels"][preferredParty]
            #seatsForParty = get_count_party_all_seats(preferredParty, election)
            seatsForParty = get_count_party_all_seats(preferredParty, currentElection,t)
            seatsToMajority = int(majorityQuota - seatsForParty)
            #print(seatsToMajority)

    # constructive
            pricesOfGain = compute_election_statistics(preferredParty,
                currentElection, None, seatsToMajority)
            #print(pricesOfGain)
            #print("Budget: {}, Districts: {}, Majority requirement: {}".format(
            #  budget, districtsCount-mergesCount, seatsToMajority))
            knapsackInput = gain_prices_to_knapsack(pricesOfGain)
            #pretty_knapsack_data(knapsackInput)
            #print(knapsackInput_test)


            #knapsackInput = RunExperiment3_multi.get_max_additional_seats_dhondt(currentElection, party,t, seatsToMajority)
            #print(knapsackInput)


    # destructive
            #halfSeats = int(seatsForParty/2)
            #print(seatsForParty)
            #print(halfSeats)

            #knapsackInput_des = RunExperiment3_multi.get_max_prevented_seats_dhondt(currentElection, party, t, halfSeats)
            #print(knapsackInput_des)


            #print("No. of gained seats: {}".format(kILP.solve_for_given_budget(knapsackInput, budget)))
            #print("Price of majority:{}".format(kILP.solve_for_given_value(knapsackInput,seatsToMajority)))
            priceOfMajority = kILP.solve_for_given_value(knapsackInput, seatsToMajority)
            if priceOfMajority != None:
                trialScores.append(float(priceOfMajority)/seatsToMajority)
            else:
                trial_none_number += 1


      all_none_number.append(trial_none_number)
      if len(trialScores)>0:
          print(statistics.mean(trialScores))
          totalScore.append(statistics.mean(trialScores))
      else:
          totalScore.append(None)
      #print(totalScore)
          #avgTime = int(time.time() - start) / trialsCount
          #print(build_line(name, districtsCount-mergesCount, preferredParty,
          #    mergesCount, seatsToMajority, int(avgTime), trialScores))
    #      print("Time: {}".format(time.time() - start))

    all_election_none_numbers.append(all_none_number)
    finalScores.append(totalScore)

  print(finalScores)
  print(all_election_none_numbers)