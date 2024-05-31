import pulp as plp
import numpy as np
import csv
import pprint
from collections import OrderedDict

# D'hondt apportionment in single district. Slow algorithm, calculates all needed quotients.
# votes: vector of votes
# seats: number of seats to allocate
def dhondt_single_district(votes, seats, t):
    votes_with_threshold = votes.copy()
    T = np.ceil(t * sum(votes)).copy()

    for i in range(0, len(votes_with_threshold)):
        if votes_with_threshold[i] < T:
            votes_with_threshold[i] = 0

    result = np.array([0 for _ in votes_with_threshold])
    divisions = np.array([float(v) for v in votes_with_threshold] * seats).reshape((seats, len(votes_with_threshold)))
    vector = np.array(range(1, seats+1))
    divisions /= vector[:, None]
    while seats > 0:
        y, x = np.unravel_index(divisions.argmax(), divisions.shape)
        divisions[y][x] = 0
        result[x] += 1
        seats -= 1
    return result
# output: vector of allocated seats in the same order as votes in vector of votes

# D'hondt for multi-district
# elections: python dict with full election data. see read_elections_from_file
def dhondt(elections):
    result = np.zeros(elections["parties"])
    for district in elections["districts"]:
        result += dhondt_single_district(district["votes"], district["seats"])
    return result
# output: vector of allocated seats in the same order as votes in vectors of votes in election data

# Zips vector of allocated seats with parties labels
def pretty_results(elections, results):
    return sorted(zip(elections["labels"], results), key=lambda t: t[1], reverse=True)

# Reads election from csv file to python dict
# filename: path to election csv file. format of csv file has to be the same as in ../data/ directory
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
# output: python dict with election data e. i.:
# {
#   districts: [
#       {
#            number: 1,
#            seats: 12,
#            votes: [400, 600, 20, 19, 10]
#       },
#       {
#            number: 2,
#            seats: 10,
#            votes: [4100, 6300, 120, 719, 210]
#       },
#   ],
#   parties: 5,
#   labels: ["PiS", "PO", "Nowoczesna", "PSL", "MN"]
# }
