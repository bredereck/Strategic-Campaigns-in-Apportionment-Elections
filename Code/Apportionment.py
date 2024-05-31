from Election import Election
from collections import defaultdict

"""
  E = Election
  T = Threshold
  K = Seats total
  prefer = Party that should be preferred in case of ties. Important in campaigns.
"""

def dhondt_allocation(E: Election, T: int, K: int, prefer = None):
    seatalloc = {p : 0 for p in E.parties()}
    E = E.apply_threshold(T)
    if E.is_empty():
        raise ValueError("No party is above threshold.")
    if prefer not in E.parties():
        prefer = None
    divisorsequences = dict()
    for party in E.parties():
        divisorsequences[party] = [E[party] / i for i in range(1, K+1)]

    seats_left = K
    while seats_left > 0:
        best_divisor = max([divisorsequence[0] for divisorsequence in divisorsequences.values()])
        if prefer != None and divisorsequences[prefer][0] == best_divisor:
            best_party = prefer
        else:
            best_party = [p for p in E.parties() if divisorsequences[p][0] == best_divisor][0]
        divisorsequences[best_party].pop(0)
        seatalloc[best_party] += 1
        seats_left -= 1
    return seatalloc

def sainte_lague_allocation(E: Election, T: int, K: int, prefer = None):
    seatalloc = {p : 0 for p in E.parties()}
    E = E.apply_threshold(T)
    if E.is_empty():
        raise ValueError("No party is above threshold.")
    if prefer not in E.parties():
        prefer = None
    divisorsequences = dict()
    for party in E.parties():
        divisorsequences[party] = [E[party] / (2*i-1) for i in range(1, K+1)]

    seats_left = K
    while seats_left > 0:
        best_divisor = max([divisorsequence[0] for divisorsequence in divisorsequences.values()])
        if prefer != None and divisorsequences[prefer][0] == best_divisor:
            best_party = prefer
        else:
            best_party = [p for p in E.parties() if divisorsequences[p][0] == best_divisor][0]
        divisorsequences[best_party].pop(0)
        seatalloc[best_party] += 1
        seats_left -= 1
    return seatalloc
