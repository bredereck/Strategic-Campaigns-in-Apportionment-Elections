from Election import Election
from Apportionment import *
import copy
import numpy as np

"""
  E = Election
  T = Threshold
  K = Seats total
  P = Target party
  L = Target seats
  B = Budget
"""

#-----------------------------------------------------------------------------------
#                 Seats Count
#-----------------------------------------------------------------------------------
#

# D'hondt apportionment in single district. Slow algorithm, calculates all needed quotients.
# votes: vector of votes
# seats: number of seats to allocate
def dhondt_single_district(votes, seats, T):
    votes_with_threshold = votes.copy()

    for i in range(0,len(votes_with_threshold)):
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

def get_count_party_seats_per_district(party, election, T):
  seatsPerDistrict = dict()
  for i in range(0,len(election["districts"])):
    district = election["districts"][i]
    t = T[i]
    seatsPerDistrict[district["number"]] = dhondt_single_district(
        district["votes"], district["seats"], t)[party]
  #print(seatsPerDistrict)
  return seatsPerDistrict


def get_count_party_all_seats(party, election, T):
  return sum( districtSeats for districtSeats in
      get_count_party_seats_per_district(party, election, T).values())

#-----------------------------------------------------------------------------------
#                 The Campaigns
#-----------------------------------------------------------------------------------
#
# These are the actual campaigns. Each campaign will return a vector (implemented as
# a dict) if there exist a campaign, and return False otherwise.


def constructive_bribery(E, T, K, P, L, B, allocation_method = dhondt_allocation):

    #------------------------------------------------------------
    def get_gamma_cached():
        def phi_dhondt(x):
            assert x >= 0
            if x < T:
                return 0
            if x <= (E[P]+B)/L:
                return 0
            for l in range(K, 0, -1):
                if x/l > (E[P]+B)/L:
                    return l
        def phi_sainte_lague(x):
            assert x >= 0
            if x < T:
                return 0
            if x <= (E[P]+B)/(2*L-1):
                return 0
            for l in range(K, 0, -1):
                if x/(2*l-1) > (E[P]+B)/(2*L-1):
                    return l

        if allocation_method == dhondt_allocation:
            phi = phi_dhondt
        elif allocation_method == sainte_lague_allocation:
            phi = phi_sainte_lague

        ret = {p : dict() for p in E.parties_w_o(P)}

        for p in E.parties_w_o(P):
            max_rm = min(B, E[p])
            min_seats = phi(E[p] - max_rm) # if we do max bribery against p
            current_seats = phi(E[p]) # without bribery
            # Now the binary search:
            lower = 0
            upper = max_rm
            ret[p][current_seats] = 0
            while lower < upper and current_seats != min_seats:
                current = lower + (upper-lower)//2 + 1
                if phi(E[p]-current+1) == current_seats and phi(E[p]-current) < current_seats:
                    ret[p][phi(E[p]-current)] = current
                    current_seats = phi(E[p]-current)
                    lower = current
                    upper = max_rm
                    continue
                elif phi(E[p]-current) < current_seats:
                    upper = current - 1
                elif phi(E[p]-current) == current_seats:
                    lower = current
        return ret
    #------------------------------------------------------------

    B = min(E.num_votes() - E[P], B)


    if E[P] + B < T:
        return [False] # P won't reach the threshold
    if allocation_method(E, T, K, prefer = P)[P] >= L:
            return [{p:0 for p in E.parties()},-1] # Nothing to do.

    S = K - L # The number of seats the other parties may get at
              # most before P gets L

    gamma = get_gamma_cached() # gamma[p][s] is the min budget
            # needed s.t. p receives exactly s seats before P gets L

    #print(gamma["Likud"])
    table = [[float('inf') for s in range(S+1)] for i in range(len(E.parties()))]
    table[0][0] = 0
    order = tuple(E.parties_w_o(P))

    for i in range(1, len(order)+1):
        curr_party = order[i-1]
        for s in range(S+1):
            # Determine minimum budget needed s.t. parties 1..i receive 
            # at most s seats in total before P receivces its L'th seat.
            for (seats_for_curr, cost) in gamma[curr_party].items():
                if s - seats_for_curr >= 0:
                    tmp = table[i-1][s-seats_for_curr] + cost
                    if tmp < table[i][s]:
                        table[i][s] = tmp

    # Now check if Yes Instance
    i = len(E.parties_w_o(P))
    for s in range(S + 1):
        if table[i][s] <= B:
            # Yes Instance -> Compute vector
            vector = {P:-B}
            left_over = B - table[i][s] # These additional voters have 
                        # to be removed from some parties to sum up to B
            while i > 0:
                curr_party = order[i-1]
                for (seats_for_curr, cost) in gamma[curr_party].items():
                    if s-seats_for_curr >= 0 and table[i][s] - cost == table[i-1][s-seats_for_curr]:
                        vector[curr_party] = cost
                        i -= 1
                        s -= seats_for_curr
                        if left_over > 0:
                            additional_removal = min(E[curr_party] - cost, left_over)
                            vector[curr_party] += additional_removal
                            left_over -= additional_removal
                        break
            assert sum(vector.values()) == 0 # No voters are added or removed
            assert allocation_method(E.apply_bribery_control(vector), T, K, prefer=P)[P] >= L
            return [vector,allocation_method(E.apply_bribery_control(vector), T, K, prefer=P)[P]]
    assert approx_constructive_bribery(E, T, K, P, B, allocation_method) < L # Verify result
    return [False]




def destructive_bribery(E, T, K, P, L, B, allocation_method = dhondt_allocation):

    def phi_dhondt(x):
        assert x >= 0
        if x < T:
            return 0
        if x <= (E[P]-B)/(L+1):
            return 0
        for l in range(K, 0, -1):
            if x/l > (E[P]-B)/(L+1):
                return l
    def phi_sainte_lague(x):
        assert x >= 0
        if x < T:
            return 0
        if x <= (E[P]-B)/(2*(L+1)-1):
            return 0
        for l in range(K, 0, -1):
            if x/(2*l-1) > (E[P]-B)/(2*(L+1)-1):
                return l

    if allocation_method == dhondt_allocation:
        phi = phi_dhondt
    elif allocation_method == sainte_lague_allocation:
        phi = phi_sainte_lague

    #------------------------------------------------------------
    def get_gamma_cached():
        ret = {p : dict() for p in E.parties_w_o(P)}
        for p in E.parties_w_o(P):
            max_seats = phi(E[p] + B) # seats if we do max bribery for p
            lower = 0
            upper = B
            current_seats = phi(E[p]) # seats without bribery
            ret[p][current_seats] = 0
            while lower < upper and current_seats != max_seats:
                current = lower + (upper-lower)//2 + 1
                if phi(E[p]+current-1) == current_seats and phi(E[p]+current) > current_seats:
                    ret[p][phi(E[p]+current)] = current
                    current_seats = phi(E[p]+current)
                    lower = current
                    upper = B
                    continue
                elif phi(E[p]+current) > current_seats:
                    upper = current - 1
                elif phi(E[p]+current) == current_seats:
                    lower = current
        return ret
    #------------------------------------------------------------

    B = min(E[P], B)

    if E[P] - B < T:
        return [{P: B},-1]
    if allocation_method(E, T, K, prefer = P)[P] <= L:
            return [{p:0 for p in E.parties()},-1] # Nothing to do.

    S = K - L # The number of seats the other parties must get at least before P gets L
    gamma = get_gamma_cached() # gamma[p][s] is the min budget needed s.t. p receives 
                               # exactly s seats before P gets L+1


    table = [[float('inf') for s in range(S+1)] for i in range(len(E.parties()))]
    table[0][0] = 0
    order = tuple(E.parties_w_o(P))
    for i in range(1, len(order)+1):
        curr_party = order[i-1]
        for s in range(S+1):
            # Determine minimum budget needed s.t. parties 1..i receive at least
            # s seats in total before P receivces its L+1'th seat.
            for (seats_for_curr, cost) in gamma[curr_party].items():
                if s - seats_for_curr >= 0:
                    tmp = table[i-1][s-seats_for_curr] + cost
                    if tmp < table[i][s]:
                        table[i][s] = tmp

        # Check if we can immediately solve the problem with parties 1..i, i.e., if these
        # parties are sufficient to prevent P from receiving the L+1'th seat
        for s in range(S+1):
            if table[i-1][s] <= B:
                if s + phi(E[curr_party] + (B - table[i-1][s])) >= S:
                    #Yes Instance
                    vector = {P:B, curr_party:-(B - table[i-1][s])}
                    j = i-1
                    l = s
                    while j > 0:
                        curr_party = order[j-1]
                        for (seats_for_curr, cost) in gamma[curr_party].items():
                            if l-seats_for_curr >= 0 and table[j][l] - cost == table[j-1][l-seats_for_curr]:
                                vector[curr_party] = - gamma[curr_party][seats_for_curr]
                                j -= 1
                                l -= seats_for_curr
                                break
                    assert sum(vector.values()) == 0 # No voters are added or removed
                    assert allocation_method(E.apply_bribery_control(vector), T, K, prefer=P)[P] <= L
                    return [vector,allocation_method(E.apply_bribery_control(vector), T, K, prefer=P)[P]]
    assert approx_destructive_bribery(E, T, K, P, B, allocation_method) > L # Verify result
    return [False]






#-----------------------------------------------------------------------------------
#                 Approximation Campaigns
#-----------------------------------------------------------------------------------
#
# These are used as a layer of bug detection. If approximation is ever better than
# the optimal campaign, an error is raised.

def approx_constructive_bribery(E, T, K, P, B, allocation_method):
    B = min(E.num_votes() - E[P], B)
    original_E = E.deepcopy()
    if E[P] + B < T:
        return dict()
    vector = {P : -B}
    E = E.remove(P)
    E = E.apply_threshold(max(T,1))
    while B > 0:
        worst = E.get_worst_party()
        vector[worst] = min(E[worst], B)
        E.remove(worst)
        B -= vector[worst]
    return allocation_method(original_E.apply_bribery_control(vector), T, K, prefer=P)[P]

def approx_destructive_bribery(E, T, K, P, B, allocation_method):
    B = min(E.num_votes() - E[P], B)
    original_E = E.deepcopy()
    vector = {P : B}
    E = E.remove(P)
    parties_below = [p for p in E.parties() if E[p] < T]
    while B > 0 and len(parties_below) > 0:
        best = max(parties_below, key=lambda p: E[p])
        vector[best] = - min(T-E[best], B)
        parties_below.remove(best)
        B += vector[best]
    if B > 0:
        vector[E.get_best_party()] = -B
    return allocation_method(original_E.apply_bribery_control(vector), T, K, prefer=P)[P]





def divisor_app(r, D, k, T):
    """P -- vote distribution for parties (list)
       D -- at least k divisors for the method used (list)
       k -- number of seats to be allocated
       returns: a list of seats (tie-breaking for parties with smaller number)
    """

#    print("Divisors:", list(D))
    r_copy = r.copy()
    m = len(r)
    A = [0]*m

    for i in range(0,m):
        if r_copy[i] < T:
            r_copy[i] = 0

    for _ in range(k):
        # find currently best party, respecting tie-breaking
        #   would be faster with a heap, but with our numbers of parties
        #   that's irrelevant, and tie-breaking is easier here
        max_value = -1
        party = None
        for j in range(m):
            p = float(r_copy[j])/D[A[j]]
            if p > max_value:
                max_value = p
                party = j
        A[party] += 1

    return A


def dhondt_app(p, k, T):
    return divisor_app(p, range(1, k+1), k, T)
