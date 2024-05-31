# PF's code implementing apportionment methods and getting additional
# seats for the first party

import pulp as plp
from itertools import product


# votes: vector of votes for each party
# seats: number of seats to allocate in district
# party_no: number of party for which we want to gain seats
#           (numbering starts with 0)
# target: desired number of seats for party with number party_no
# solve: determines if the method should run solver to solve ILP
# filename: pulp can write LP file to disk to be solved with external solver
def optbribery_ilp(votes, seats, target, divisors, BIG=None,
                   solve=True, filename=None):
    if target > seats:
        raise Exception
    prob = plp.LpProblem("Apportionment-Bribery-for-Divisor-Methods",
                         plp.LpMinimize)

    # Helpers
    # String values of parties indices; party numbers
    party_indices = [str(i) for i in range(len(votes))]
    # String values of seats numbers; round numbers
    seats_indices = [str(i) for i in range(seats+1)]
    # Thus we will construct parties x seats matrices, we need their product
    # as list of strings
    perms = [a + "|" + b for a, b in product(party_indices, seats_indices)]
    # Large (but not too large) constant
    if BIG is None:
        BIG = sum(votes)

    # Variables
    # Number of votes given for or taken from every party
    x = plp.LpVariable.dicts("x", party_indices, None, None, plp.LpInteger)
    # Divisions matrix taking x-es into account
    t = plp.LpVariable.dicts("t", perms, 0, None, plp.LpContinuous)
    # Binary variables for expressing if our result is bigger than other result
    p = plp.LpVariable.dicts("p", perms, 0, 1, plp.LpInteger)
    # q = plp.LpVariable.dicts("q", perms, 0, 1, plp.LpInteger)
    # Number of bigger results
    s = plp.LpVariable("s", 0, None, plp.LpInteger)

    # Constraints
    # Sum of gives and takes
    prob += plp.lpSum(x) == 0, "Sum of all moves is zero"
    for i in range(1, len(votes)):
        prob += x[str(i)] <= 0
    # Whole D'Hondt Matrix
    for z in perms:
        i, j = z.split("|")
        i, j = int(i), int(j)
        prob += ((votes[i] + x[str(i)]) * 1. / divisors[j] == t[z],
                 "Party {} in round {}".format(i, j))
    # For each element of t determine if it is bigger than ours
    check_idx = "{}|{}".format(0, target - 1)
    for z in perms:
        if z[0] != "0":
            prob += (t[check_idx] - t[z] + BIG * p[z] >= 0,
                     "Is {} bigger than {} p1".format(check_idx, z))
        # prob += (t[z] - t[check_idx] + BIG * p[z] <= BIG - SMALL,
        #          "Is {} bigger than {} p2".format(check_idx, z))
        # prob += q[z] == 1 - p[z]
    # Count bigger elements
    prob += s == plp.lpSum(p), "Number of bigger elements"
    # Problem solution constraint
    prob += s <= seats - target
    # Optimization subject
    prob += x["0"]
    # Solution
    if filename is not None:
        prob.writeLP(filename)
    if solve:
        prob.solve(plp.GUROBI(options=[("FeasibilityTol", 1e-9)],
                                      msg=False))
        mvs = [plp.value(x[i]) for i in party_indices]
        return int(plp.value(prob.objective)), mvs
    else:
        raise Exception("Not solved")
# output: a tuple = vote transfers to achieve desired target


# APPORTIONMENT

def divisor_app(P, D, k):
    """P -- vote distribution for parties (list)
       D -- at least k divisors for the method used (list)
       k -- number of seats to be allocated
       returns: a list of seats (tie-breaking for parties with smaller number)
    """

#    print("Divisors:", list(D))

    m = len(P)
    A = [0]*m

    for _ in range(k):
        # find currently best party, respecting tie-breaking
        #   would be faster with a heap, but with our numbers of parties
        #   that's irrelevant, and tie-breaking is easier here
        max_value = -1
        party = None
        for j in range(m):
            p = float(P[j])/D[A[j]]
            if p > max_value:
                max_value = p
                party = j
        A[party] += 1

    return A


def dhondt_app(P, k):
    return divisor_app(P, range(1, k+1), k)


def sl_app(P, k):
    return divisor_app(P, [2*i+1 for i in range(k)], k)


# GETTING ADDITIONAL SEATS


def seats_fun(p, x, D, k):
    lam = 0

    if float(p)/D[lam] <= x:
        return 0

    while True:
        if (float(p)/D[lam] > x) and (float(p)/D[lam+1] <= x):
            return lam+1
        if(lam+1 == k):
            return k
        lam += 1


class DivisorBribery:
    def __init__(self, P, D, k):
        self.m = len(P)
        self.P = P
        self.D = D
        self.k = k
        self.seats = None

    def compute_F(self, i, s, b):
        # print(i, s, b)
        if s < 0 or b < 0:
            return [False]

        if (i, s, b) in self.F:
            return self.F[(i, s, b)]

        possbribesorder = [0] + list(reversed(range(1, b+1)))
        # [b] + [self.P[i]] + list(range(b))
        for bi in possbribesorder:
            if self.P[i]-bi < 0:  # negative number of votes
                continue
            seats = self.seats(self.P[i]-bi)
            val = self.compute_F(i-1, s - seats, b - bi)
            if val[0]:
                self.F[(i, s, b)] = [True, seats, bi]
                return self.F[(i, s, b)]

        self.F[(i, s, b)] = [False]

        return [False]

    def recover_bribes(self, S, B):
        (_, s, b) = self.F[(self.m-1, S, B)]
        bribes = []
        fullB = B
        for i in range(self.m-1, 0, -1):
            bribes.append(-b)
            B -= b
            S -= s
            (_, s, b) = self.F[(i-1, S, B)]

        bribes.append(fullB)
        bribes.reverse()
        return bribes

    def get_seats(self, ell, B):
        """get ell seats for Party_0, given
           - vote distribution P,
           - divisors D
           - number of seats k
           - budget B"""

        self.F = dict()
        self.seats = lambda p: seats_fun(p, float(self.P[0]+B)/(self.D[ell-1]),
                                         self.D, self.k)

        # initiate the F function
        for s in range(self.k+1):
            for b in range(B+1):
                if (s != 0) or (b != 0):
                    self.F[(0, s, b)] = [False]
                self.F[(0, s, 0)] = [True, -1, -1]

        # for kprim in range(self.k-ell, -1, -1):
        kprim = self.k - ell
        val = self.compute_F(self.m-1, kprim, B)
        if val[0]:
            return [True, self.recover_bribes(kprim, B)]

        return [False, [0 for _ in range(self.m)]]


def fuzzyint(x):
    if x > 0:
        if x - (int(x)) > 0.99:
            return int(x)+1
        elif x - (int(x)) < 0.01:
            return int(x)
        else:
            raise Exception("Output ambiguous ({})".format(x))
    else:
        if (int(x)) - x > 0.99:
            return int(x)-1
        elif (int(x)) - x < 0.01:
            return int(x)
        else:
            raise Exception("Output ambiguous ({})".format(x))


def findoptimalbribery(votes, k, target, divisors, ilp=False):
    print("*"*40)
    print(f"Computing opitmal bribery")
    print(f"votes: {votes}")
    print(f"seats: {k}")
    print(f"target: {target}")
    print(f"divisors: {divisors}")
    if ilp:
        res = optbribery_ilp(votes, k, target, divisors)
        bestbrb = [fuzzyint(x) for x in res[1]]
    else:
        resolutions = [10000, 1000, 100, 10, 1]
        print(f"Staring bin-search for optimal bribery with resolutions: {resolutions}")
        upperbound = None
        lowerbound = 0
        bestbrb = []
        for resindex, resolution in enumerate(resolutions):
            print(f"Current bin-search resolution: {resolution}")
            votesmod = [v // resolution for v in votes]
            for i in range(1, len(votesmod)):
                if votesmod[i] * resolution < votes[i]:
                    votesmod[i] += 1
            if upperbound is None:
                upperbound = sum([v // resolution for v in votes][1:])
            # print(votesmod)
            # print(dhondt_app(votes, k))
            bribery = DivisorBribery(votesmod, divisors, k)
            print(f"Initial bounds: {lowerbound} -- {upperbound}")
            while lowerbound < upperbound:
                budget = (lowerbound + upperbound) // 2
                # print("LU",lowerbound, upperbound)
                # print("target budget",target, budget)
                res, brb = bribery.get_seats(
                    target, budget)  # ell, B
                # print(res, brb)
                if res:
                    upperbound = budget
                    bestbrb = brb
                else:
                    lowerbound = budget+1
            print(f"Final bounds: {lowerbound} -- {upperbound}")
            if resolution != 1 and (resindex + 1 < len(resolutions)):
                lowerbound = 0 #(lowerbound-1) * 10
                upperbound = (upperbound * resolution) // resolutions[resindex + 1]
        bestbrb = [b*resolutions[-1] for b in bestbrb]

    # check if bribery is correct
    app = dhondt_app(votes, k)
    votesmod = [votes[i]+bestbrb[i] for i in range(len(votes))]
    appmod = dhondt_app(votesmod, k)
    if appmod[0] < target:
        print("optimal bribery failed")
        print(votes)
        print(bestbrb)
        print(target, appmod, app)
        print(votesmod)
        raise Exception

    # check if bribery is optimal
    ind = bestbrb.index(min(bestbrb))
    votesmod[0] -= 1
    votesmod[ind] += 1
    appmod = dhondt_app(votesmod, k)
    if appmod[0] >= target:
        print("Warning: bribery is not optimal")
        print(votes)
        print(bestbrb)
        print(target, appmod, app)
        print(votesmod)
        print("-----")

    return bestbrb


def findoptimalabstainers(votes, k, target, divisors):
    upperbound = sum(votes) * (k + 1)
    lowerbound = 0
    while lowerbound < upperbound:
        budget = (lowerbound + upperbound) // 2
        # print(lowerbound, upperbound)
        addvotes = [0]*len(votes)
        addvotes[0] = budget
        newresult = divisor_app([votes[i]+addvotes[i]
                                 for i in range(len(votes))],
                                divisors, k)
        if newresult[0] >= target:
            upperbound = budget
        else:
            lowerbound = budget+1

    # check if abstainer bribery is correct
    app = dhondt_app(votes, k)
    votesmod = [votes[i] for i in range(len(votes))]
    votesmod[0] += upperbound
    appmod = dhondt_app(votesmod, k)
    if appmod[0] < target:
        print("abstainer bribery failed")
        print(votes)
        print(upperbound)
        print(target, appmod, app)
        print(votesmod)
        raise Exception
    votesmod[0] -= 1
    appmod = dhondt_app(votesmod, k)
    if appmod[0] >= target:
        print("abstainer bribery failed (not optimal)")
        print(votes)
        print(upperbound)
        print(target, appmod, app)
        print(votesmod)
        raise Exception

    return upperbound


def findbalancedbribery(votes, k, target, divisors):
    upperbound = max(votes[1:])
    lowerbound = 0
    while lowerbound < upperbound:
        maxred = (lowerbound + upperbound) // 2
        # print(lowerbound, upperbound)
        diffvotes = [int(-maxred * votes[i] / max(votes[1:]))
                     for i in range(len(votes))]
        diffvotes[0] = -sum(diffvotes[1:])
        newvotes = [votes[i] + diffvotes[i] for i in range(len(votes))]
        newresult = divisor_app(newvotes, divisors, k)
        if newresult[0] >= target:
            upperbound = maxred
            balbrb = diffvotes
        else:
            lowerbound = maxred+1

    # check if balanced bribery is correct
    app = dhondt_app(votes, k)
    votesmod = [votes[i]+balbrb[i] for i in range(len(votes))]
    appmod = dhondt_app(votesmod, k)
    if appmod[0] < target:
        print("optimal bribery failed")
        print(votes)
        print(balbrb)
        print(target, appmod, app)
        print(votesmod)
        raise Exception

    return balbrb


# party = 0 is the strongest party,
#       = 1 the 2nd strongest
#       = -1 the weakest party
def findonepartybribery(votes, k, target, divisors, party=0):
    sortedvotes = sorted(votes[1:], reverse=True)
    pindex = votes.index(sortedvotes[party])
    upperbound = votes[pindex]
    lowerbound = 0
    bestred = None
    while lowerbound < upperbound:
        red = (lowerbound + upperbound) // 2
        # print(lowerbound, upperbound)
        newvotes = [v for v in votes]
        newvotes[0] += red
        newvotes[pindex] -= red
        newresult = divisor_app(newvotes, divisors, k)
        if newresult[0] >= target:
            upperbound = red
            bestred = red
        else:
            lowerbound = red+1

    diffvotes = [0] * len(votes)
    if bestred is None:
        # too few voters in strongest party, add abstainers
        votesmod = [v for v in votes]
        votesmod[0] += votes[pindex]
        votesmod[pindex] = 0
        abstainers = findoptimalabstainers(votesmod, k, target, divisors)
        diffvotes[0] = votes[pindex] + abstainers
        diffvotes[pindex] = -votes[pindex]
    else:
        diffvotes[0] += bestred
        diffvotes[pindex] -= bestred

    # check if balanced bribery is correct
    app = dhondt_app(votes, k)
    votesmod = [votes[i]+diffvotes[i] for i in range(len(votes))]
    appmod = dhondt_app(votesmod, k)
    if appmod[0] < target:
        print("oneparty bribery failed")
        print(votes)
        print(diffvotes)
        print(target, appmod, app)
        print(votesmod)
        raise Exception

    return diffvotes
