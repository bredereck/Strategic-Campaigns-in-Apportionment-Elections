import pulp as plp
from itertools import product


# votes: vector of votes for each party
# seats: number of seats to allocate in district
# party_no: number of party for which we want to gain seats
#           (numbering starts with 0)
# target: desired number of seats for party with number party_no
# solve: determines if the method should run solver to solve ILP
# filename: pulp can write LP file to disk to be solved with external solver
def solve(votes, seats, party_no, target, divisors,
          solve=True, filename=None):
    #print(seats)
    #print(party_no)
    #print(target)
    #print(divisors)
    #print(votes)
    if target > seats:
        raise Exception
    prob = plp.LpProblem("D'Hondt", plp.LpMinimize)

    # to avoid winning with tie-breaking
    # votes[party_no] -= .001

    # Helpers
    # String values of parties indices; party numbers
    party_indices = [str(i) for i in range(len(votes))]
    # String values of seats numbers; round numbers
    seats_indices = [str(i) for i in range(seats)]
    # Thus we will construct parties x seats matrices, we need their product
    # as list of strings
    perms = [a + "|" + b for a, b in product(party_indices, seats_indices)]
    # Arbitrary big number
    eps = 0.0001
    K = 2 * sum(votes) + eps

    # Variables
    # Number of votes given for or taken from every party
    x = plp.LpVariable.dicts("x", party_indices, None, None, plp.LpInteger)
    # Absolute value of x
    abs_x = plp.LpVariable.dicts("abs_x", party_indices, 0, None, plp.LpInteger)
    # Divisions matrix taking x-es into account
    t = plp.LpVariable.dicts("t", perms, 0, None, plp.LpContinuous)
    # Binary variables for expressing if our result is bigger than other result
    p = plp.LpVariable.dicts("p", perms, 0, 1, plp.LpInteger)
    q = plp.LpVariable.dicts("q", perms, 0, 1, plp.LpInteger)
    # Number of bigger results
    s = plp.LpVariable("s", 0, None, plp.LpInteger)
    # Sum of all moves
    moves = plp.LpVariable("moves", 0, None)


    # Constraints
    # Sum of gives and takes
    prob += plp.lpSum(x) == 0, "Sum of all moves is zero"
    # Whole D'Hondt Matrix
    for z in perms:
        i, j = z.split("|")
        i, j = int(i), int(j)
        prob += (votes[i] * 1./divisors[j] + x[str(i)] * 1./divisors[j] == t[z],
                 "Party {} in round {}".format(i, j))
    # For each element of t determine if it is bigger than ours
    check_idx = "{}|{}".format(party_no, target - 1)
    for z in perms:
        if z == check_idx:
          #we have to ensure that if we compare t[z=check_idx] with
          #t[check_idx], then we have p[z] = 1. It is because
          #p[z] is 0 also if t[z] = t[check_idx]. In such a case
          #the corresponding q[z] would be 0 (by the below constraint
          #"bigger ..."). However, as the name suggest, we want to
          #count bigger t[z]'s not bigger and equal, so p[z] = 0
          #would give incorrent q[z]
          #Furthermore, relying on t[z] - t[z] being 0 we might end
          #up comparing K to K-eps whose result is not reliable
          #for small values of eps in cumputer arithmetic
          p[z].setInitialValue(1.0)
          p[z].fixValue
        else:
          prob += t[z] - t[check_idx] + K * p[z] >= 0, "Is {} bigger than {} p1".format(check_idx, z)
          prob += t[z] - t[check_idx] + K * p[z] <= K - eps, "Is {} bigger than {} p2".format(check_idx, z)
        prob += q[z] == 1 - p[z], "bigger {}".format(z)
    # Count bigger elements
    prob += s == plp.lpSum(q), "Number of bigger elements"
    # Problem solution constraint
    prob += s <= seats - 1
    # Compute pseudo-absolutes
    for i in party_indices:
        prob += abs_x[i] >= x[i]
        prob += abs_x[i] >= -x[i]

    # Optimization subject
    prob += plp.lpSum(abs_x)
    # Sum of all moves
    prob += moves == plp.lpSum(abs_x) / 2.0



    # Solution
    if filename is not None:
        prob.writeLP(filename)
    if solve:
        flag = prob.solve(plp.apis.GUROBI(msg=False, MIPgap=0.0))
        if flag == -1:
          return -1, -1
        #for v in prob.variables():
            #print(v.name, "=", v.varValue)
        mvs = [plp.value(x[i]) for i in party_indices]
        return int(plp.value(prob.objective) / 2), mvs
# output: a tuple = vote transfers to achieve desired target
