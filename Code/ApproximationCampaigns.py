from core.Election import Election


#-----------------------------------------------------------------------------------
#                 Approximation
#-----------------------------------------------------------------------------------
"""
Explaination of parameters:
E = Election
P = Targeted party (for campaign)
t = threshold
B = Budget (max voters to move/add/remove)
"""


def bribery_weakest_rival_above_threshold(E:Election, t:int, P, B:int)-> dict():
    E = E.thresholdconform(t).remove(P)
    if E.num_parties() == 0:
        raise ValueError("There is no rival above threhsold")
    weakestRivalAboveThreshold = E.get_worst_party()
    usable_budget = min((E[weakestRivalAboveThreshold], B))
    return {P: -usable_budget, weakestRivalAboveThreshold: usable_budget}



def bribery_strongest_rival_above_threshold(E:Election, t:int, P, B:int)-> dict():
    E = E.thresholdconform(t).remove(P)
    if E.num_parties() == 0:
        raise ValueError("There is no rival above threhsold")
    strongestRivalAboveThreshold = E.get_best_party()
    usable_budget = min((E[strongestRivalAboveThreshold], B))
    return {P: -usable_budget, strongestRivalAboveThreshold: usable_budget}

def destructive_bribery_strongest_rival_above_threshold(E:Election, t:int, P, B:int)-> dict():
    tmp = E.thresholdconform(t).remove(P)
    if tmp.num_parties() == 0:
        raise ValueError("There is no rival above threhsold")
    strongestRivalAboveThreshold = tmp.get_best_party()
    usable_budget = min((E[P], B))
    return {P: usable_budget, strongestRivalAboveThreshold: -usable_budget}

def destructive_bribery_strongest_rival_below_threshold(E:Election, t:int, P, B:int)-> dict():
    tmp = E.notthresholdconform(t).remove(P)
    if tmp.num_parties() == 0:
        raise ValueError("There is no rival below threhsold")
    strongestRivalBelowThreshold = tmp.get_best_party()
    usable_budget = min((E[P], B))
    return {P: usable_budget, strongestRivalBelowThreshold: -usable_budget}

def destructive_control_add_strongest_rival_above_threshold(E:Election, t:int, P, B:int)-> dict():
    E = E.thresholdconform(t).remove(P)
    if E.num_parties() == 0:
        raise ValueError("There is no rival above threhsold")
    strongestRivalAboveThreshold = E.get_best_party()
    return {strongestRivalAboveThreshold: -B}

def destructive_control_add_strongest_rival_below_threshold(E:Election, t:int, P, B:int)-> dict():
    E = E.notthresholdconform(t).remove(P)
    if E.num_parties() == 0:
        raise ValueError("There is no rival below threhsold")
    strongestRivalBelowThreshold = E.get_best_party()
    return {strongestRivalBelowThreshold: -B}

def control_del_weakest_rival_above_threshold(E:Election, t:int, P, B:int)-> dict():
    E = E.thresholdconform(t).remove(P)
    if E.num_parties() == 0:
        raise ValueError("There is no rival above threhsold")
    weakestRivalAboveThreshold = E.get_worst_party()
    usable_budget = min((E[weakestRivalAboveThreshold], B))
    return {weakestRivalAboveThreshold: usable_budget}

def control_del_strongest_rival_above_threshold(E:Election, t:int, P, B:int)-> dict():
    E = E.thresholdconform(t).remove(P)
    if E.num_parties() == 0:
        raise ValueError("There is no rival above threhsold")
    strongestRivalAboveThreshold = E.get_best_party()
    usable_budget = min((E[strongestRivalAboveThreshold], B))
    return {strongestRivalAboveThreshold: usable_budget}

def cloning_weakest_rival_above_threshold(E:Election, t:int, P, B:int)-> dict():
    E = E.thresholdconform(t).remove(P)
    if E.num_parties() == 0:
        raise ValueError("There is no rival above threhsold")
    weakestRivalAboveThreshold = E.get_worst_party()
    return {weakestRivalAboveThreshold: B}

def cloning_strongest_rival_above_threshold(E:Election, t:int, P, B:int)-> dict():
    E = E.thresholdconform(t).remove(P)
    if E.num_parties() == 0:
        raise ValueError("There is no rival above threhsold")
    strongestRivalAboveThreshold = E.get_best_party()
    return {strongestRivalAboveThreshold: B}

def destructive_cloning_approx(E:Election, t:int, P, B:int)-> dict():
    return {P: B}
