

"""
  The Election class works similar to a dictionary and stores THE PRIMARY VOTES
  for each party (i.e., NOT complete votes as linear orders).

  NOTE: Elections are semi-immutable in the sense that, except for the E[X] = YY
  syntax, all operations that alter the election object will result in returning
  a copy of this election object with the requested change.
"""
class Election:
    def num_votes(self):
        return sum(self.votealloc.values())

    def num_parties(self):
        return len(self.votealloc)

    def parties(self):
        return sorted(self.votealloc.keys())

    def get_x_best_party(self, x=1):
        parties = list(self.votealloc.keys())
        parties = sorted(parties, key = lambda x: self.votealloc[x])
        return parties[-x]
    
    def get_best_party(self):
        return max(self.votealloc.keys(), key = lambda key: self.votealloc[key])
    
    def get_worst_party(self):
        return min(self.votealloc.keys(), key = lambda key: self.votealloc[key])

    def parties_w_o(self, exclude):
        return sorted([p for p in self.votealloc.keys() if p != exclude])

    def parties_below_threshold(self, T):
        return [p for p in self.votealloc.keys() if self.votealloc[p] < T]

    def is_empty(self):
        return self.num_votes() == 0
    
    def remove(self, party):
        ret = self.deepcopy()
        if party in ret.votealloc:
            ret.votealloc.pop(party)
        return ret

    def deepcopy(self):
        ret = Election()
        for p in self.votealloc:
            ret[p] = self.votealloc[p]
        return ret

    def apply_threshold(self, threshold):
        assert threshold >= 0
        ret = Election()
        for p,v in self.votealloc.items():
            if v >= threshold:
                ret[p] = v
        return ret

    def apply_bribery_control(self, vector:dict):
        ret = self.deepcopy()
        for party, bribery in vector.items():
            ret[party] -= bribery
            assert ret[party] >= 0
        return ret

    # -- Special Methods --

    def __init__(self):
        self.votealloc = dict()

    def __getitem__(self, party):
        if party in self.votealloc:
            return self.votealloc[party]
        return None

    def __setitem__(self, party, votes):
        self.votealloc[party] = votes

    def __str__(self):
        ret = "Vote Count: %d \n" % self.num_votes()
        ret += ("Party ["+', '.join(['%8.8s']*len(self.votealloc))+"]\n") % tuple(
            sorted(self.votealloc.keys(), key = self.votealloc.get, reverse=True))
        ret += ("Votes ["+', '.join(['%8d']*len(self.votealloc))+"]\n") % tuple(
            sorted(self.votealloc.values(), reverse=True))
        ret += ("Percent ["+', '.join(['%.2f  ']*len(self.votealloc))+"]") % tuple(
            [x/self.num_votes() * 100 for x in sorted(self.votealloc.values(), reverse=True)])
        return ret


def election_from_file(path) -> Election:
    ret = Election()
    file = open(path, 'r')
    while True:
        line = file.readline()
        if not line:
            break
        if line.startswith("#"):
        	continue
        ret[str(line.split()[0])] = int(line.split()[1])
    file.close()
    return ret


# election in a district for multi-district elections
def election_from_file_multi(parties,votes) -> Election:
    ret = Election()
    for i in range(0,len(parties)):
        ret[str(parties[i])] = int(votes[i])
    return ret