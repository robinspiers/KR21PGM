from BayesNet import *
from BNReasoner import *
from utils import *


net = './testing/lecture_example.BIFXML'

# constructs a BN object
bn = BayesNet()
# Loads the BN from an BIFXML file
bn.load_from_bifxml(net)
# create reasoner
bnr = BNReasoner(bn)


evidence = {'Winter?': True}

x=['Wet Grass?','Rain?']
y=['Slippery Road?']


"""
USAGE
"""

""" checks whether x and y are d-separated, given (possibly empty) evidence 
    x and y must be lists, evidence is (always) a dict, output is boolean"""
dsep = bnr.d_separated(x,y,evidence)
print('Are X and Y d-separated by evidence: ', dsep)

""" ordering functions for determining elimination order
    inputs should be network structure, y = list of query variables, and optional z = priority variables 
    which are ordered first   """
pi1 = mindeg_order(net=bn, query=y, priority=x)
pi2 = minfill_order(net=bn, query=y, priority=x)
print('\n\nVariable orderings:\n', pi1)
print('\n', pi2)

"""prune network (the network which is used to instantiate the BN Reasoner) wrt query and evidence"""
pruned_net = bnr.prune_network(query=x, evidence=evidence)

"""Marginal distributions, with optional ordering function (name of function is input)"""
# without evidence:
prior_marginal = bnr.compute_marginal(query=y, ordering_function=mindeg_order)
print(f'\n\nPrior Marginal: Query={y}\n', prior_marginal)
# without evidence:
posterior_marginal = bnr.compute_marginal(query=y, evidence=evidence, ordering_function=minfill_order)
print(f'\n\nPosterior Marginal: Query={y}  ---  Evidence={evidence}\n', posterior_marginal)

""" Most likely instantiations: """
# MAP:
MAP = bnr.MAP(query=x, evidence=evidence, ordering_function=mindeg_order)
print('\n\nMAP:\n', MAP)
# MPE:
MPE = bnr.MPE(evidence=evidence, ordering_function=minfill_order)
print('\n\nMPE:\n', MPE)
