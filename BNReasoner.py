import random
from typing import Union
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def prune_network(self, query: list = None, evidence: dict = None):
        """
        This function prunes the network w.r.t. X given evidence
        :param X: Set of variables we are interested in
        :param evidence: Set of variables that are observed
        :return: pruned network G_pruned
        """
        G_pruned = BayesNet(self.bn.structure.copy())
        gp_nodes = self.bn.get_all_variables()

        # iteratively remove leaves not in X or evidence
        if query:
            while True:
                vars = gp_nodes.copy()
                remove_vars = [node for node in vars if not G_pruned.get_children(node) and node not in query and node not in evidence]
                G_pruned.structure.remove_nodes_from(remove_vars)
                for x in remove_vars:
                    del gp_nodes[gp_nodes.index(x)]
                if gp_nodes == vars:
                    break

        # Remove all edges going from evidence
        if evidence:
            for node in evidence.keys():
                children = G_pruned.get_children(node)
                if children:
                    for child in children:
                        G_pruned.structure.remove_edge(node, child)

            # update CPTs
            updated_cpts = G_pruned.get_all_cpts()
            for key in updated_cpts:
                if any(item in evidence.keys() for item in updated_cpts[key].columns):
                    for idx, row in updated_cpts[key].iterrows():
                        overlapping_vars = list(set(row.index).intersection(set(evidence.keys())))
                        for ev in overlapping_vars:
                            a = updated_cpts[key][ev][idx]; b = evidence[ev]
                            if a != b:
                                updated_cpts[key].drop(idx)
                                break
                    G_pruned.update_cpt(key, updated_cpts[key])

        return G_pruned

    def d_separated(self, X: list, Y: list, evidence: dict, prune=True):
        """
        Is X independent of Y, given evidence Z.
        Following algorithm from p.75 of 'Probabilistic Graphical Models: Principles and Techniques'
        :parameters: lists of variables in graph G
        :output: Boolean (d-separated or not)
        """
        if prune:
            G = self.prune_network(X,evidence)
        else:
            G = BayesNet(self.bn.structure.copy)

        # phase I: get all ancestors of Z
        Z_ancestors = G.get_all_ancestors(list(evidence.keys()))

        # phase II: traverse active trails starting from X
        # first initialize to_be_visited, visited and reachable nodes
        nodes_to_visit = [(node, "up") for node in X]
        visited_nodes = []
        reachable_nodes = []

        # while there are still nodes to visit, look for reachable nodes...
        while len(nodes_to_visit) > 0:
            # select some node (including its traversal direction) and remove it from list
            selected_node = nodes_to_visit.pop(nodes_to_visit.index(random.choice(nodes_to_visit)))
            if selected_node not in visited_nodes:
                if selected_node[0] not in evidence:
                    reachable_nodes.append(selected_node[0])
                    if selected_node[0] in Y:
                        return False
                visited_nodes.append(selected_node)

                if selected_node[1] == "up" and selected_node[0] not in evidence:
                    nodes_to_visit = nodes_to_visit + [(parent, "up") for parent in G.get_parents(selected_node[0])]
                    nodes_to_visit = nodes_to_visit + [(child, "down") for child in G.get_children(selected_node[0])]

                if selected_node[1] == "down":
                    if selected_node[0] not in evidence:
                        nodes_to_visit = nodes_to_visit + [(child, "down") for child in G.get_children(selected_node[0])]
                    if selected_node[0] in Z_ancestors:
                        nodes_to_visit = nodes_to_visit + [(parent, "up") for parent in G.get_parents(selected_node[0])]
        return True

    def compute_marginal(self, query: list, evidence: dict = None, ordering_function=None):
        cpts = self.bn.get_all_cpts()
        if ordering_function:
            pi = ordering_function(self.bn, [var for var in self.bn.get_all_variables() if var not in query])
        else:
            pi = [var for var in self.bn.get_all_variables() if var not in query]
        # if calculating posterior marginals, then we normalize CPTs wrt the evidence
        if evidence:
            cpts = self.bn.normalize_factors(cpts, evidence)

        for i, var in enumerate(pi):
            fk = {}
            for idx, cpt in enumerate(cpts):
                if var in cpts[cpt].columns:
                    fk[cpt] = cpts[cpt]

            for cpt in fk:
                cpts.pop(cpt)

            f = [fk[cpt] for cpt in fk]
            f = self.bn.factor_product(f)
            f = self.bn.marginalize(f, [var])
            new_key = 'f'+str(i)
            cpts[new_key] = f

        return cpts

    def MPE(self, evidence: dict, ordering_function=None):
        N_pr = self.prune_network(evidence=evidence)   # pruned network
        Q = N_pr.get_all_variables()
        cpts = N_pr.get_all_cpts()

        pi = ordering_function(N_pr, Q)

        cpts = self.bn.normalize_factors(cpts, evidence)

        for i, var in enumerate(pi):
            fk = {}
            for idx, cpt in enumerate(cpts):
                if var in cpts[cpt].columns:
                    fk[cpt] = cpts[cpt]

            f = [cpts.pop(cpt) for cpt in fk]
            f = self.bn.factor_product(f)
            f = self.bn.maxxing(f, [var])

            new_key = 'f('+'-'.join(list(fk.keys()))+')'
            cpts[new_key] = f

        return cpts

    def MAP(self, query: list, evidence: dict, ordering_function=None):
        N_pr = self.prune_network(query=query, evidence=evidence)      # pruned network
        Q = N_pr.get_all_variables()
        cpts = N_pr.get_all_cpts()

        pi = ordering_function(N_pr, Q, priority=[var for var in Q if var not in query])

        cpts = self.bn.normalize_factors(cpts, evidence)

        for i, var in enumerate(pi):
            fk = {}
            for idx, cpt in enumerate(cpts):
                if var in cpts[cpt].columns:
                    fk[cpt] = cpts[cpt]

            f = [cpts.pop(cpt) for cpt in fk]
            f = self.bn.factor_product(f)

            if var in query:
                f = self.bn.maxxing(f, [var])
            else:
                f = self.bn.marginalize(f, [var])
            new_key = 'f('+'-'.join(list(fk.keys()))+')'
            cpts[new_key] = f

        return cpts
