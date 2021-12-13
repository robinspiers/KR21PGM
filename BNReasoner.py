import random
from typing import Union
from BayesNet import BayesNet
import networkx as nx


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

    def prune_network(self, X: list, Y: list):
        """
        This function prunes the network w.r.t. X given Y
        :param X: Set of variables we are interested in
        :param Y: Set of variables that are observed
        :return: pruned network G_pruned
        """
        G_pruned = BayesNet(self.bn.structure.copy())
        gp_nodes = self.bn.get_all_variables()

        # iteratively remove leaves not in X or Y
        while True:
            vars = gp_nodes.copy()
            remove_vars = [node for node in vars if not G_pruned.get_children(node) and node not in X and node not in Y]
            G_pruned.structure.remove_nodes_from(remove_vars)
            for x in remove_vars:
                del gp_nodes[gp_nodes.index(x)]
            if gp_nodes == vars:
                break

        # Remove all edges from Y
        for node in Y:
            children = G_pruned.structure.get_children(node)
            if children:
                for child in children:
                    G_pruned.structure.remove_edge(node, child)

        return G_pruned, gp_nodes

    def d_separated(self, X: list, Y: list, Z: list, prune=True):
        """
        Is X independent of Y, given Z.
        Following algorithm from p.75 of 'Probabilistic Graphical Models: Principles and Techniques'
        :parameters: lists of variables in graph G
        :output: Boolean (d-separated or not)
        """
        if prune:
            G, gp_vars = self.prune_network(X,Z)
        else:
            G = BayesNet(self.bn.structure.copy)

        # phase I: get all ancestors of Z
        Z_ancestors = G.get_all_ancestors(Z)

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
                if selected_node[0] not in Z:
                    reachable_nodes.append(selected_node[0])
                    if selected_node[0] in Y:
                        return False
                visited_nodes.append(selected_node)

                if selected_node[1] == "up" and selected_node[0] not in Z:
                    nodes_to_visit = nodes_to_visit + [(parent, "up") for parent in G.get_parents(selected_node[0])]
                    nodes_to_visit = nodes_to_visit + [(child, "down") for child in G.get_children(selected_node[0])]

                if selected_node[1] == "down":
                    if selected_node[0] not in Z:
                        nodes_to_visit = nodes_to_visit + [(child, "down") for child in G.get_children(selected_node[0])]
                    if selected_node[0] in Z_ancestors:
                        nodes_to_visit = nodes_to_visit + [(parent, "up") for parent in G.get_parents(selected_node[0])]
        return True


    def mindeg_order(self, X: list):
        """
        This function orders variables for elimination with the minimum degree heuristic
        :param X: list of variables from self.structure.nodes
        :return: ordered list of variables
        """
        # initialize
        G = self.bn.get_interaction_graph()
        pi = []; degrees = []

        while len(X) > 0:
            degrees = [G.degree[var] for var in X]
            chosen_node = X.pop(degrees.index(min(degrees)))
            pi.append(chosen_node)

            # connect all non-adjacent neighbors of chosen node
            nbors = [neighbor for neighbor in G.neighbors()]
            for u in nbors:
                for v in [n for n in nbors if n is not u]:
                    if (u, v) not in G.edges() and (v, u) not in G.edges():
                        G.add_edge(u, v)
            G.remove_node(chosen_node)

        return pi

    def minfill_order(self, X: list):
        """
        This function orders variables for elimination with the minimum fill heuristic
        :param X: list of variables from self.structure.nodes
        :return: ordered list of variables
        """
        # initialize
        G = self.bn.get_interaction_graph()
        pi = []

        while len(X) > 0:
            # find node that produces lowest number of new edges when eliminated
            n_fills = []
            for node in X:
                n_fill = 0
                nbors = [neighbor for neighbor in G.neighbors(node)]
                for u in nbors:
                    for v in [n for n in nbors if n is not u]:
                        if (u, v) not in G.edges() and (v, u) not in G.edges():
                            n_fill += 1
                n_fills.append(n_fill)

            chosen_node = X.pop(n_fills.index(min(n_fills)))
            pi.append(chosen_node)

            # connect all non-adjacent neighbors of chosen node
            nbors = [neighbor for neighbor in G.neighbors(chosen_node)]
            for u in nbors:
                for v in [n for n in nbors if n is not u]:
                    if (u, v) not in G.edges() and (v, u) not in G.edges():
                        G.add_edge(u, v)
            G.remove_node(chosen_node)

        return pi

    def compute_marginal(self, query, pi, evidence=None):
        if evidence:
            marg = 'posterior'
        else:
            marg = 'prior'

        cpts = self.bn.get_all_cpts()

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

        return f
