def mindeg_order(net, query: list, priority=None):
    """
    This function orders variables for elimination with the minimum degree heuristic
    :param X: list of variables from self.structure.nodes
    :return: ordered list of variables
    """
    # initialize
    G = net.get_interaction_graph()
    pi = []
    degrees = []

    while len(query) > 0:
        if priority:
            degrees = [G.degree[var] if var in priority else (len(query) + 1) for var in query]
        else:
            degrees = [G.degree[var] for var in query]
        chosen_node = query.pop(degrees.index(min(degrees)))
        pi.append(chosen_node)

        # connect all non-adjacent neighbors of chosen node
        nbors = [neighbor for neighbor in G.neighbors(chosen_node)]
        for u in nbors:
            for v in [n for n in nbors if n is not u]:
                if (u, v) not in G.edges() and (v, u) not in G.edges():
                    G.add_edge(u, v)
        G.remove_node(chosen_node)

    return pi


def minfill_order(net, query: list, priority: list=None):
    """
    This function orders variables for elimination with the minimum fill heuristic
    :param priority: if there is a list of prioritized variables (e.g. for MAP/MPE), then use this
    :param X: list of variables from self.structure.nodes
    :return: ordered list of variables
    """
    # initialize
    G = net.get_interaction_graph()
    pi = []

    while len(query) > 0:
        # find node that produces lowest number of new edges when eliminated
        n_fills = []
        for node in query:
            n_fill = 0
            nbors = [neighbor for neighbor in G.neighbors(node)]

            for u in nbors:
                for v in [n for n in nbors if n is not u]:
                    if (u, v) not in G.edges() and (v, u) not in G.edges():
                        n_fill += 1
            if priority:
                if node not in priority: n_fill = len(nbors) + 1
            n_fills.append(n_fill)

        chosen_node = query.pop(n_fills.index(min(n_fills)))
        if priority:
            del priority[priority.index(chosen_node)]
        pi.append(chosen_node)

        # connect all non-adjacent neighbors of chosen node
        nbors = [neighbor for neighbor in G.neighbors(chosen_node)]
        for u in nbors:
            for v in [n for n in nbors if n is not u]:
                if (u, v) not in G.edges() and (v, u) not in G.edges():
                    G.add_edge(u, v)
        G.remove_node(chosen_node)

    return pi
