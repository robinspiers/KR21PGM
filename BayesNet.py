from typing import List, Tuple, Dict
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.readwrite import XMLBIFReader
import math
import itertools
import pandas as pd
from copy import deepcopy


class BayesNet:

    def __init__(self, net=None) -> None:
        # initialize graph structure
        if net is None:
            self.structure = nx.DiGraph()
        else:
            self.structure = net

    # LOADING FUNCTIONS ------------------------------------------------------------------------------------------------
    def create_bn(self, variables: List[str], edges: List[Tuple[str, str]], cpts: Dict[str, pd.DataFrame]) -> None:
        """
        Creates the BN according to the python objects passed in.
        
        :param variables: List of names of the variables.
        :param edges: List of the directed edges.
        :param cpts: Dictionary of conditional probability tables.
        """
        # add nodes
        [self.add_var(v, cpt=cpts[v]) for v in variables]

        # add edges
        [self.add_edge(e) for e in edges]

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            raise Exception('The provided graph is not acyclic.')

    def load_from_bifxml(self, file_path: str) -> None:
        """
        Load a BayesNet from a file in BIFXML file format. See description of BIFXML here:
        http://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/

        :param file_path: Path to the BIFXML file.
        """
        # Read and parse the bifxml file
        with open(file_path) as f:
            bn_file = f.read()
        bif_reader = XMLBIFReader(string=bn_file)

        # load cpts
        cpts = {}
        # iterating through vars
        for key, values in bif_reader.get_values().items():
            values = values.transpose().flatten()
            n_vars = int(math.log2(len(values)))
            worlds = [list(i) for i in itertools.product([False, True], repeat=n_vars)]
            # create empty array
            cpt = []
            # iterating through worlds within a variable
            for i in range(len(values)):
                # add the probability to each possible world
                worlds[i].append(values[i])
                cpt.append(worlds[i])

            # determine column names
            columns = bif_reader.get_parents()[key]
            columns.reverse()
            columns.append(key)
            columns.append('p')
            cpts[key] = pd.DataFrame(cpt, columns=columns)
        
        # load vars
        variables = bif_reader.get_variables()
        
        # load edges
        edges = bif_reader.get_edges()

        self.create_bn(variables, edges, cpts)

    # METHODS THAT MIGHT ME USEFUL -------------------------------------------------------------------------------------

    def get_children(self, variable: str) -> List[str]:
        """
        Returns the children of the variable in the graph.
        :param variable: Variable to get the children from
        :return: List of children
        """
        return [p for p in self.structure.successors(variable)]

    def get_all_descendants(self, variable: list):
        """
        Returns all the nodes from self.structure which are descendants of variable
        :param variable: ancestor(s) of output variables; Must be a list of any length
        :return: list of all nodes with variable(s) as ancestor
        """
        descendants = []; nodes_to_visit = variable

        while len(nodes_to_visit) > 0.0:
            node = nodes_to_visit.pop()
            for child in self.get_children(node):
                if child not in descendants:
                    nodes_to_visit.append(child)
                    descendants.append(child)
        return descendants

    def get_parents(self, variable: str) -> List[str]:
        """
        Returns the children of the variable in the graph.
        :param variable: Variable to get the children from
        :return: List of children
        """
        return [c for c in self.structure.predecessors(variable)]

    def get_all_ancestors(self, variable: list):
        """
        Returns all the nodes from self.structure which are ancestors of variable
        :param variable: descendant(s) of output variables; Must be a list of any length
        :return: list of all nodes with variable as descendant
        """
        ancestors = []; nodes_to_visit = variable

        while len(nodes_to_visit) > 0.0:
            node = nodes_to_visit.pop()
            for parent in self.get_parents(node):
                if parent not in ancestors:
                    nodes_to_visit.append(parent)
                    ancestors.append(parent)
        return ancestors

    def get_cpt(self, variable: str) -> pd.DataFrame:
        """
        Returns the conditional probability table of a variable in the BN.
        :param variable: Variable of which the CPT should be returned.
        :return: Conditional probability table of 'variable' as a pandas DataFrame.
        """
        try:
            return self.structure.nodes[variable]['cpt']
        except KeyError:
            raise Exception('Variable not in the BN')

    @staticmethod
    def mindeg_order(net, X: list, priority=None):
        """
        This function orders variables for elimination with the minimum degree heuristic
        :param X: list of variables from self.structure.nodes
        :return: ordered list of variables
        """
        # initialize
        G = net.get_interaction_graph()
        pi = []; degrees = []

        while len(X) > 0:
            if priority:
                degrees = [G.degree[var] if var in priority else (len(X)+1) for var in X]
            else:
                degrees = [G.degree[var] for var in X]
            chosen_node = X.pop(degrees.index(min(degrees)))
            pi.append(chosen_node)

            # connect all non-adjacent neighbors of chosen node
            nbors = [neighbor for neighbor in G.neighbors(chosen_node)]
            for u in nbors:
                for v in [n for n in nbors if n is not u]:
                    if (u, v) not in G.edges() and (v, u) not in G.edges():
                        G.add_edge(u, v)
            G.remove_node(chosen_node)

        return pi

    @staticmethod
    def minfill_order(net, X: list, priority: list = None):
        """
        This function orders variables for elimination with the minimum fill heuristic
        :param priority: if there is a list of prioritized variables (e.g. for MAP/MPE), then use this
        :param X: list of variables from self.structure.nodes
        :return: ordered list of variables
        """
        # initialize
        G = net.get_interaction_graph()
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
                if priority:
                    if node not in priority: n_fill = len(nbors)+1
                n_fills.append(n_fill)

            chosen_node = X.pop(n_fills.index(min(n_fills)))
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

    def maxxing(self, cpt: pd.DataFrame, Z_names: list):
        # create maxf(X), Y
        factor_x = cpt['p']
        Y = cpt.drop([*Z_names, 'p'], axis=1)

        marg = {}
        for index, row in Y.iterrows():
            if tuple(row.values) not in marg:
                marg[tuple(row.values)] = [factor_x[index]]
            else:
                marg[tuple(row.values)].append(factor_x[index])

        max_cpt = pd.DataFrame(columns=[*list(Y.columns),'p'])
        for key in marg:
            new_row = {}
            for i, var in enumerate([y for y in Y.columns]):
                new_row[var] = key[var]
            new_row['p'] = max(marg[key])
            nr = [[a for a in new_row.values()]]
            new_row = pd.DataFrame(nr, columns=list(new_row.keys()))
            marg_cpt = pd.concat([max_cpt, new_row])

        return max_cpt

    def marginalize(self, cpt: pd.DataFrame, Z_names: list):
        # create f(X), Y
        factor_x = cpt['p']
        Y = cpt.drop([*Z_names, 'p'], axis=1)

        marg = {}
        for index, row in Y.iterrows():
            if tuple(row.values) not in marg:
                marg[tuple(row.values)] = factor_x[index]
            else:
                marg[tuple(row.values)] += factor_x[index]

        marg_cpt = pd.DataFrame(columns=[*list(Y.columns),'p'])
        for key in marg:
            new_row = {}
            for i, var in enumerate([y for y in Y.columns]):
                new_row[var] = key[var]
            new_row['p'] = marg[key]; nr = [[a for a in new_row.values()]]
            new_row = pd.DataFrame(nr, columns=list(new_row.keys()))
            marg_cpt = pd.concat([marg_cpt, new_row])

        return marg_cpt

    def factor_product(self, cpts: list):
        all_names = list(set.union(*[set(names.columns) for names in cpts]).difference({'p'}))
        ttable = list(itertools.product([False, True], repeat=len(all_names)))
        new_cpt = pd.DataFrame(data=ttable, columns=all_names)

        d = new_cpt.copy()
        for cpt in cpts:
            vars = [var for var in cpt.columns if var is not 'p']
            d = pd.merge(d, cpt, on=vars, how='inner')
        columns = [var for var in d.columns if var not in all_names]
        new_cpt['p'] = d[columns].product(axis=1)

        return new_cpt

    def normalize_factors(self, cpts: dict, evidence: dict):
        normalized_cpts = cpts.copy()
        for key in cpts:
            if any(item in evidence.keys() for item in cpts[key].columns):
                for idx, row in normalized_cpts[key].iterrows():
                    overlapping_vars = list(set(row.index).intersection(set(evidence.keys())))
                    for ev in overlapping_vars:
                        a = cpts[key][ev][idx]; b = evidence[ev]
                        if a != b:
                            normalized_cpts[key]['p'][idx] = 0.0
                            break
        return normalized_cpts

    def get_all_variables(self) -> List[str]:
        """
        Returns a list of all variables in the structure.
        :return: list of all variables.
        """
        return [n for n in self.structure.nodes]

    def get_all_cpts(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary of all CPTs in the network indexed by the variable they belong to.
        :return: Dictionary of all CPTs
        """
        cpts = {}
        for var in self.get_all_variables():
            cpts[var] = self.get_cpt(var)

        return cpts

    def get_interaction_graph(self):
        """
        Returns a networkx.Graph as interaction graph of the current BN.
        :return: The interaction graph based on the factors of the current BN.
        """
        # Create the graph and add all variables
        int_graph = nx.Graph()
        [int_graph.add_node(var) for var in self.get_all_variables()]

        # connect all variables with an edge which are mentioned in a CPT together
        for var in self.get_all_variables():
            involved_vars = list(self.get_cpt(var).columns)[:-1]
            for i in range(len(involved_vars)-1):
                for j in range(i+1, len(involved_vars)):
                    if not int_graph.has_edge(involved_vars[i], involved_vars[j]):
                        int_graph.add_edge(involved_vars[i], involved_vars[j])
        return int_graph

    @staticmethod
    def get_compatible_instantiations_table(instantiation: pd.Series, cpt: pd.DataFrame):
        """
        Get all the entries of a CPT which are compatible with the instantiation.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param cpt: cpt to be filtered
        :return: table with compatible instantiations and their probability value
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        compat_indices = cpt[var_names] == instantiation[var_names].values
        compat_indices = [all(x[1]) for x in compat_indices.iterrows()]
        compat_instances = cpt.loc[compat_indices]
        return compat_instances

    def update_cpt(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Replace the conditional probability table of a variable.
        :param variable: Variable to be modified
        :param cpt: new CPT
        """
        self.structure.nodes[variable]["cpt"] = cpt

    @staticmethod
    def reduce_factor(instantiation: pd.Series, cpt: pd.DataFrame) -> pd.DataFrame:
        """
        Creates and returns a new factor in which all probabilities which are incompatible with the instantiation
        passed to the method to 0.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param cpt: cpt to be reduced
        :return: cpt with their original probability value and zero probability for incompatible instantiations
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        if len(var_names) > 0:  # only reduce the factor if the evidence appears in it
            new_cpt = deepcopy(cpt)
            incompat_indices = cpt[var_names] != instantiation[var_names].values
            incompat_indices = [any(x[1]) for x in incompat_indices.iterrows()]
            new_cpt.loc[incompat_indices, 'p'] = 0.0
            return new_cpt
        else:
            return cpt

    def draw_structure(self) -> None:
        """
        Visualize structure of the BN.
        """
        nx.draw(self.structure, with_labels=True, node_size=3000)
        plt.show()

    # BASIC HOUSEKEEPING METHODS ---------------------------------------------------------------------------------------

    def add_var(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Add a variable to the BN.
        :param variable: variable to be added.
        :param cpt: conditional probability table of the variable.
        """
        if variable in self.structure.nodes:
            raise Exception('Variable already exists.')
        else:
            self.structure.add_node(variable, cpt=cpt)

    def add_edge(self, edge: Tuple[str, str]) -> None:
        """
        Add a directed edge to the BN.
        :param edge: Tuple of the directed edge to be added (e.g. ('A', 'B')).
        :raises Exception: If added edge introduces a cycle in the structure.
        """
        if edge in self.structure.edges:
            raise Exception('Edge already exists.')
        else:
            self.structure.add_edge(edge[0], edge[1])

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            self.structure.remove_edge(edge[0], edge[1])
            raise ValueError('Edge would make graph cyclic.')

    def del_var(self, variable: str) -> None:
        """
        Delete a variable from the BN.
        :param variable: Variable to be deleted.
        """
        self.structure.remove_node(variable)

    def del_edge(self, edge: Tuple[str, str]) -> None:
        """
        Delete an edge form the structure of the BN.
        :param edge: Edge to be deleted (e.g. ('A', 'B')).
        """
        self.structure.remove_edge(edge[0], edge[1])
