import igraph as ig
import math
from typing import Set

def rate_ss(graph: ig.Graph, S: Set[int], A: Set[int]) -> float:
    if A == set(): 
        return math.inf
    elif S.issubset(A): 
        return float(0)
    
    new_node = len(graph.vs)
    flows = []
    
    if len(S) == 1:
        s = list(S)[0]
        for j in A:            
            min_cut_value = graph.mincut(s, j, capacity=graph.es["capacity"]).value
            flows.append(min_cut_value)
        return min(flows)
    
    for j in A:
        mapping_vector = [i if i not in S - {j} else new_node for i in range(new_node)]
        contracted_graph = graph.copy()
        contracted_graph.contract_vertices(mapping_vector, combine_attrs={"capacity": sum})
        min_cut_value = contracted_graph.mincut(new_node, j, capacity=contracted_graph.es["capacity"]).value
        flows.append(min_cut_value)
            
    return min(flows)

def rate_uc(graph: ig.Graph, S: Set[int], A: Set[int]) -> float:
    if A == set():
        return math.inf
    if not S.isdisjoint(A):
        raise ValueError("Source set S must be disjoint from sink set A.")
    new_node = len(graph.vs)
    flows = []
    
    if len(S) == 1:
        s = list(S)[0]
        for j in A:            
            min_cut_value = graph.mincut(s, j, capacity=graph.es["capacity"]).value
            flows.append(min_cut_value)
        return min(flows)
    
    for j in A:
        mapping_vector = [i if i not in S - {j} else new_node for i in range(new_node)]
        contracted_graph = graph.copy()
        contracted_graph.contract_vertices(mapping_vector, combine_attrs={"capacity": sum})
        min_cut_value = contracted_graph.mincut(new_node, j, capacity=contracted_graph.es["capacity"]).value
        flows.append(min_cut_value)
            
    return min(flows)

def rate_ma(graph: ig.Graph, S: set[int], A: set[int]) -> float:
    """
    Compute multiple-access multicast rate: 
    rho_ma(S, A) = min_{s in S} min_{a in A} lambda(s, a)

    Parameters
    ----------
    graph : igraph.Graph
        Directed or undirected graph with edge attribute 'capacity'.
    S : set[int]
        Source nodes.
    A : set[int]
        Sink nodes.

    Returns
    -------
    float
        Minimum capacity over all source-sink pairs.
    """
    if not A:
        return math.inf
    if not S:
        return 0.0
    if S.issubset(A):
        return 0.0

    capacities = graph.es["capacity"] if "capacity" in graph.es.attributes() else [1] * graph.ecount()
    mincuts = []

    for s in S:
        for a in A:
            if s == a:
                continue
            val = graph.mincut(s, a, capacity=capacities).value
            mincuts.append(val)

    return min(mincuts) if mincuts else 0.0


def rate_ms(graph: ig.Graph, S: Set[int], A: Set[int]) -> float:
    """
    Compute rho(S, A): the multicast rate from source set S to sink set A.

    Parameters:
    -----------
    S : set of int
        Source node indices.
    A : set of int
        Sink node indices.

    Returns:
    --------
    float
        The minimum multicast rate across all transformed (S, a) pairs.
    """
    if not A:
        return float("inf")
    if S.issubset(A):
        return 0.0

    new_node = graph.vcount()
    flows = []
    capacities = graph.es["capacity"]

    if len(S) == 1:
        s = list(S)[0]
        for j in A:
            min_cut_value = graph.mincut(s, j, capacity=capacities).value
            flows.append(min_cut_value)
        return min(flows)

    for j in A:
        mapping_vector = [i if i not in S - {j} else new_node for i in range(new_node)]
        contracted_graph = graph.copy()
        contracted_graph.contract_vertices(mapping_vector, combine_attrs={"capacity": sum})
        min_cut_value = contracted_graph.mincut(new_node, j, capacity=contracted_graph.es["capacity"]).value
        flows.append(min_cut_value)

    return min(flows)