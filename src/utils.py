import igraph as ig
import matplotlib.pyplot as plt
import cvxpy as cp
import itertools

def draw_graph(graph: ig.Graph) -> None:
    """
    Draw the graph using igraph and matplotlib.
    Parameters:
    ----------
    graph : igraph.Graph
        The input graph to be drawn.
    """
    fig, ax = plt.subplots()
    ig.plot(
        graph, 
        layout="kk", 
        target=ax,
        edge_width=2,
        edge_label=graph.es["capacity"],
        vertex_size=25,
        vertex_label=range(graph.vcount()),
        edge_background="white"
        )
    plt.show()

def clean_disconnected_components(graph: ig.Graph) -> ig.Graph:
    """
    Remove disconnected components from the graph.

    Parameters:
    ----------
    graph : igraph.Graph
        The input graph.

    Returns:
    -------
    igraph.Graph
        The largest connected subgraph of the original graph.
    """
    return graph if graph.is_connected(mode="weak") else graph.connected_components(mode="weak").giant()

def rho_ms(graph: ig.Graph, S: set[int], A: set[int]) -> float:
    """
    Computes the multi-source multicast rate ρ_ms(S, A)
    from Definition 6 of the ISIT paper (LP-based).

    Parameters:
        graph: igraph.Graph with edge attribute 'capacity'
        S: set of source node indices
        A: set of sink node indices

    Returns:
        float: maximum total rate r(S) that can be multicast to A
    """
    V = set(range(graph.vcount()))
    S = set(S)
    A = set(A)
    capacities = graph.es["capacity"] if "capacity" in graph.es.attributes() else [1] * graph.ecount()

    # Variables: r_u ≥ 0 for u ∈ S
    r_vars = {u: cp.Variable(nonneg=True) for u in S}

    constraints = []

    # Generate cut sets B ⊂ V such that B ⊈ A
    for size in range(1, len(V)):
        for B in itertools.combinations(V, size):
            B = set(B)
            if B.issuperset(A):
                continue  # skip cuts that fully contain A

            # Compute λ(B, V\B)
            cut_cap = 0
            for e in graph.es:
                u, v = e.source, e.target
                if u in B and v not in B:
                    cut_cap += e["capacity"]

            # Add constraint: sum_{u in B ∩ S} r_u ≤ λ(B, V\B)
            if cut_cap < 1e-9:
                continue  # skip zero-cut (avoid overconstraining)
            lhs = cp.sum([r_vars[u] for u in B & S])
            constraints.append(lhs <= cut_cap)

    # Objective: maximize total injected rate from S
    total_rate = cp.sum([r_vars[u] for u in S])
    prob = cp.Problem(cp.Maximize(total_rate), constraints)
    prob.solve()

    return prob.value