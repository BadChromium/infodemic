import heapq
import igraph as ig
from typing import Callable, Set, Tuple

def greedy_forward(graph: ig.Graph, A: Set[int], k: int, rate_fn: Callable[[ig.Graph, Set[int], Set[int]], float]) -> Tuple[float, Set[int]]:
    """
    Generic greedy forward search for any monotonic rate function.

    Parameters
    ----------
    graph : igraph.Graph
        The input graph.
    A : set[int]
        Sink node indices.
    k : int
        Source budget.
    rate_fn : callable
        A function (graph, S, A) → float computing ρ(S, A).

    Returns
    -------
    r_hat : float
        Achieved multicast rate.
    S : set[int]
        Selected source set.
    """
    S, r_hat = set(), 0.0
    V = set(range(graph.vcount())) - A

    while len(S) < k:
        best_gain, best_node = -1, None
        for v in V - S:
            new_rate = rate_fn(graph, S | {v}, A)
            gain = new_rate - r_hat
            if gain > best_gain:
                best_gain = gain
                best_node = v
        if best_node is None:
            break
        S.add(best_node)
        r_hat += best_gain

    return r_hat, S

def lazy_greedy_forward(graph: ig.Graph, A: Set[int], k: int, rate_fn: Callable[[ig.Graph, Set[int], Set[int]], float]) -> Tuple[float, Set[int]]:
    """
    Lazy-greedy forward search for any rate function.
    """
    import heapq
    S, r_hat = set(), 0.0
    V = set(range(graph.vcount())) - A

    pq = [(-rate_fn(graph, {v}, A), v) for v in V]
    heapq.heapify(pq)

    while len(S) < k and pq:
        neg_cached_gain, v = heapq.heappop(pq)
        if v in S:
            continue
        true_gain = rate_fn(graph, S | {v}, A) - r_hat
        if true_gain < -neg_cached_gain:
            heapq.heappush(pq, (-true_gain, v))
            continue
        S.add(v)
        r_hat += true_gain

    return r_hat, S