import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    graph = nx.DiGraph(conflicts=0)
    for result in results:
        entities = {entity.node_id: entity for entity in result.entities}
        relations = [relation for relation in result.relations if relation.matches_schema()]
        for relation in relations:
            entities[relation.subject.node_id] = relation.subject
            entities[relation.object.node_id] = relation.object
        for node_id, entity in entities.items():
            if node_id not in graph:
                graph.add_node(node_id, type=entity.type, name=entity.name, sources=[])
            graph.nodes[node_id]["sources"] = sorted(set(graph.nodes[node_id]["sources"]) | {result.chunk_id})
        for relation in relations:
            left, right = relation.subject.node_id, relation.object.node_id
            if not graph.has_edge(left, right):
                graph.add_edge(left, right, type=relation.type, weight=0, sources=[])
            edge = graph.edges[left, right]
            edge["sources"] = sorted(set(edge["sources"]) | {result.chunk_id})
            edge["weight"] = len(edge["sources"])
    handled = set()
    for left, right, edge in list(graph.edges(data=True)):
        pair = frozenset((left, right))
        if pair in handled or edge["type"] != "REPLACED_BY" or not graph.has_edge(right, left):
            continue
        reverse = graph.edges[right, left]
        if reverse["type"] != "REPLACED_BY":
            continue
        handled.add(pair)
        graph.graph["conflicts"] += 1
        if edge["weight"] <= reverse["weight"]:
            graph.remove_edge(left, right)
        if edge["weight"] >= reverse["weight"] and graph.has_edge(right, left):
            graph.remove_edge(right, left)
    return graph
