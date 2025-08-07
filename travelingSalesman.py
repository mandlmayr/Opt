from abc import ABC, abstractmethod
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

class Node(ABC):
    @abstractmethod
    def cost_to(self, other):
        """Return cost from this node to another node."""
        pass

class CityNode(Node):
    def __init__(self, x, y, name=""):
        self.x = x
        self.y = y
        self.name = name

    def cost_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return int(((dx ** 2 + dy ** 2) ** 0.5) * 1000)  # scaled to int for OR-Tools

class Network:
    def __init__(self, nodes):
        self.nodes = nodes

    def build_distance_matrix(self):
        size = len(self.nodes)
        matrix = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i != j:
                    matrix[i][j] = self.nodes[i].cost_to(self.nodes[j])
        return matrix

    def solve_tsp(self):
        distance_matrix = self.build_distance_matrix()
        num_nodes = len(distance_matrix)
        manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_idx, to_idx):
            return distance_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

        transit_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        solution = routing.SolveWithParameters(search_params)
        if not solution:
            return None, None

        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # Return to depot

        total_cost = solution.ObjectiveValue()
        return route, total_cost
    

# # Example usage:
# cities = [
#     CityNode(0, 0, "A"),
#     CityNode(1, 2, "B"),
#     CityNode(4, 0, "C"),
#     CityNode(5, 3, "D"),
# ]

# network = Network(cities)
# route, cost = network.solve_tsp()

# print("Route:", ' -> '.join(network.nodes[i].name for i in route))
# print("Total cost (scaled):", cost)
