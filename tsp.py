import random
import time
import heapq as hq
from typing import List
import math
import sys
# This is class for the graph nodes
class Node:

    def __init__(self, name, latitude = 0, longitude = 0):
        self.name = name
        self.edge_set = set()
        self.neighbour_nodes = set()
        self.latitude = latitude
        self.longitude = longitude

    def connect(self, node):
        con = (self.name, node.name)
        self.edge_set.add(con)
        self.neighbour_nodes.add(node)

    def __lt__(self, other):
        return (self.name < other.name) 

    def __gt__(self, other):
        return (self.name > other.name)

            
#this is a class for the edges
class Edge:

    def __init__(self, left, right, weight=1):
        self.left = left
        self.right = right

        self.weight = weight

#this is a class for the graph
class Graph:

    def __init__(self, directed = False):
        self.verticies = {}
        self.edges = {}
        self.directed = directed #This is property for the graph if it is directed or not.
      
    def add_node(self, node):
        if node.name not in self.verticies:
            self.verticies[node.name] = node
        else:
            print("Node with the same name exists.")
    
    def add_edge(self, left, right, weight=1):
        if left.name not in self.verticies:
            self.verticies[left.name] = left
        
        if right.name not in self.verticies:
            self.verticies[right.name] = right

        edge = Edge(left, right, weight)
        key = (left.name, right.name)
        self.edges[key] = edge
        left.connect(right)

        if not self.directed:
            edge = Edge(right, left, weight) 
            key = (right.name, left.name)
            self.edges[key] = edge
            right.connect(left)
 
    def removeEdge(self, left, right):
        if ((left.name, right.name)) in self.edges:
            del self.edges[(left.name, right.name)]
            left.edge_set.remove((left.name,right.name))
            left.neighbour_nodes.remove(right)

            if not self.directed:
                del self.edges[(right.name, left.name)]
                right.edge_set.remove((right.name,left.name))
                right.neighbour_nodes.remove(left)

    def removeNode(self, node):
        del self.verticies[node.name]

        for con_node in node.neighbour_nodes:
            con_node.neighbour_nodes.remove(node)
            con_node.edge_set.remove((con_node.name, node.name))
    
    def degree(self, node):
        if node.name in self.verticies:
            return len(node.neighbour_nodes)
        return 0
        
        
    def adjecency_list(self, node):
        if node.name in self.verticies:
            return node.neighbour_nodes

    @property
    def no_edges(self):
        return len(self.edges)
    @property
    def no_vertices(self):
        return len(self.verticies)
    def dijkstra_search(self, initial, target):
        #here are the hasmaps to hold the shortest distances to a node and pathes
        shortest_distances = {}
        path = {}
        
        

        visited = set()
        # here min heap data structue is used to place the one with smallest cost at the top.
        heap = []
        if initial.name not in self.verticies or target.name not in self.verticies:
            return ["Departure or Destination node not found", 0]
        if initial==target:
            return [initial.name, 0]

        shortest_distances[initial] = 0
        path[initial] = [initial.name]
        #A cost of 0 and initial node is added to the min heap.
        hq.heappush(heap, [0, initial])
        
        while heap:
            explored = hq.heappop(heap)
            if explored[1] in visited:
                continue
            if explored[1] == target:
                return [path[explored[1]], shortest_distances[explored[1]]]
            #here nodes are added only after they are explored. not when they are inserted into the heap.
            visited.add(explored[1])
            for neighbour in explored[1].neighbour_nodes:
                if neighbour in visited:
                    continue
                path_cost = explored[0] + self.edges[(explored[1].name, neighbour.name)].weight

                # this checks weather neighbour is in the heap and we get smaller path cost.
                # if neighbour is in the heap and smaller path is detected an array containing 
                # the new smaller path cost and new path is added into the heap. In this practice
                # we won't explore the one which was inserted before since the minimum is 
                # encounteredd first and inserted into the visited set.
                if neighbour in shortest_distances and path_cost> shortest_distances[neighbour]:
                    continue
                    
                shortest_distances[neighbour] = path_cost
                path[neighbour] = path[explored[1]] + [neighbour.name]
                hq.heappush(heap, [path_cost, neighbour])
        
        return ["path not found", 0] 
myGraph = Graph()

# TSP using Genetic algorithm
#randonmly generate 

def generate_random_path(graph: Graph) -> list:
    
    verticies = list(graph.verticies.copy())
    path = random.sample(verticies, len(verticies))
    # print(path)
    return path
    # print(path)

def generate_population(size: int, graph: Graph) -> list:
    population = []
    for _ in range(size):
        population.append(generate_random_path(graph))
    return population


def crossover(path1: list, path2: list) -> tuple:
    patha = path1[:]
    pathb = path2[:]
    cross_over_point = random.randint(0, len(patha)-1)
    cross1 = patha[cross_over_point:]
    cross2 = pathb[cross_over_point:]

    for city1 in cross1:
        pathb.remove(city1)
    pathb+=(cross1)

    for city2 in cross2:
        patha.remove(city2)
    patha+=(cross2)
    
    return patha, pathb
    
  
def mutation(path: list) -> list:
    index1, index2 = random.randint(0, len(path)-1), random.randint(0, len(path)-1)
    path[index1], path[index2] = path[index2], path[index1]
    return path

def fitness_func(path: list) -> int:
    fitness = 0
    for i in range(len(path)):
        fitness += myGraph.dijkstra_search(myGraph.verticies[path[i-1]], myGraph.verticies[path[i]])[1]
    return fitness

#uses tournament selection so that to keep the diversity.
# tournament of 3 is chosen for selection, this is to keep diversity 
#for both the good one and bad one which could have better chromosome for the next generation.
def selection(population: list) -> tuple:
    # print(population[0])
    # print(population[1])
    tournamet_indices1 = random.sample(range(0, len(population)-1), 3)
    tournamet_indices2 = random.sample(range(0, len(population)-1), 3)
    # print(tournamet_indices1, tournamet_indices2)
    tournament1 = []
    tournament2 = []

    for index in range(3):
        tournament1.append(population[tournamet_indices1[index]])
        tournament2.append(population[tournamet_indices2[index]])
    # print(tournament1)
    # print(tournament2)
    parent1 = sortpopulation(tournament1)[0]
    parent2 = sortpopulation(tournament2)[0]

    return parent1, parent2

    


def sortpopulation(population: list) -> list:
    return sorted(population, key=fitness_func)

def tsp_evolution(no_of_generation: int, population_size: int, graph: Graph) -> list:
    population = generate_population(population_size ,graph)
    # print(population)
    
    for i in range(no_of_generation):
        newGeneration = []

        for _ in range(int(population_size/2)):
            parent = selection(population)
            # print("-----------parents---------------")
            # print(parent[0])
            # print(parent[1])
            # print("------------children-------------")
            child1, child2 = crossover(parent[0], parent[1])
            # print(child1)
            # print(child2)
            # print("----------after mutation------------")

            child1 = mutation(child1)
            child2 = mutation(child2)
            # print(child1)
            # print(child2)
            # print("---------next generation-----------")
            
            newGeneration.append(child1)
            newGeneration.append(child2)
        nextGeneration = sortpopulation(population+newGeneration)[:population_size]
        # print(population)
        # print(len(population))
        population = nextGeneration
        # print("---------------assigned parrent ----------------------")
        # print(population)
        # print(len(population))
    population = sortpopulation(population)
    return population[0], fitness_func(population[0])
      

def bestneighbour_with_bestroutelength(path: list) -> tuple:
    neighbours = []
    for city_index in range(len(path)):
        for second_index in range(city_index, len(path)):
            neighbour = path.copy()
            neighbour[city_index], neighbour[second_index] = neighbour[second_index], neighbour[city_index]
            neighbours.append(neighbour)
    neighbour = sorted(neighbours, key= fitness_func)
    bestneighbour = neighbour[0]
    bestroute = fitness_func(bestneighbour)
    return bestneighbour, bestroute

def hill_climbing_tsp(graph: Graph):
    current_path = generate_random_path(graph)
    current_route_length = fitness_func(current_path)
    bestneignbour, best_route_length = bestneighbour_with_bestroutelength(current_path)
    while(best_route_length<current_route_length):
        current_route_length = best_route_length
        current_path = bestneignbour
        bestneignbour, best_route_length = bestneighbour_with_bestroutelength(current_path)
    return current_path, current_route_length


def accept_probability(current: int, best_neighbour: int, temperature: float):
    return math.exp(-abs(current - best_neighbour)/ temperature)


    

def simmulated_annealling(graph: Graph, temperature, iteration_limit, cut_off_temperature= 0 , alpha = 0.995):
    current_path = generate_random_path(graph)
    current_route_length = fitness_func(current_path)
    iteration = 1
    
    while(temperature>cut_off_temperature and iteration < iteration_limit):
        
        bestneignbour, best_route_length = bestneighbour_with_bestroutelength(current_path)

        if best_route_length < current_route_length:
            current_path = bestneignbour
            current_route_length = best_route_length
        elif accept_probability(current_route_length, best_route_length, temperature) > random.random():
            current_path = bestneignbour
            current_route_length = best_route_length
        
        temperature*= alpha
        iteration+=1



    return current_path, current_route_length


def main():
    inputFile = sys.argv[4]
            
    Graph_file = open(inputFile, 'r')
    Lines = Graph_file.readlines()


    for line in Lines:
        node1, node2, weight = line.strip().split()
        if node1 not in myGraph.verticies:
            myGraph.add_node(Node(node1))
        if node2 not in myGraph.verticies:
            myGraph.add_node(Node(node2))
        myGraph.add_edge(myGraph.verticies[node1], myGraph.verticies[node2],int(weight))
    start = time.time()
    # this is for genetic algorithm 
    if sys.argv[2] == "ga":        
        print(tsp_evolution(100, 15, myGraph))
        print("finished in {} seconds.".format(time.time()- start))
    # this is for simmulated annealling    
    elif sys.argv[2] == "sa":
        print(simmulated_annealling(myGraph, 39, 12))
        print("finished in {} seconds.".format(time.time()- start))

    # this is for hill climbing
    elif sys.argv[2] == "hc":
        print(hill_climbing_tsp(myGraph))
        print("finished in {} seconds.".format(time.time()- start))

main()




# path = generate_random_path(myGraph)
# path2 = generate_random_path(myGraph)

# print(generate_population(2, myGraph))

# print(tsp_evolution(10, 10, myGraph))
# print(fitness_func(path))
# print(path)
# print("--------------------------------------------------")
# hello = (selection(generate_population(20, myGraph)))
# # print(generate_population(5, myGraph))
# print(fitness_func(hello[0]))
# print(fitness_func(hello[1]))
# # population = generate_population(20, myGraph)
# # population = sortpopulation(population)
# print ("----------------------------------")

# print(tsp_evolution(100, 20, myGraph))
