# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 21:26:15 2025

@author: Michael
"""
import numpy as np
from travelingSalesman import Node, Network

TASTES = ["sweet", "salty", "sour", "bitter", "umami"]

DESIRABILITY = np.array([
    # From:    sweet salty sour bitter umami
    [0.1,    0.5,  0.6,  0.6,  0.3 ],  # To sweet
    [0.5,    0.1,  0.4,  0.9,  0.1],  # To salty
    [0.6,    0.4,  0.1,  0.3,  0.2],  # To sour
    [0.4,    0.7,  0.3,  0.1,  0.4],  # To bitter
    [0.3,    0.1,  0.4,  0.9,  0.1],  # To umami
])

FAT_EFFECT = 10*np.array([
    [ 0.0,  0.0, -0.1,  0.0,  0.0],  # TO sweet
    [ 0.0,  0.0, -0.1,  0.0,  0.0],  # TO salty
    [ 0.1,  0.1,  0.0,  0.0,  0.1],  # TO sour (encourage sour if prior was fatty)
    [ 0.0,  0.0,  0.0,  0.0,  0.0],  # TO bitter
    [ 0.0,  0.0,  0.0,  0.0,  0.0],  # TO umami
])

class FoodNode(Node):
    def __init__(self, name, sweet, salty, sour, bitter, umami, fat):
        self.name = name
        self.taste = np.array([sweet, salty, sour, bitter, umami])
        self.fat =fat
        
    def cost_to(self, other):
        if self.name == "dummy" or other.name == "dummy":
            return 0

        # 1. Ideal next profile = current * desirability matrix
        ideal_next =(DESIRABILITY+FAT_EFFECT*self.fat)@self.taste 
        match_score=np.dot(ideal_next, other.taste)
        # 3. Cost = negative match (so lower cost = better match)
        return round(50000-match_score*1000)

# Example usage
foods = [
    FoodNode("dummy", 0, 0, 0, 0, 0, 0.0),

    FoodNode("Lemon Vinaigrette", 0.1, 0.2, 0.9, 0.0, 0.1, 0.0),  # bright and sour

    FoodNode("Grilled Chicken", 0.1, 0.5, 0.1, 0.0, 0.8, 0.3),    # savory/umami and salty

    FoodNode("Roasted Beet Salad", 0.4, 0.3, 0.3, 0.0, 0.2, 0.2),  # earthy sweetness, some sour

    FoodNode("Bitter Greens", 0.0, 0.2, 0.2, 0.8, 0.1, 0.1),       # mostly bitter

    FoodNode("Tomato Bruschetta", 0.2, 0.4, 0.5, 0.0, 0.5, 0.1),   # acidic and umami-rich

    FoodNode("Smoked Salmon", 0.0, 0.6, 0.1, 0.1, 0.9, 0.6),       # strong umami, salty, fatty

    FoodNode("Dark Chocolate", 0.3, 0.1, 0.0, 0.7, 0.1, 0.4),      # more bitter than sweet

    FoodNode("Fresh Strawberries", 0.8, 0.1, 0.4, 0.0, 0.0, 0.0),  # fruity sweet + a touch of acid

    FoodNode("Parmesan Cheese", 0.0, 0.7, 0.0, 0.1, 1.0, 0.7),     # umami bomb, salty, fatty

    FoodNode("Spiced Lentil Soup", 0.1, 0.4, 0.2, 0.1, 0.8, 0.5),  # warming, savory, slightly acidic
]


network=Network(foods)
network.solve_tsp()


route, cost = network.solve_tsp()

print("Route:", ' -> '.join(network.nodes[i].name for i in route))
print("Total cost (scaled):", cost)