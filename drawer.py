import pygame
import sys
import math
import random
import numpy as np
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

def linsig(x):
    return max(1, 5 / (1 + np.exp(-0.4 * x)))

class NeuralNetworkVisualizer:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Neural Network Visualization")
        self.nodes = []
        self.connections = []
        self.data = ""
    def draw_neural_network(self):
        self.screen.fill(BLACK)
        self._draw_nodes(draw=False)
        self._draw_connections()
        self._draw_nodes()
        self.showdata()
        pygame.display.flip()

    def update_network(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections
        self.draw_neural_network()

    def _draw_connections(self):
        for connection in self.connections:
            start_node = self.nodes[connection.start]
            end_node = self.nodes[connection.end]
            weight = connection.weight
            thickness = int(linsig(weight)) 
            start_x = start_node.x
            start_y = start_node.y
            end_x = end_node.x
            end_y = end_node.y
            pygame.draw.line(self.screen, GREEN, (start_x, start_y), (end_x, end_y), thickness)

    def _draw_nodes(self,draw=True):
        font = pygame.font.Font(None, 36)
        for layer in set(node.layer for node in self.nodes):
            layer_nodes = [node for node in self.nodes if node.layer == layer]
            num_nodes = len(layer_nodes)
            for i, node in enumerate(layer_nodes):
                node_x = node.layer * 200 + 100
                node_y = (self.screen_height / (num_nodes + 1)) * (i + 1)
                node.x = node_x
                node.y = node_y
                if draw:
                    pygame.draw.circle(self.screen, WHITE, (node_x, int(node_y)), 20)
                    text_surface = font.render(str(node.idx), True, BLACK)
                    text_rect = text_surface.get_rect(center=(node_x, int(node_y)))
                    self.screen.blit(text_surface, text_rect)
                    
    def showdata(self):
        font = pygame.font.Font(None, 36)
        tsurf = font.render(self.data, True, WHITE)
        trect = tsurf.get_rect(topleft=(5,5))
        self.screen.blit(tsurf, trect)
        

