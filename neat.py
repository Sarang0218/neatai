import random
import time
import numpy as np
import copy
glob_innov = 0
glob_innov_map = []
def sig(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


class Node:
    def __init__(self, idx, layer=0, type="hidden"):

        self.idx = idx
        self.type = type
        self.layer = layer

        self.sum_out = 0
        self.sum_inp = 0

        # for drawing
        self.x = 0
        self.y = 0

    def set_data(self, data):
        self.data = data

    def __str__(self):
        return f"[{self.idx} - {self.type}] {self.data}"

    def __repr__(self):
        return f"[{self.idx} - {self.type}] {self.data}"


class Connection:
    def __init__(self, startidx, endidx, isrecurrent=False):
        global glob_innov
        global glob_innov_map
        
        # This only runs if the length of the glob_innov_map is smaller than the start_idx, so it doesn't get a index overflow.
        while len(glob_innov_map) <= startidx:
          glob_innov_map.append([])   
        
        # This only runs if the length of the glob_innov_map[start_idx] is smaller than the end idx, so it doesn't get a index overflow.
        while len(glob_innov_map[startidx]) <= endidx:
            glob_innov_map[startidx].append(glob_innov)
            glob_innov += 1
            
        self.evolution = glob_innov_map[startidx][endidx]         

        self.start = startidx
        self.end = endidx
        self.enabled = True
        self.weight = (random.random() - 0.5)*10
        self.isrecurrent = isrecurrent
        

    def __repr__(self):
        return f"[{self.evolution}] {self.start} -> {self.end}\n"


class Model:
    def __init__(self, inputN, outputN, hiddenN, biasN, pConn):
        global glob_innov
        
        self.nodes = []
        self.connections = []
        
        ### just in case ###
        self.inputN = inputN
        self.biasN = biasN
        self.outputN = outputN

        ######################

        self.fitness = 0
        self.speciesID = 0
        self.adjusted_F = 0
        #####################

        for _ in range(inputN):
            self.nodes.append(Node(len(self.nodes), 0, "input"))
        for _ in range(biasN):
            self.nodes.append(Node(len(self.nodes), 0, "bias"))

        if hiddenN > 0:
            for _ in range(outputN):
                self.nodes.append(Node(len(self.nodes), 2, "output"))
            for _ in range(hiddenN):
                self.nodes.append(Node(len(self.nodes), 1, "hidden"))
        else:
            for _ in range(outputN):
                self.nodes.append(Node(len(self.nodes), 1, "output"))

        for i in range(0, inputN+biasN):
            for j in range(inputN+biasN+outputN, len(self.nodes)):
                if random.random() < pConn:
                    self.connections.append(Connection(
                        self.nodes[i].idx, self.nodes[j].idx))
                    
        for i in range(inputN+biasN, inputN+biasN+outputN):
            for j in range(inputN+biasN+outputN, len(self.nodes)):
                if random.random() < pConn:
                    self.connections.append(Connection(
                        self.nodes[j].idx, self.nodes[i].idx))
                    

        if hiddenN == 0:
            for i in range(0, inputN+biasN):
                for j in range(inputN+biasN, inputN+biasN+outputN):
                    if random.random() < pConn:
                        self.connections.append(Connection(
                            self.nodes[i].idx, self.nodes[j].idx))
                        

    def load_inputs(self, inp):
        for i in range(self.inputN):
            self.nodes[i].sum_inp = inp[i]
            self.nodes[i].sum_out = inp[i]

    def forward(self):
        clayer = 1
        coldsearch = self.searchlayer(clayer-1)
        csearch = self.searchlayer(clayer)
        while len(csearch) != 0:
            for node in csearch:
                self.nodes[node].suminp = 0
                for snode in coldsearch:
                    for conn in self.searchconns(snode):

                        if self.connections[self.innov_to_idx(conn)].end == node:

                            self.nodes[node].sum_inp += self.nodes[snode].sum_out * \
                                self.connections[self.innov_to_idx(conn)].weight
                self.nodes[node].sum_out = sig(self.nodes[node].sum_inp)
            clayer += 1
            coldsearch = csearch
            csearch = self.searchlayer(clayer)

    def get_output(self, idx):
        return self.nodes[idx].sum_out

    def innov_to_idx(self, innov):
      i = 0
      for k in self.connections:
        if k.evolution == innov:
          return i
        i+=1
    
    ################ LEGACY ###################
    def update_innov(self):
        glob_innov = len(self.connections)

    # def mutate(self):
    #     k = random.random()
    #     if k < 0.4:
    #         self.mutate_conn()
    #     elif k < 0.7:
    #         self.mutate_w()
    #     else:
    #         self.mutate_add()

    # def mutate_add(self):
    #     for k in self.nodes:
    #         for j in self.nodes:
    #             if k.layer < j.layer:
    #                 if random.random() > 0.2:
    #                     self.connections.append(Connection(k.idx, j.idx))

    # def mutate_conn(self):
    #     conn = random.randint(0, len(self.connections)-1)
    #     enode = self.connections[conn].end
    #     enode_ob = self.nodes[enode]
    #     self.push_nodes(enode_ob.layer)
    #     newnode = Node(len(self.nodes), enode_ob.layer-1)

    #     self.nodes.append(newnode)
    #     self.connections[conn].end = newnode.idx
    #     self.connections.append(Connection(newnode.idx, enode, glob_innov))
    #     glob_innov += 1

    # def mutate_w(self):
    #     self.connections[random.randint(
    #         0, len(self.connections)-1)].weight += (random.random()-0.5)

    # def forward(self,input_,bias):
    #   kix = 0
    #   for i in self.nodes:
    #     if i.type == "input":
    #       i.data = input_[kix]
    #       kix+=1
    #     elif i.type == "bias":
    #       i.data = bias
    #   clayer = 1

    #   while len(self.searchlayer(clayer)) != 0:

    #     for cnode in self.searchlayer(clayer):

    #       for n in self.searchlayer(clayer-1):

    #         for enor in self.searchconns(n):

    #           if self.connections[enor].end == cnode:
    #             self.nodes[cnode].z += self.connections[enor].weight * self.nodes[self.connections[enor].start].data

    #       self.nodes[cnode].data = sig(self.nodes[cnode].z)

    #     clayer+=1

    #   return_val = []
    #   for sres in self.searchtype("output"):
    #     return_val.append(self.nodes[sres].data)
    #   return return_val

    def searchtype(self, type_):
        k = []
        for n in self.nodes:

            if n.type == type_:
                k.append(n.idx)
        return k

    def searchlayer(self, lay):

        returnsha = []
        for i in self.nodes:
            if i.layer == lay:
                returnsha.append(i.idx)
        return returnsha

    def searchconns(self, nodeid):
        beta = []
        for conns in self.connections:
            if conns.start == nodeid:
                beta.append(conns.evolution)
        return beta

    def push_nodes(self, pushlayer):
        for node in self.nodes:
            if node.layer >= pushlayer:
                
                node.layer += 1

def max_innov(conns):
  cmax = 0
  for conn in conns:
    if conn.evolution > cmax:
      cmax = conn.evolution
  return cmax

def compdiff(mod1, mod2, c1=1,c2=1,c3=0.4, debug=False):
  excess = 0
  disjoint = 0
  weightdiff = 0
  c=0
  m1max = max_innov(mod1.connections) 
  m2max = max_innov(mod2.connections) 
  wdsum = 0
  if m1max > m2max:
    for conn in mod1.connections:
      if conn.evolution > m2max:
        excess += 1  
      if conn.evolution in [k.evolution for k in mod2.connections]:
        for k in mod2.connections: 
          if k.evolution == conn.evolution:
            wdsum += abs(conn.weight - k.weight)
            c+=1
            break
        
  else:
    for conn in mod2.connections:
      if conn.evolution > m1max:
        excess += 1
      if conn.evolution in [k.evolution for k in mod1.connections]:
        for k in mod1.connections: 
          if k.evolution == conn.evolution:
            wdsum += abs(conn.weight - k.weight)
            c+=1
            break
  for mod1conn in mod1.connections:
    if (mod1conn.evolution <= m2max) and (mod1conn.evolution not in [mod2conn.evolution for mod2conn in mod2.connections]):
      disjoint+=1
  for mod2conn in mod2.connections:
    if (mod2conn.evolution <= m1max) and (mod2conn.evolution not in [mod1conn.evolution for mod1conn in mod1.connections]):
      disjoint+=1
      
      
  try:
    weightdiff = wdsum/c
  except ZeroDivisionError:
    weightdiff = 1000000000
    
  N = max(len(mod1.connections),len(mod2.connections))
  
  CD = c1*excess/N + c2*disjoint/N + c3*weightdiff
  if debug:
    print(excess, disjoint, weightdiff, N)
  return CD
  

class StupidConnection:
  def __init__(self, innov, weight=0):
    self.evolution = innov
    self.weight = weight

class StupidModel:
  def __init__(self):
    self.connections = []
def testCD():
  mod1 = StupidModel()
  mod1.connections = [
    StupidConnection(1, -3.5),
    StupidConnection(2, 5.2),
    StupidConnection(3, 9),
    StupidConnection(11),
    StupidConnection(17, -22),
    StupidConnection(4, 1),
    StupidConnection(19),
    StupidConnection(29),
    StupidConnection(21, 11)
  ]
  mod2 = StupidModel()
  mod2.connections = [
    StupidConnection(1, 4),
    StupidConnection(2, -14),
    StupidConnection(3, 2.5),
    StupidConnection(9),
    StupidConnection(12),
    StupidConnection(17,10),
    StupidConnection(4, -1),
    StupidConnection(36),
    StupidConnection(21, 5),
    StupidConnection(34),
    StupidConnection(16)
  ]
  
  print(compdiff(mod1,mod2, debug=True))
  
  
  
  
    
      
    
   
def speciate(models, threshold=4, gen0=False):
    samples = []
    pool = []

    def assign_species(model, samples, pool):
        for sample in samples:
            if compdiff(sample, model) < threshold:
                model.speciesID = sample.speciesID
                pool[sample.speciesID - 1].append(model)
                return True
        return False

    for model in models:
        if gen0:
            end_circuit = assign_species(model, samples, pool)
            if not end_circuit:
                model.speciesID = len(samples) + 1
                samples.append(model)
                pool.append([model])
        else:
            assigned = assign_species(model, samples, pool)
            if not assigned:
                model.speciesID = len(samples) + 1
                samples.append(model)
                pool.append([model])

    poolfitness = []
    poolsum = 0
    for species in pool:
        species_fit_sum = sum(model.fitness for model in species)
        poolfitness.append(species_fit_sum / len(species))
        poolsum += species_fit_sum

    glob_avg = poolsum / sum(len(species) for species in pool)
    poolallowed = [round(fit / glob_avg * len(species), 0) for fit, species in zip(poolfitness, pool)]

    return poolallowed

    
  
  
  
  
  
    
      
      
    
  
  
      
    
      
        
        
  