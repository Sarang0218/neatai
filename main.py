import drawer
import neat
import copy
import pygame
import sys
# print(m.forward([2]))


training = [
    [0, 0],
    [1, 1],
    [1, 0],
    [0, 1]
]
train_ans = [[0], [1], [1], [0]]
threshold = 99
thresh_stepsize = 0.5
target_specs = 7 

pop = []
popsize = 50

# GEN 0 #
for i in range(popsize):
    model = neat.Model(inputN=2, outputN=1, hiddenN=0, biasN=1, pConn=1)
    k = 0

    for test in training:
        model.load_inputs(test)
        model.forward()
        model.fitness += 1-abs(model.get_output(3)-train_ans[k][0])
        k += 1
    pop.append(model)


#   poploss = {}
#   for m in pop:
#     m.mutate()
#     loss = 0
#     for k in range(len(training)):
#       ans = m.forward(training[k],1)
#       loss += (ans[0] - train_ans[k][0])**2
#     poploss[m.modidx] = loss
#   sorted_dict = dict(sorted(poploss.items()))
#   allt = len(sorted_dict)
#   select_few = [allt-1, allt-2]
#   pop = []
#   for awemod in select_few:
#     for _ in range(5):
#       M = find_model(awemod)
#       m = copy.deepcopy(M)
#       m.modidx = len(allpop)
#       m.mutate()
#       pop.append(m)
#       allpop.append(m)
#   return pop

# Call the function to draw the neural network diagram

visualizer = drawer.NeuralNetworkVisualizer()
visualizer.nodes = pop[-1].nodes
visualizer.connections = pop[-1].connections
visualizer.draw_neural_network()

# print(neat.speciate(pop, threshold=1, gen0=True))
# print([jk.speciesID for jk in pop])
for mod in pop:
    mod.speciesID += 1
    
tmod = neat.Model(inputN=2, outputN=1, hiddenN=1000, biasN=1, pConn=1)
tmod.fitness = 3
pop.append(tmod)

# print([jk.speciesID for jk in pop])
# print(sum(neat.speciate(pop, threshold=1, gen0=False)))
# print([jk.speciesID for jk in pop])

pop[-1].load_inputs([0, 1])
pop[-1].forward()

visualizer.data = f"SPECIES {pop[-1].speciesID}"
# print(pop[-1].get_output(3))
# print(pop[-1].fitness)
neat.testCD()
# print(pop[-1].connections)
# print(neat.glob_innov_map)
# print(pop[-1].speciesID)

model_index=0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            model_index = (model_index + 1) % len(pop)  # Cycle through models
            current_model = pop[model_index]  # Switch to the next model
            visualizer.update_network(current_model.nodes, current_model.connections) 
            visualizer.data = f"SPECIES: {current_model.speciesID} ID: {model_index}"
            


pygame.quit()
sys.exit()


# kloss = 100000000000000
# for i in range(1000):
#     pop = natural_selection(pop)

#     showmod = pop[-1]
#     loss = 0
#     for k in range(len(training)):
#         ans = showmod.forward(training[k],1)

#         loss += (ans[0] - train_ans[k][0])**2

#     kloss = loss

#     # visualization_thread = threading.Thread(target=run_visualization)
#     # visualization_thread.start()
#     # visualizer.update_network(showmod.nodes, showmod.connections)
