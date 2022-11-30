import grafo as cg
import networkx as nx

gr = cg.Grafo()
a = cg.Vertice('a')
b = cg.Vertice('b')
c = cg.Vertice('c')
d = cg.Vertice('d')
e = cg.Vertice('e')
f = cg.Vertice('f')
g = cg.Vertice('g')
h = cg.Vertice('h')
i = cg.Vertice('i')


gr.agregar_vertice(a)
gr.agregar_vertice(b)
gr.agregar_vertice(c)
gr.agregar_vertice(d)
gr.agregar_vertice(e)
gr.agregar_vertice(f)
gr.agregar_vertice(g)
gr.agregar_vertice(h)
gr.agregar_vertice(i)
gr.agregar_arista(a,e,weight=21)
gr.agregar_arista(a,c,weight=81)
gr.agregar_arista(a,h,weight=9)
gr.agregar_arista(a,i,weight=104)

gr.agregar_arista(b,g,weight=11)
gr.agregar_arista(b,f,weight=2)

gr.agregar_arista(c,e,weight=12)
gr.agregar_arista(c,d,weight=4)
gr.agregar_arista(c,f,weight=20)
gr.agregar_arista(c,h,weight=64)

gr.agregar_arista(d,h,weight=1)
gr.agregar_arista(d,e,weight=8)
gr.agregar_arista(d,i,weight=2)
gr.agregar_arista(d,f,weight=6)

gr.agregar_arista(e,f,weight=4)
gr.agregar_arista(e,h,weight=7)

gr.agregar_arista(f,i,weight=12)

gr.agregar_arista(g,h,weight=16)

gr.agregar_arista(i,h,weight=3)

print([f'{key.id}: {value.id}' for key,value in gr.prim().items() if value is not None])
print([(i.id, j.id) for (i,j) in gr.kruskal()])
print([f'{key.id}: {value.id}' for key,value in gr.dijkstra(a).items() if value is not None])

camino = gr.camino_minimo(a, b)
camino_ids = [v.id for v in camino]
import matplotlib.pyplot as plt
G = gr.convertir_a_NetworkX()

aam = nx.Graph()
aam.add_nodes_from(camino_ids)
new_edges = [(camino_ids[i], camino_ids[i+1]) for i in range(len(camino_ids)-1)]
aam.add_edges_from([arista for arista in G.edges(data=True) if (arista[0], arista[1]) in new_edges or (arista[1], arista[0]) in new_edges])
print('')
print(aam.edges(data=True))
plt.figure(figsize=(10, 10))
plt.plot()
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_size=10, pos=pos)
nx.draw(aam, with_labels=False, node_size=10, edge_color='r', pos=pos)
edge_labels = {(i, j): data['weight'] for (i,j,data) in aam.edges(data = True)}

nx.draw_networkx_edge_labels(aam, pos, edge_labels)

plt.show()

e = [(1, 2), (2, 3), (3, 4)]  # list of edges
G = nx.Graph(e)
print(G.edges(data=True))
print(G.nodes(data=True))