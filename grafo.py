from typing import List, Tuple, Dict
import networkx as nx
import sys
import math

import heapq

INFTY = sys.float_info.max

class Vertice:
    def __init__(self, id=None, calles=[], coordenadas=()):
        self.id = id
        self.calles = calles
        self.coordenadas = coordenadas


class Grafo:
    # Diseñar y construirl a clase grafo
    def __init__(self, dirigido=False, vertices_ids={}, aristas=[], vertices_coords={}):
        """ Crea un grafo dirigido o no dirigido.

        Args:
            dirigido: Flag que indica si el grafo es dirigido o no.
        Returns: Grafo o grafo dirigido (según lo indicado por el flag)
        inicializado sin vértices ni aristas.
        """
        self.dirigido = dirigido
        self.vertices_ids = {}
        self.aristas = {}
        self.vertices_coords = {}
        self.vertices = []
        self.matriz_adyacencia = {}

    #### Operaciones básicas del TAD ####
    def es_dirigido(self) -> bool:
        """ Indica si el grafo es dirigido o no

        Args: None
        Returns: True si el grafo es dirigido, False si no.
        """
        return self.dirigido

    def agregar_vertice(self, v: object) -> None:
        """ Agrega el vértice v al grafo.
    
        Args: v vértice que se quiere agregar
        Returns: None
        """
        self.vertices_ids[v.id]= v
        self.vertices.append(v)
        self.matriz_adyacencia[v] = set()
        if v.coordenadas:
            self.vertices_coords[v.coordenadas] = v
        return None

    def agregar_arista(self, s: object, t: object, data: object = {'codigo':-1,'calle':'-1'}, weight: float = 1) -> None:
        """ Si los objetos s y t son vértices del grafo, agrega
        una orista al grafo que va desde el vértice s hasta el vértice t
        y le asocia los datos "data" y el peso weight.
        En caso contrario, no hace nada.

        Args:
            s: vértice de origen (source)
            t: vértice de destino (target)
            data: datos de la arista
            weight: peso de la arista
        Returns: None
        """
        if s in self.vertices and t in self.vertices:
            self.aristas[(s, t)] = {"codigo": data['codigo'], "calle": data['calle'], "weight": weight}
            self.matriz_adyacencia[s].add(t)
            if not self.dirigido:
                self.matriz_adyacencia[t].add(s)
        return None

    def eliminar_vertice(self, v: object) -> None:
        """ Si el objeto v es un vértice del grafo lo elimiina.
        Si no, no hace nada.

        Args: v vértice que se quiere eliminar
        Returns: None
        """
        if v in self.vertices:
            self.vertices_ids.pop(v.id)
            self.vertices.remove(v)
            self.matriz_adyacencia.pop(v)
            if v.coordenadas:
                self.vertices_coords.pop(v.coordenadas)
        return None

    def eliminar_arista(self, s: object, t: object) -> None:
        """ Si los objetos s y t son vértices del grafo y existe
        una arista de u a v la elimina.
        Si no, no hace nada.

        Args:
            s: vértice de origen de la arista
            t: vértice de destino de la arista
        Returns: None
        """
        if (s,t) in self.aristas:
            self.aristas.pop((s,t))
        elif (t,s) in self.aristas:
            self.aristas.pop((t,s))
        self.matriz_adyacencia[s].remove(t)
        if not self.dirigido:
            self.matriz_adyacencia[t].remove(s)
        return None

    def obtener_arista(self, s: object, t: object) -> Tuple[object, float] or None:
        """ Si los objetos s y t son vértices del grafo y existe
        una arista de u a v, devuelve sus datos y su peso en una tupla.
        Si no, devuelve None

        Args:
            s: vértice de origen de la arista
            t: vértice de destino de la arista
        Returns: Una tupla (a,w) con los datos de la arista "a" y su peso
        "w" si la arista existe. None en caso contrario.
        """
        
        if (s,t) in self.aristas:
            return (self.aristas[(s,t)]["codigo"], self.aristas[(s,t)]["calle"], self.aristas[(s,t)]["weight"])
        if (t,s) in self.aristas:
            return (self.aristas[(t,s)]["codigo"], self.aristas[(t,s)]["calle"], self.aristas[(t,s)]["weight"])
        return None

    def lista_adyacencia(self, u: object) -> List[object] or None:
        """ Si el objeto u es un vértice del grafo, devuelve
        su lista de adyacencia.
        Si no, devuelve None.

        Args: u vértice del grafo
        Returns: Una lista [v1,v2,...,vn] de los vértices del grafo
        adyacentes a u si u es un vértice del grafo y None en caso
        contrario
        """
        if u in self.vertices:
            return list(self.matriz_adyacencia[u])
        return None

    #### Grados de vértices ####
    def grado_saliente(self, u: object) -> int or None:
        """ Si el objeto u es un vértice del grafo, devuelve
        su grado saliente.
        Si no, devuelve None.

        Args: u vértice del grafo
        Returns: El grado saliente (int) si el vértice existe y
        None en caso contrario.
        """
        if u in self.vertices:
            return len(self.lista_adyacencia(u))
        return None

    def grado_entrante(self, u: object) -> int or None:
        """ Si el objeto u es un vértice del grafo, devuelve
        su grado entrante.
        Si no, devuelve None.

        Args: u vértice del grafo
        Returns: El grado entrante (int) si el vértice existe y
        None en caso contrario.
        """
        if u in self.vertices:
            if not self.es_dirigido:
                return self.grado_saliente(u)
            count = 0
            for _, ady in self.matriz_adyacencia.items():
                if u in ady:
                    count += 1
            return count
        return None

    def grado(self, u: object) -> int or None:
        """ Si el objeto u es un vértice del grafo, devuelve
        su grado si el grafo no es dirigido y su grado saliente si
        es dirigido.
        Si no pertenece al grafo, devuelve None.

        Args: u vértice del grafo
        Returns: El grado (int) o grado saliente (int) según corresponda
        si el vértice existe y None en caso contrario.
        """
        if u in self.vertices:
            return self.grado_saliente(u)
        return None

    #### Algoritmos ####

    def dijkstra_parada(self,origen: object, destino: object) -> bool:
        padre = {i: None for i in self.vertices}
        visitados = {i: False for i in self.vertices}
        distancia = {i: math.inf for i in self.vertices}
        distancia[origen] = 0
        
        q = []
        heapq.heappush(q, (0, origen.id))
        while q:
            _, u = heapq.heappop(q)
            u = self.vertices_ids[u]
            if u == destino:
                return padre
            if visitados[u] == False:
                visitados[u] = True
                lst_u = self.matriz_adyacencia[u]
                for v in lst_u:
                    if visitados[v] == False:
                        dist_uv = self.obtener_arista(u, v)[2]
                        if distancia[v] > distancia[u] + dist_uv:
                            distancia[v] = distancia[u] + dist_uv
                            padre[v] = u
                            heapq.heappush(q, (distancia[v], v.id))
        return padre

    def dijkstra(self, origen: object) -> Dict[object, object]:
        """ Calcula un Árbol Abarcador Mínimo para el grafo partiendo
        del vértice "origen" usando el algoritmo de Dijkstra. Calcula únicamente
        el árbol de la componente conexa que contiene a "origen".

        Args: origen vértice del grafo de origen
        Returns: Devuelve un diccionario que indica, para cada vértice alcanzable
        desde "origen", qué vértice es su padre en el árbol abarcador mínimo.
        """
        return self.dijkstra_parada(origen, None)


    def camino_minimo(self, origen: object, destino: object) -> List[object]:
        padre = self.dijkstra_parada(origen, destino)
        if destino in padre:
            camino = [destino]
            while camino[0] != origen:
                camino.insert(0, padre[camino[0]])
            return camino
        return 'No hay camino'

    def prim(self) -> Dict[object, object]:
        """ Calcula un Árbol Abarcador Mínimo para el grafo
        usando el algoritmo de Prim.

        Args: None
        Returns: Devuelve un diccionario que indica, para cada vértice del
        grafo, qué vértice es su padre en el árbol abarcador mínimo.
        """
        padre = {i: None for i in self.vertices}
        coste_minimo = {i: INFTY for i in self.vertices}
        coste_minimo[self.vertices[0]] = 0
        q = []
        for key, value in coste_minimo.items():
            heapq.heappush(q, (value, key.id))
        while q:
            _, u = heapq.heappop(q)
            u = self.vertices_ids[u]
            q_ids = {i[1] for i in q}
            for w in set(self.lista_adyacencia(u)):
                if w.id in q_ids:
                    if coste_minimo[w] > self.obtener_arista(u, w)[2]:
                        q.remove((coste_minimo[w], w.id))
                        coste_minimo[w] = self.obtener_arista(u, w)[2]
                        padre[w] = u
                        heapq.heappush(q, (coste_minimo[w], w.id))
        return padre
                             

    def kruskal(self) -> List[Tuple[object, object]]:
        """ Calcula un Árbol Abarcador Mínimo para el grafo
        usando el algoritmo de Prim.

        Args: None
        Returns: Devuelve una lista [(s1,t1),(s2,t2),...,(sn,tn)]
        de los pares de vértices del grafo
        que forman las aristas del arbol abarcador mínimo.
        """
        aristas = []
        c = {}
        for v in self.vertices:
            c[v] = {v}
        l = []
        for ar, data in self.aristas.items():
            ar = (ar[0].id, ar[1].id)
            heapq.heappush(l, (data['weight'], ar))
        while l:
            _, a = heapq.heappop(l)
            a = (self.vertices_ids[a[0]], self.vertices_ids[a[1]])
            if c[a[0]] != c[a[1]]:
                aristas.append((a[0], a[1]))
                c[a[0]] = c[a[0]] | c[a[1]]
                c[a[1]] = c[a[0]]
            for w in c[a[0]]:
                c[w] = c[a[0]]
        return aristas

    #### NetworkX ####
    def convertir_a_NetworkX(self) -> nx.Graph or nx.DiGraph:
        """ Construye un grafo o digrafo de Networkx según corresponda
        a partir de los datos del grafo actual.

        Args: None
        Returns: Devuelve un objeto Graph de NetworkX si el grafo es
        no dirigido y un objeto DiGraph si es dirigido. En ambos casos,
        los vértices y las aristas son los contenidos en el grafo dado.
        """
        if self.dirigido:
            G2 = nx.DiGraph()
        else:
            G2 = nx.Graph()
        G2.add_nodes_from([i for i in self.vertices_ids])
        G2.add_edges_from([(i.id,j.id,data) for (i,j), data in self.aristas.items()])
        
        return G2


