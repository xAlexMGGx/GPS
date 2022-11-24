from typing import List, Tuple, Dict
import networkx as nx
import sys
import math

import heapq

INFTY = sys.float_info.max

class Vertice:
    def __init__(self, id=None, calles=[], nodos_adyacentes=[], coordenadas=()):
        self.id = id
        self.calles = calles
        self.nodos_adyacentes = nodos_adyacentes
        self.coordenadas = coordenadas

class Grafo:
    # Diseñar y construirl a clase grafo

    def __init__(self, dirigido=False, vertices_ids=[], aristas=[], vertices_data={}):
        """ Crea un grafo dirigido o no dirigido.

        Args:
            dirigido: Flag que indica si el grafo es dirigido o no.
        Returns: Grafo o grafo dirigido (según lo indicado por el flag)
        inicializado sin vértices ni aristas.
        """
        self.dirigido = dirigido
        self.vertices_ids = vertices_ids
        self.aristas = aristas
        self.vertices_data = vertices_data
        self.vertices = vertices_data.values()

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
        self.vertices_ids.append(v.id)
        self.vertices_data[v.coordenadas] = v

    def agregar_arista(self, s: object, t: object, data: object, weight: float = 1) -> None:
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
        self.aristas.append((s.id, t.id, {"data": data, "weight": weight}))
        return None

    def eliminar_vertice(self, v: object) -> None:
        """ Si el objeto v es un vértice del grafo lo elimiina.
        Si no, no hace nada.

        Args: v vértice que se quiere eliminar
        Returns: None
        """
        self.vertices_ids.remove(v.id)
        self.vertices_data.pop(v.coordenadas)
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
        for arista in self.aristas:
            if s.id in arista and t.id in arista:
                self.aristas.remove(arista)
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
        for arista in self.aristas:
            if s.id in arista and t.id in arista:
                return (arista[2]["data"], arista[2]["weight"])
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
            lista_adyacencia = []
            for arista in self.aristas:
                if u in arista:
                    lista_adyacencia.append(arista[1]) if u == arista[0] else lista_adyacencia.append(arista[0])
            return lista_adyacencia
        return None

    #### Grados de vértices ####
    def grado_saliente(self, u: Vertice) -> int or None:
        """ Si el objeto u es un vértice del grafo, devuelve
        su grado saliente.
        Si no, devuelve None.

        Args: u vértice del grafo
        Returns: El grado saliente (int) si el vértice existe y
        None en caso contrario.
        """
        if u in self.vertices:
            lista_entrante = []
            for aristas in self.aristas:
                if u == aristas[0]:
                    lista_entrante.append(aristas[1])
            return len(aristas)
        return None

    def grado_entrante(self, u: Vertice) -> int or None:
        """ Si el objeto u es un vértice del grafo, devuelve
        su grado entrante.
        Si no, devuelve None.

        Args: u vértice del grafo
        Returns: El grado entrante (int) si el vértice existe y
        None en caso contrario.
        """
        if u in self.vertices:
            lista_entrante = []
            for aristas in self.aristas:
                if u == aristas[1]:
                    lista_entrante.append(aristas[0])
            return len(aristas)
        return None

    def grado(self, u: Vertice) -> int or None:
        """ Si el objeto u es un vértice del grafo, devuelve
        su grado si el grafo no es dirigido y su grado saliente si
        es dirigido.
        Si no pertenece al grafo, devuelve None.

        Args: u vértice del grafo
        Returns: El grado (int) o grado saliente (int) según corresponda
        si el vértice existe y None en caso contrario.
        """
        if u in self.vertices:
            if self.es_dirigido():
                return self.grado_saliente(u)
            else:
                return len(u.nodos_adyacentes)
        return None

    #### Algoritmos ####
    def dijkstra(self, origen: object) -> Dict[object, object]:
        """ Calcula un Árbol Abarcador Mínimo para el grafo partiendo
        del vértice "origen" usando el algoritmo de Dijkstra. Calcula únicamente
        el árbol de la componente conexa que contiene a "origen".

        Args: origen vértice del grafo de origen
        Returns: Devuelve un diccionario que indica, para cada vértice alcanzable
        desde "origen", qué vértice es su padre en el árbol abarcador mínimo.
        """
        padre = {i: None for i in self.vertices}
        visitados = {i: False for i in self.vertices}
        distancia = {i: math.inf for i in self.vertices}
        distancia[origen] = 0
        q = [origen]
        while q:
            u = q.pop(0)
            if not visitados[u]:
                visitados[u] = True
                for v in self.lista_adyacencia(u):
                    if distancia[v] > distancia[u] + self.obtener_arista(u, v)[1]:
                        distancia[v] = distancia[u] + self.obtener_arista(u, v)[1]
                        padre[v] = u
                        q.append(v)
        return padre


    def camino_minimo(self, origen: object, destino: object) -> List[object]:
        pass

    def prim(self) -> Dict[object, object]:
        """ Calcula un Árbol Abarcador Mínimo para el grafo
        usando el algoritmo de Prim.

        Args: None
        Returns: Devuelve un diccionario que indica, para cada vértice del
        grafo, qué vértice es su padre en el árbol abarcador mínimo.
        """
        padre = {i: None for i in self.vertices}
        coste_minimo = {i: math.inf for i in self.vertices}
        coste_minimo[self.vertices[0]] = 0
        q = [self.vertices[0]]
        while q:
            # extraer el de coste mínimo
            u = sorted(q, key=lambda x: coste_minimo[x]).pop(0)
            for w in (set(self.lista_adyacencia(u)) & set(q)):
                if coste_minimo[w] > self.obtener_arista(u, w)[1]:
                    coste_minimo[w] = self.obtener_arista(u, w)[1]
                    padre[w] = u
                    
        
                    
            

    def kruskal(self) -> List[Tuple[object, object]]:
        """ Calcula un Árbol Abarcador Mínimo para el grafo
        usando el algoritmo de Prim.

        Args: None
        Returns: Devuelve una lista [(s1,t1),(s2,t2),...,(sn,tn)]
        de los pares de vértices del grafo
        que forman las aristas del arbol abarcador mínimo.
        """
        pass

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
        G2.add_nodes_from(self.vertices_ids)
        G2.add_edges_from(self.aristas)
        return G2


