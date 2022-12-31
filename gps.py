import pandas as pd
import grafo as gf
import networkx as nx
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def extract():
    cruces = pd.read_csv('cruces.csv', sep=';', encoding='latin1')
    direcciones = pd.read_csv('direcciones.csv', sep=';', encoding='latin1')
    direcciones = direcciones[direcciones['Coordenada X (Guia Urbana) cm'] != '000000-100']
    direcciones['Coordenada X (Guia Urbana) cm'] = direcciones['Coordenada X (Guia Urbana) cm'].astype('int64')
    direcciones['Coordenada Y (Guia Urbana) cm'] = direcciones['Coordenada Y (Guia Urbana) cm'].astype('int64')
    return cruces, direcciones


def clean(cruces, direcciones):
    # Quitamos los espacios innecesarios de los datasets
    for df in [cruces, direcciones]:
        for column in df.columns:
            if df[column].dtype == type(object):
                df[column] = df[column].str.strip()
    # En el nombre de las columnas reemplazamos ï¿½ por í usando regex
    direcciones.columns = direcciones.columns.str.replace('ï¿½', 'í')
    direcciones['Direccion'] = direcciones['Codigo de via'].astype(str) + '-' + direcciones['Literal de numeracion'].str.replace('[a-zA-Z. ]', '')
    direcciones.drop_duplicates('Direccion',keep='first',inplace=True)
    direcciones['Nombre completo de calle'] = direcciones['Clase de la via'] + ' ' + direcciones['Partícula de la vía'] + ' ' + direcciones['Nombre de la vía']
    return cruces, direcciones


def unify_vertices(cruces):
    # Ponemos una coordenada común de las glorietas (media de las coordenadas de los cruces)

    # Para empezar nos quedamos con un dataset de las glorietas (solo necesitamos la glorieta como primera via)
    cruces_glorietas = cruces[cruces['Clase de la via tratado'] == 'GLORIETA']
    # Guardamos el número de cruces de cada glorieta en un diccionario, usando value_counts y diferenciando por el código de la vía
    num_cruces = {}
    for glorieta in cruces_glorietas['Codigo de vía tratado'].unique():
        num_cruces[glorieta] = cruces_glorietas['Codigo de vía tratado'].value_counts()[glorieta]
    # Guardamos la suma de las coordenadas (diferenciamos entre X e Y) de los cruces de cada glorieta en un diccionario
    coords_glorietas = {}
    for glorieta in cruces_glorietas['Codigo de vía tratado'].unique():
        coords_glorietas[glorieta] = []
        coords_glorietas[glorieta].append(cruces_glorietas[cruces_glorietas['Codigo de vía tratado'] == glorieta]['Coordenada X (Guia Urbana) cm (cruce)'].sum())
        coords_glorietas[glorieta].append(cruces_glorietas[cruces_glorietas['Codigo de vía tratado'] == glorieta]['Coordenada Y (Guia Urbana) cm (cruce)'].sum())
    # Calculamos la media de las coordenadas de los cruces de cada glorieta
    for glorieta in coords_glorietas:
        coords_glorietas[glorieta][0] = coords_glorietas[glorieta][0] // num_cruces[glorieta]
        coords_glorietas[glorieta][1] = coords_glorietas[glorieta][1] // num_cruces[glorieta]
    # Finalmente, en el dataset de cruces, si hay una glorieta en un cruce, ponemos las coordenadas de la glorieta
    for index, row in cruces.iterrows():
        if row['Clase de la via tratado'] == 'GLORIETA':
            cruces.iloc[index, cruces.columns.get_loc('Coordenada X (Guia Urbana) cm (cruce)')] = coords_glorietas[row['Codigo de vía tratado']][0]
            cruces.iloc[index, cruces.columns.get_loc('Coordenada Y (Guia Urbana) cm (cruce)')] = coords_glorietas[row['Codigo de vía tratado']][1]
        elif row['Clase de la via que cruza'] == 'GLORIETA':
            cruces.iloc[index, cruces.columns.get_loc('Coordenada X (Guia Urbana) cm (cruce)')] = coords_glorietas[row['Codigo de via que cruza o enlaza']][0]
            cruces.iloc[index, cruces.columns.get_loc('Coordenada Y (Guia Urbana) cm (cruce)')] = coords_glorietas[row['Codigo de via que cruza o enlaza']][1]
    return cruces


def select_relevant_info(cruces, direcciones):
    d = direcciones[['Codigo de via', 'Coordenada X (Guia Urbana) cm', 'Coordenada Y (Guia Urbana) cm', 'Direccion']]
    d['tipo'] = ['direcciones' for i in range(len(d))]
    a = []

    for i in d['Direccion'].values:
        i = i.replace('ï¿½', '')
        try:
            a.append(int(i.split('-')[1]))
        except:
            print(i)
    d['Direccion'] = a
    d = d.rename(columns={'Codigo de via': 'codigo'})
    d = d.rename(columns={'Coordenada X (Guia Urbana) cm': 'x'})
    d = d.rename(columns={'Coordenada Y (Guia Urbana) cm': 'y'})

    d = d[d['x'] != '000000-100']
    d['codigo'] = d['codigo'].astype(int)


    d['x'] = d['x'].astype(int)
    d['y'] = d['y'].astype(int)


    c = cruces[['Codigo de vía tratado', 'Coordenada X (Guia Urbana) cm (cruce)', 'Coordenada Y (Guia Urbana) cm (cruce)']]
    c['tipo'] = ['cruces' for i in range(len(c))]

    c = c.rename(columns={'Codigo de vía tratado': 'codigo'})
    c = c.rename(columns={'Coordenada X (Guia Urbana) cm (cruce)': 'x'})
    c = c.rename(columns={'Coordenada Y (Guia Urbana) cm (cruce)': 'y'})
    c['codigo'] = c['codigo'].astype(int)
    c['x'] = c['x'].astype(int)
    c['y'] = c['y'].astype(int)

    # for cruce in c.iterrows():

    # ordenar d por coordenadas y codigo de via

    d = d.sort_values(by=['codigo', 'x', 'y'])

    d['par'] = d['Direccion']%2== 0
    d['par'] = d['par'].astype(int)
    return c, d


def assign_vertices(c, d):
    numeros_cruces = []
    for cruce in c.iterrows():
        d_temp = d[d['codigo'] == cruce[1]['codigo']]
        x, y = cruce[1]['x'], cruce[1]['y']
        distancia_min = 10e10
        num = ''
        n_tot = len(d_temp)
        pares = sum(d_temp['par'])
        d_temp = d_temp[d_temp['par'] == 0] if pares < (n_tot-pares) else d_temp[d_temp['par'] == 1]
        for dir in d_temp.iterrows():
            x1, y1 = dir[1]['x'], dir[1]['y']
            distancia = ((x1 - x)**2 + (y1 - y)**2)**0.5
            distancia_min = min(distancia, distancia_min)
            if distancia_min == distancia:
                num = dir[1]['Direccion']
        numeros_cruces.append(num)
    c['numero'] = numeros_cruces
    c.sort_values(by=['codigo', 'numero'], inplace=True)
    c.reset_index(inplace=True, drop=True)
    with open('cruces_procesado.csv', 'w') as f:
        c.to_csv(f)
    return c


def get_weight(via, origen, destino, type_graph):
    velocidades = {
        'AUTOVIA': 100/60,
        'AVENIDA': 90/60,
        'CARRETERA': 70/60,
        'CALLEJON': 30/60,
        'CAMINO': 30/60,
        'ESTACION': 20/60,
        'PASADIZO': 20/60,
        'PLAZUELA': 20/60,
        'COLONIA': 20/60
    }
    if via not in velocidades:
        vel = 50/60
    else:
        vel = velocidades[via]
    distancia = ((origen[0] - destino[0])**2 + (origen[1] - destino[1])**2)**0.5
    distancia = distancia / 100000
    if type_graph == 'tiempo':
        return distancia/vel
    else:
        return distancia


def create_graph(cruces, c, type_graph):
    G = gf.Grafo(False, {}, [], {})
    id = 1
    last = None
    for idx, row in c.iterrows():
        coords = (row['x'], row['y'])
        if coords in G.vertices_coords:
            vertice_actual = G.vertices_coords[coords]
            vertice_actual.calles.append(row['codigo'])
        else:
            vertice_actual = gf.Vertice(id, [row['codigo']], (row['x'], row['y']))
            G.agregar_vertice(vertice_actual)
        if last:
            if last[1] == row['codigo'] and last[0].coordenadas != coords:
                nombre_calle = cruces[cruces['Codigo de vía tratado'] == row['codigo']]['Literal completo del vial tratado'].values[0]
                data = {'codigo': row['codigo'], 'calle': nombre_calle}
                weight = get_weight(cruces[cruces['Codigo de vía tratado'] == row['codigo']]['Clase de la via tratado'].values[0], last[0].coordenadas, coords, type_graph)
                G.agregar_arista(last[0], vertice_actual, data, weight)
        id += 1
        last = [vertice_actual, row['codigo']]
    return G


def show_shortest_path():
    pass


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def comprobar_direccion(direccion):
    global posibles_calles
    origen_posible = ''
    distancia_str_origen = 10e10
    for calle in posibles_calles:
        dist = levenshtein(calle.lower(), ' '.join(direccion[:-1]))
        if dist < distancia_str_origen:
            distancia_str_origen = dist
            origen_posible = calle
    return origen_posible
    
def comprobar_numero(numero, direccion):
    global cruces, c, d
    # Comprobar que existe el numero de origen
    codigo_direccion = cruces[cruces['Literal completo del vial tratado'] == direccion]['Codigo de vía tratado'].values[0]
    c_direccion = c[c['codigo'] == codigo_direccion].reset_index(drop=True)
    d_direccion = direcciones[direcciones['Nombre completo de calle'] == direccion].reset_index(drop=True)
    direccion = direccion + ' ' + numero
    posibles_numeros = set([int(x.split('-')[1]) for x in d_direccion['Direccion']])
    try:
        if int(numero) in posibles_numeros:
            return int(numero), c_direccion, d_direccion, codigo_direccion
    except:
        print('El numero de origen no es valido')
    return None, None, None, None


def main():
    global posibles_calles, cruces, direcciones, c, d
    # 1. Extraemos y limpiamos los datos para crear el grafo
    cruces, direcciones = extract()
    cruces, direcciones = clean(cruces, direcciones)
    cruces = unify_vertices(cruces)
    c, d = select_relevant_info(cruces, direcciones)
    c = assign_vertices(c, d)
    G_time = create_graph(cruces, c, 'tiempo')
    G_distance = create_graph(cruces, c, 'distancia')
    
    # 2. Interfaz para elegir calles
    end = False
    posibles_calles = direcciones['Nombre completo de calle'].unique()
    while not end:
        print('Introduzca las calles de la siguiente manera: Calle X')
        origen = input('Elige la dirección de origen: ').split(' ')
        if len(origen) == 0:
            end = True
            continue
        numero_origen = origen[-1]
        origen_posible = comprobar_direccion(origen)
        consultar_origen = input('¿Quieres salir de {}? (s/n)'.format(origen_posible))
        while consultar_origen not in ['s', 'n']:
            print('Entrada incorrecta')
            consultar_origen = input('¿Quieres salir de {}? (s/n)'.format(origen_posible))
        if consultar_origen == 'n':
            print('Introduce de nuevo la calle de origen')
            continue
        # Comprobar que existe el numero de origen
        numero_origen, c_origen, d_origen, codigo_origen = comprobar_numero(numero_origen, origen_posible)
        if numero_origen is None:
            print('Introduce de nuevo la calle de origen')
            continue

        destino = input('Elige la dirección de destino: ').split(' ')
        if len(destino) == 0:
            end = True
            continue
        numero_destino = destino[-1]
        destino_posible = comprobar_direccion(destino)
        consultar_destino = input('¿Quieres ir a {}? (s/n)'.format(destino_posible))
        while consultar_destino not in ['s', 'n']:
            print('Entrada incorrecta')
            consultar_destino = input('¿Quieres ir a {}? (s/n)'.format(destino_posible))
        if consultar_destino == 'n':
            print('Introduce de nuevo la calle de destino')
            continue
        # Comprobar que existe el numero de destino
        numero_destino, c_destino, d_destino, codigo_destino = comprobar_numero(numero_destino, destino_posible)
        if numero_destino is None:
            print('Introduce de nuevo la calle de destino')
            continue
        
        # Preguntamos que tipo de ruta queremos
        tipo_ruta = input('¿Quieres una ruta corta o rápida? (c/r)')
        while tipo_ruta not in ['c', 'r']:
            print('Entrada incorrecta')
            tipo_ruta = input('¿Quieres una ruta corta o rápida? (c/r)')
        if tipo_ruta == 'c':
            G = G_distance
        else:
            G = G_time

        # crear un nodo en el grafo con el origen conectándolo con los dos nodos más cercanos de su calle
        i1 = i2 = -1
        for i in c_origen.index:
            if c_origen.loc[i, 'numero'] > numero_origen:
                i1 = i - 1
                i2 = i
                break
        aristas = []
        if i1 == -1: 
            aristas.append((c_origen['x'][0], c_origen['y'][0]))
        elif i2 == -1:
            aristas.append((c_origen['x'][-1], c_origen['y'][-1]))
        else:
            aristas.append((c_origen['x'][i1], c_origen['y'][i1]))
            aristas.append((c_origen['x'][i2], c_origen['y'][i2]))
        
        
        # crear un nodo en el grafo con el destino conectándolo con los dos nodos más cercanos de su calle
        j1 = j2 = -1
        for j in c_destino.index:
            if c_destino.loc[j, 'numero'] > numero_destino:
                j1 = j - 1
                j2 = j
                break
        aristas = []
        if j1 == -1: 
            aristas.append((c_destino['x'][0], c_destino['y'][0]))
        elif j2 == -1:
            aristas.append((c_destino['x'][-1], c_destino['y'][-1]))
        else:
            aristas.append((c_destino['x'][j1], c_destino['y'][j1]))
            aristas.append((c_destino['x'][j2], c_destino['y'][j2]))
        print(aristas)
        # direccion_origen = d_origen(d_origen['Direccion'] == codigo_origen)
        # vertice_actual = gf.Vertice('O', [codigo_origen], (row['x'], row['y']))
        # G.agregar_vertice(vertice_actual)
        # nombre_calle = cruces[cruces['Codigo de vía tratado'] == codigo_origen]['Literal completo del vial tratado'].values[0]
        # data = {'codigo': codigo_origen, 'calle': nombre_calle}
        # for a in aristas:
        #     weight = get_weight(cruces[cruces['Codigo de vía tratado'] == codigo_origen]['Clase de la via tratado'].values[0], last[0].coordenadas, coords)
        #     G.agregar_arista(last[0], vertice_actual, data, weight)
        



    exit()


if __name__ == '__main__':
    main()
# 1) Al arrancar, leera los ficheros "cruces.csv" y "direcciones.csv" y construira el grafo de calles de la parte 2. Se crearan
# dos grafos: uno en el que el peso de cada arista sea la distancia euclidea entre los nodos y otro en el que el peso sea
# el tiempo que tarda un coche en recorrer dicha arista a la velocidad maxima permitida en dicha calle.
#       TERMINADO
#       
# 2) Permitira al usuario seleccionar dos direcciones (origen y destino) de la base de datos de direcciones ("direcciones.csv").
#       TERMINADO
# 
# 3) Permitira elegir al usuario si desea encontrar la ruta mas corta o mas rapida entre estos puntos.
#       TERMINADO
# 
# 4) Usando el grafo correspondiente en funcion de lo elegido en el punto (3), se calculara el camino minimo desde el
# origen al destino. Para ello, se deberan usar las funciones programadas en grafo.py.
#       TERMINADO
# 
# 5) La aplicacion analizara dicho camino y construira una lista de instrucciones detalladas que permitan a un automovil
# navegar desde el origen hasta el destino.
#       TERMINADO, angulo?
# 
# 6) Finalmente, usando NetworkX, se mostrara la ruta elegida resaltada sobre el grafo del callejero.
#       TERMINADO
# 
# 7) Tras mostrar la ruta, se volvera al punto 2 para seleccionar una nueva ruta hasta que se introduzca un origen o
# destino vacios.
#       PARTE DE LA INTERFAZ PARA PARAR EL BUCLE
# 
# Cuanta distancia (en metros) se debe continuar por cada via antes de tomar un giro hacia otra calle.
#       por ver
#       
# Al tomar un desvio, cual sera el nombre de la siguiente calle por la que se debera circular.
#       TERMINADO
# 
# A la hora de girar, si se debe girar a la izquierda o a la derecha. Opcionalmente, si hay un cruce multiple, se precisara
# por que salida debe continuarse.
#       jaja
# 
# El navegador no deberia dar instrucciones redundantes mientras se continue por la misma calle (mas alla de continuar
# por dicha calle X metros).
#       perfe
# 
# CUANDO CAMBIEMOS DE CALLE, NOS GUARDAMOS EN UN SET LOS NODOS DE DICHA CALLE. BUSCAMOS EN LOS NODOS QUE NOS QUEDAN
# POR RECORRER HASTA QUE ENCONTREMOS EL QUE NO ESTÁ EN EL SET. EN ESE CASO, CALCULAMOS LA DISTANCIA ENTRE EL
# NODO ACTUAL Y EL ÚLTIMO NODO QUE ESTÁ EN EL SET, Y TAMBIÉN CALCULAMOS SI HABRÁ QUE GIRAR A DERECHA O IZQUIERDA.