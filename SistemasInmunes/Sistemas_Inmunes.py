import random

def generar_poblacion(tamano_poblacion, longitud_cadena):
    """Genera una población inicial de cadenas de bits."""
    return [''.join(random.choice('01') for _ in range(longitud_cadena)) for _ in range(tamano_poblacion)]

def evaluar_aptitud(individuo):
    """Evalúa la aptitud de un individuo como la cantidad de unos en la cadena."""
    return sum(1 for bit in individuo if bit == '1')

def seleccionar_padres(poblacion):
    """Selecciona dos padres al azar de la población."""
    padres_indices = random.sample(range(len(poblacion)), 2)
    return poblacion[padres_indices[0]], poblacion[padres_indices[1]]

def cruzar(padre1, padre2):
    """Realiza el cruce (crossover) de dos padres para producir un descendiente."""
    punto_corte = random.randint(1, len(padre1) - 1)
    descendiente = padre1[:punto_corte] + padre2[punto_corte:]
    return descendiente

def mutar(individuo, tasa_mutacion):
    """Realiza la mutación de un individuo con una cierta tasa."""
    return ''.join(bit if random.random() > tasa_mutacion else '1' if bit == '0' else '0' for bit in individuo)

def algoritmo_inmunologico(tamano_poblacion, longitud_cadena, tasa_mutacion, generaciones):
    """Implementa un algoritmo inmunológico simple."""
    poblacion = generar_poblacion(tamano_poblacion, longitud_cadena)

    for generacion in range(generaciones):
        poblacion = sorted(poblacion, key=evaluar_aptitud, reverse=True)

        mejor_individuo = poblacion[0]
        print(f"Generación {generacion + 1}: Mejor individuo - {mejor_individuo}, Aptitud - {evaluar_aptitud(mejor_individuo)}")

        if evaluar_aptitud(mejor_individuo) == longitud_cadena:
            print("Objetivo alcanzado.")
            break

        nueva_poblacion = []

        for _ in range(tamano_poblacion // 2):
            padre1, padre2 = seleccionar_padres(poblacion)
            descendiente = cruzar(padre1, padre2)
            descendiente_mutado = mutar(descendiente, tasa_mutacion)
            nueva_poblacion.extend([descendiente, descendiente_mutado])

        poblacion = nueva_poblacion

# Parámetros del algoritmo
tamano_poblacion = 10
longitud_cadena = 8
tasa_mutacion = 0.1
generaciones = 50

# Ejecutar el algoritmo inmunológico
algoritmo_inmunologico(tamano_poblacion, longitud_cadena, tasa_mutacion, generaciones)
