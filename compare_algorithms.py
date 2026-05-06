import json
import time
import random
import itertools
from math import dist
from statistics import mean


# =============================
# ASK USER FOR TSP FILE
# =============================

print("Available TSP datasets in benchmarks/json:")
print("Example names: berlin52.json, gr17.json, eil51.json")
filename = "eil51.json"

# Load the JSON file
with open(f"tsplib-json/benchmarks/json/{filename}") as f:
    tsp_data = json.load(f)

coord_dict = tsp_data["nodeCoordSection"]
cities_full = [coord_dict[str(i+1)] for i in range(len(coord_dict))]

MAX_CITIES = 18   # choose 12–18 maximum
cities = cities_full[:MAX_CITIES]

print(f"Using only first {MAX_CITIES} cities for exact TSP")


# =============================
# LOAD KNAPSACK DATA
# =============================

with open("sample_data.json") as f:
    data = json.load(f)

weights = data["knapsack"]["weights"]
values = data["knapsack"]["values"]
capacity = data["knapsack"]["capacity"]


# ==========================================
# DETERMINISTIC KNAPSACK (DP)
# ==========================================

def knapsack_dp(weights, values, capacity):
    n = len(weights)
    dp = [[0]*(capacity+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for w in range(capacity+1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    dp[i-1][w-weights[i-1]] + values[i-1]
                )
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][capacity]


# ==========================================
# DETERMINISTIC TSP (HELD-KARP)
# ==========================================

def tsp_exact(cities):
    n = len(cities)

    d = [[dist(cities[i], cities[j]) for j in range(n)]
         for i in range(n)]

    dp = {}

    for k in range(1, n):
        dp[(1 << k, k)] = (d[0][k], 0)

    for size in range(2, n):
        print("Processing subsets of size:", size)
        for subset in itertools.combinations(range(1, n), size):
            mask = sum(1 << k for k in subset)
            for k in subset:
                prev_mask = mask ^ (1 << k)

                best = min(
                    (dp[(prev_mask, m)][0] + d[m][k], m)
                    for m in subset if m != k
                )

                dp[(mask, k)] = best

    full_mask = (1 << n) - 2

    best_cost, parent = min(
        (dp[(full_mask, k)][0] + d[k][0], k)
        for k in range(1, n)
    )

    return best_cost


# ==========================================
# DETERMINISTIC GREEDY KNAPSACK
# ==========================================

def greedy_knapsack_deterministic(weights, values, capacity):
    n = len(weights)

    items = list(range(n))
    ratio = [(i, values[i] / weights[i]) for i in items]
    ratio.sort(key=lambda x: x[1], reverse=True)

    remaining = capacity
    total_value = 0

    for item, _ in ratio:
        if weights[item] <= remaining:
            remaining -= weights[item]
            total_value += values[item]

    return total_value


# ==========================================
# DETERMINISTIC GREEDY TSP (Nearest Neighbor)
# ==========================================

def greedy_tsp_deterministic(cities):
    n = len(cities)

    unvisited = list(range(1, n))
    current = 0
    total_dist = 0

    while unvisited:

        # Choose nearest city (no randomness)
        next_city = min(
            unvisited,
            key=lambda city: dist(cities[current], cities[city])
        )

        total_dist += dist(cities[current], cities[next_city])
        unvisited.remove(next_city)
        current = next_city

    total_dist += dist(cities[current], cities[0])

    return total_dist



# ==========================================
# RANDOM GREEDY KNAPSACK
# ==========================================

def greedy_knapsack_random(weights, values, capacity, k=3):
    n = len(weights)

    items = list(range(n))
    ratio = [(i, values[i]/weights[i]) for i in items]
    ratio.sort(key=lambda x: x[1], reverse=True)

    remaining = capacity
    total_value = 0

    while ratio and remaining > 0:

        candidates = ratio[:min(k, len(ratio))]
        item = random.choice(candidates)[0]

        ratio = [r for r in ratio if r[0] != item]

        if weights[item] <= remaining:
            remaining -= weights[item]
            total_value += values[item]

    return total_value


# ==========================================
# RANDOM GREEDY TSP
# ==========================================

def greedy_tsp_random(cities, k=3):
    n = len(cities)

    unvisited = list(range(1, n))
    current = 0
    total_dist = 0

    while unvisited:

        distances = [
            (city, dist(cities[current], cities[city]))
            for city in unvisited
        ]

        distances.sort(key=lambda x: x[1])
        candidates = distances[:min(k, len(distances))]

        next_city = random.choice(candidates)[0]

        total_dist += dist(cities[current], cities[next_city])
        unvisited.remove(next_city)
        current = next_city

    total_dist += dist(cities[current], cities[0])

    return total_dist


# =====================================================
# ============= LOCAL SEARCH HELPERS ==================
# =====================================================

def knapsack_total(solution, weights, values, capacity):
    """Evaluate a knapsack solution (list of 0/1). Returns value or -1 if infeasible."""
    total_w = sum(w for i, w in enumerate(weights) if solution[i])
    total_v = sum(v for i, v in enumerate(values)  if solution[i])
    return total_v if total_w <= capacity else -1

def knapsack_random_solution(n, weights, capacity):
    """Generate a random feasible knapsack solution."""
    sol = [0] * n
    indices = list(range(n))
    random.shuffle(indices)
    remaining = capacity
    for i in indices:
        if weights[i] <= remaining:
            sol[i] = 1
            remaining -= weights[i]
    return sol

def tsp_route_dist(route, cities):
    """Total distance of a TSP route (returns to start)."""
    total = 0
    n = len(route)
    for i in range(n):
        total += dist(cities[route[i]], cities[route[(i+1) % n]])
    return total

def tsp_random_solution(n):
    """Random TSP route starting from city 0."""
    route = list(range(1, n))
    random.shuffle(route)
    return [0] + route


# =====================================================
# ======= LOCAL SEARCH FIRST IMPROVEMENT =============
# =====================================================

# --- Knapsack: First Improvement ---
def ls_first_improvement_knapsack(weights, values, capacity, max_iter=1000):
    """
    Local Search - First Improvement for Knapsack.
    Neighborhood: flip one bit (add or remove one item).
    Accepts the first neighbor that improves the current solution.
    """
    n = len(weights)
    current = knapsack_random_solution(n, weights, capacity)
    current_val = knapsack_total(current, weights, values, capacity)

    for _ in range(max_iter):
        improved = False
        indices = list(range(n))
        random.shuffle(indices)  # randomize search order

        for i in indices:
            neighbor = current[:]
            neighbor[i] = 1 - neighbor[i]  # flip bit
            neighbor_val = knapsack_total(neighbor, weights, values, capacity)

            if neighbor_val > current_val:
                current = neighbor
                current_val = neighbor_val
                improved = True
                break  # first improvement: accept immediately

        if not improved:
            break  # local optimum reached

    return current_val


# --- TSP: First Improvement (2-opt) ---
def ls_first_improvement_tsp(cities, max_iter=1000):
    """
    Local Search - First Improvement for TSP using 2-opt moves.
    Accepts the first 2-opt swap that improves the current tour.
    """
    n = len(cities)
    route = tsp_random_solution(n)
    current_dist = tsp_route_dist(route, cities)

    for _ in range(max_iter):
        improved = False

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # 2-opt swap: reverse segment between i and j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_dist = tsp_route_dist(new_route, cities)

                if new_dist < current_dist:
                    route = new_route
                    current_dist = new_dist
                    improved = True
                    break  # first improvement: accept immediately

            if improved:
                break

        if not improved:
            break

    return current_dist


# =====================================================
# ======= LOCAL SEARCH BEST IMPROVEMENT ==============
# =====================================================

# --- Knapsack: Best Improvement ---
def ls_best_improvement_knapsack(weights, values, capacity, max_iter=1000):
    """
    Local Search - Best Improvement for Knapsack.
    Neighborhood: flip one bit (add or remove one item).
    Scans ALL neighbors and accepts the best improving one.
    """
    n = len(weights)
    current = knapsack_random_solution(n, weights, capacity)
    current_val = knapsack_total(current, weights, values, capacity)

    for _ in range(max_iter):
        best_neighbor = None
        best_val = current_val

        for i in range(n):
            neighbor = current[:]
            neighbor[i] = 1 - neighbor[i]
            neighbor_val = knapsack_total(neighbor, weights, values, capacity)

            if neighbor_val > best_val:
                best_val = neighbor_val
                best_neighbor = neighbor

        if best_neighbor is None:
            break  # local optimum reached

        current = best_neighbor
        current_val = best_val

    return current_val


# --- TSP: Best Improvement (2-opt) ---
def ls_best_improvement_tsp(cities, max_iter=1000):
    """
    Local Search - Best Improvement for TSP using 2-opt moves.
    Scans ALL 2-opt swaps and applies the best one per iteration.
    """
    n = len(cities)
    route = tsp_random_solution(n)
    current_dist = tsp_route_dist(route, cities)

    for _ in range(max_iter):
        best_route = None
        best_dist = current_dist

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_dist = tsp_route_dist(new_route, cities)

                if new_dist < best_dist:
                    best_dist = new_dist
                    best_route = new_route

        if best_route is None:
            break  # local optimum reached

        route = best_route
        current_dist = best_dist

    return current_dist


# =====================================================
# =========== SIMULATED ANNEALING ====================
# =====================================================

import math

# --- Knapsack: Simulated Annealing ---
def simulated_annealing_knapsack(weights, values, capacity,
                                  T=1000.0, alpha=0.995, min_T=0.1, max_iter=5000):
    """
    Simulated Annealing for Knapsack.
    Neighborhood: flip one random bit.
    Accepts worse solutions with probability exp(delta/T).
    """
    n = len(weights)
    current = knapsack_random_solution(n, weights, capacity)
    current_val = knapsack_total(current, weights, values, capacity)
    if current_val < 0:
        current_val = 0

    best = current[:]
    best_val = current_val

    temp = T

    for _ in range(max_iter):
        if temp < min_T:
            break

        # Generate neighbor: flip one random bit
        i = random.randint(0, n - 1)
        neighbor = current[:]
        neighbor[i] = 1 - neighbor[i]
        neighbor_val = knapsack_total(neighbor, weights, values, capacity)

        if neighbor_val < 0:
            neighbor_val = 0  # infeasible — treat as 0

        delta = neighbor_val - current_val

        # Accept if better, or with probability exp(delta/T) if worse
        if delta > 0 or random.random() < math.exp(delta / temp):
            current = neighbor
            current_val = neighbor_val

        if current_val > best_val:
            best = current[:]
            best_val = current_val

        temp *= alpha  # cool down

    return best_val


# --- TSP: Simulated Annealing ---
def simulated_annealing_tsp(cities,
                              T=10000.0, alpha=0.995, min_T=0.1, max_iter=10000):
    """
    Simulated Annealing for TSP.
    Neighborhood: random 2-opt swap.
    Accepts worse solutions with probability exp(-delta/T).
    """
    n = len(cities)
    route = tsp_random_solution(n)
    current_dist = tsp_route_dist(route, cities)

    best_route = route[:]
    best_dist = current_dist

    temp = T

    for _ in range(max_iter):
        if temp < min_T:
            break

        # Random 2-opt swap
        i = random.randint(1, n - 2)
        j = random.randint(i + 1, n - 1)
        neighbor = route[:i] + route[i:j+1][::-1] + route[j+1:]
        neighbor_dist = tsp_route_dist(neighbor, cities)

        delta = neighbor_dist - current_dist

        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or random.random() < math.exp(-delta / temp):
            route = neighbor
            current_dist = neighbor_dist

        if current_dist < best_dist:
            best_route = route[:]
            best_dist = current_dist

        temp *= alpha

    return best_dist


# =====================================================
# ============= GENETIC ALGORITHM ====================
# =====================================================



# ---- Knapsack GA ----

import random

def ga_knapsack_init_population(pop_size, n, weights, capacity):
    """Generate initial population of feasible knapsack solutions."""
    return [knapsack_random_solution(n, weights, capacity) for _ in range(pop_size)]

def ga_knapsack_fitness(sol, weights, values, capacity):
    """Fitness = total value if feasible, else 0."""
    v = knapsack_total(sol, weights, values, capacity)
    return v if v >= 0 else 0


# -------------------- ROULETTE SELECTION --------------------

def ga_knapsack_selection_ROULETTE(population, fitnesses):
    """Roulette wheel selection."""
    pop = population.copy()
    fits = fitnesses.copy()

    total_fit = sum(fits)

    # If all fitnesses are zero, pick random (avoid division by zero)
    if total_fit == 0:
        return random.choice(pop)

    probs = [f / total_fit for f in fits]

    # cumulative probabilities
    cumulative = []
    csum = 0
    for p in probs:
        csum += p
        cumulative.append(csum)

    r = random.random()

    for idx, c in enumerate(cumulative):
        if r <= c:
            return pop[idx]


def ga_knapsack_crossover(parent1, parent2):
    """Single-point crossover."""
    n = len(parent1)
    point = random.randint(1, n - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def ga_knapsack_mutate(sol, mutation_rate=0.05):
    """Bit-flip mutation."""
    return [1 - bit if random.random() < mutation_rate else bit for bit in sol]


def ga_knapsack_repair(sol, weights, capacity):
    """Remove random items until solution is feasible."""
    sol = sol[:]
    while sum(w for i, w in enumerate(weights) if sol[i]) > capacity:
        ones = [i for i, b in enumerate(sol) if b == 1]
        if not ones:
            break
        sol[random.choice(ones)] = 0
    return sol


def genetic_algorithm_knapsack(weights, values, capacity,
                                 pop_size=50, generations=100,
                                 mutation_rate=0.05, elite_size=5):
    """
    Genetic Algorithm for Knapsack.
    Population: 50 random feasible solutions.
    Selection: roulette wheel.
    Crossover: single-point.
    Mutation: bit-flip.
    Elitism: top elite_size individuals survive each generation.
    """
    n = len(weights)
    population = ga_knapsack_init_population(pop_size, n, weights, capacity)

    best_val = 0

    for gen in range(generations):
        fitnesses = [ga_knapsack_fitness(sol, weights, values, capacity)
                     for sol in population]

        # Track best
        gen_best = max(fitnesses)
        if gen_best > best_val:
            best_val = gen_best

        # Elitism: carry top individuals unchanged
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        new_population = [population[i][:] for i in elite_indices]

        # Fill rest with crossover + mutation children
        while len(new_population) < pop_size:
            p1 = ga_knapsack_selection_ROULETTE(population, fitnesses)
            p2 = ga_knapsack_selection_ROULETTE(population, fitnesses)

            child1, child2 = ga_knapsack_crossover(p1, p2)

            child1 = ga_knapsack_mutate(child1, mutation_rate)
            child2 = ga_knapsack_mutate(child2, mutation_rate)

            child1 = ga_knapsack_repair(child1, weights, capacity)
            child2 = ga_knapsack_repair(child2, weights, capacity)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    # Return best fitness in final population
    fitnesses = [ga_knapsack_fitness(sol, weights, values, capacity) for sol in population]
    return max(fitnesses)


# ---- TSP GA ----

import random

def ga_tsp_init_population(pop_size, n):
    """Generate initial population of random TSP routes."""
    return [tsp_random_solution(n) for _ in range(pop_size)]

def ga_tsp_fitness(route, cities):
    """Fitness = negative distance (we maximize fitness, minimize distance)."""
    return -tsp_route_dist(route, cities)


# -------------------- ROULETTE SELECTION --------------------

def ga_tsp_selection(population, fitnesses):
    """Roulette wheel selection with recalculation."""
    pop = population.copy()
    fits = fitnesses.copy()

    # Convert fitness to positive (higher is better)
    min_fit = min(fits)
    shifted = [f - min_fit + 1e-6 for f in fits]

    total = sum(shifted)
    probs = [f / total for f in shifted]

    # cumulative probabilities
    cumulative = []
    csum = 0
    for p in probs:
        csum += p
        cumulative.append(csum)

    r = random.random()

    for idx, c in enumerate(cumulative):
        if r <= c:
            return pop[idx]


# -------------------- REPAIR (ORDERING) --------------------

def ga_tsp_repair(child, parent):
    """
    Repair a child with duplicates using ordering from parent.
    """
    n = len(child)
    seen = set()
    missing = [city for city in range(n) if city not in child]

    repaired = child[:]

    for i in range(1, n):  # skip city 0
        if repaired[i] in seen:
            # Replace duplicate using parent ordering
            for p_city in parent:
                if p_city in missing:
                    repaired[i] = p_city
                    missing.remove(p_city)
                    break
        else:
            seen.add(repaired[i])

    return repaired


# -------------------- SAFE OX --------------------

def ga_tsp_crossover(parent1, parent2):
    """
    Order Crossover (OX) for TSP.
    Preserves relative order of cities from both parents.
    """
    n = len(parent1)
    start, end = sorted(random.sample(range(1, n), 2))

    child = [None] * n
    child[0] = 0  # always start from city 0
    child[start:end+1] = parent1[start:end+1]

    # Fill remaining positions from parent2 in order
    remaining = [city for city in parent2 if city not in child]
    pos = 0
    for i in range(1, n):
        if child[i] is None:
            child[i] = remaining[pos]
            pos += 1

    # Repair if anything went wrong
    if len(set(child)) != n:
        child = ga_tsp_repair(child, parent1)

    return child


def ga_tsp_mutate_swap(route, mutation_rate=0.1):
    """Swap mutation: randomly swap two cities in the route."""
    route = route[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route


def ga_tsp_mutate_inversion(route, mutation_rate=0.1):
    """Inversion mutation: reverse a random sub-segment."""
    route = route[:]
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(1, len(route)), 2))
        route[i:j+1] = route[i:j+1][::-1]
    return route


def genetic_algorithm_tsp(cities,
                            pop_size=50, generations=200,
                            mutation_rate=0.1, elite_size=5):
    """
    Genetic Algorithm for TSP.
    Population: 50 random routes.
    Selection: roulette wheel.
    Crossover: Order Crossover (OX) with repair.
    Mutation: inversion mutation.
    Elitism: top elite_size individuals survive each generation.
    """
    n = len(cities)
    population = ga_tsp_init_population(pop_size, n)

    best_dist = float('inf')

    for gen in range(generations):
        fitnesses = [ga_tsp_fitness(route, cities) for route in population]

        # Track best
        gen_best_dist = -max(fitnesses)
        if gen_best_dist < best_dist:
            best_dist = gen_best_dist

        # Elitism
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        new_population = [population[i][:] for i in elite_indices]

        # Fill rest with crossover + mutation children
        while len(new_population) < pop_size:
            p1 = ga_tsp_selection(population, fitnesses)
            p2 = ga_tsp_selection(population, fitnesses)

            child1 = ga_tsp_crossover(p1, p2)
            child2 = ga_tsp_crossover(p2, p1)

            child1 = ga_tsp_mutate_inversion(child1, mutation_rate)
            child2 = ga_tsp_mutate_inversion(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    fitnesses = [ga_tsp_fitness(route, cities) for route in population]
    return -max(fitnesses)


# =============================
# RUN COMPARISONS
# =============================

RUNS = 20


# =====================================================
# ==================== KNAPSACK =======================
# =====================================================

print("\n" + "="*60)
print("KNAPSACK COMPARISON")
print("="*60)

# ---------- Exact DP ----------
print("\n[1] Exact Dynamic Programming")

start = time.perf_counter()
optimal_value = knapsack_dp(weights, values, capacity)
optimal_time = time.perf_counter() - start

print(f"Optimal value: {optimal_value}")
print(f"Execution time: {optimal_time:.6f}s")


# ---------- Deterministic Greedy ----------
print("\n[2] Deterministic Greedy (Best ratio first)")

start = time.perf_counter()
det_greedy_value = greedy_knapsack_deterministic(weights, values, capacity)
det_greedy_time = time.perf_counter() - start

similarity = (det_greedy_value / optimal_value) * 100

print(f"Greedy value: {det_greedy_value}")
print(f"Similarity to optimal: {similarity:.2f}%")
print(f"Execution time: {det_greedy_time:.6f}s")


# ---------- Random Greedy ----------
print(f"\n[3] Random Greedy (Top-k, {RUNS} runs)")

greedy_values = []
greedy_times = []
similarities = []

for _ in range(RUNS):
    start = time.perf_counter()
    val = greedy_knapsack_random(weights, values, capacity)
    elapsed = time.perf_counter() - start

    greedy_values.append(val)
    greedy_times.append(elapsed)
    similarities.append((val / optimal_value) * 100)

print(f"Average value: {mean(greedy_values):.2f}")
print(f"Best value: {max(greedy_values)}")
print(f"Average similarity: {mean(similarities):.2f}%")
print(f"Best similarity: {max(similarities):.2f}%")
print(f"Average execution time: {mean(greedy_times):.6f}s")


# ---------- Local Search First Improvement ----------
print(f"\n[4] Local Search - First Improvement ({RUNS} runs)")

ls_fi_values = []
ls_fi_times = []
ls_fi_similarities = []

for _ in range(RUNS):
    start = time.perf_counter()
    val = ls_first_improvement_knapsack(weights, values, capacity)
    elapsed = time.perf_counter() - start

    ls_fi_values.append(val)
    ls_fi_times.append(elapsed)
    ls_fi_similarities.append((val / optimal_value) * 100)

print(f"Average value: {mean(ls_fi_values):.2f}")
print(f"Best value: {max(ls_fi_values)}")
print(f"Average similarity: {mean(ls_fi_similarities):.2f}%")
print(f"Best similarity: {max(ls_fi_similarities):.2f}%")
print(f"Average execution time: {mean(ls_fi_times):.6f}s")


# ---------- Local Search Best Improvement ----------
print(f"\n[5] Local Search - Best Improvement ({RUNS} runs)")

ls_bi_values = []
ls_bi_times = []
ls_bi_similarities = []

for _ in range(RUNS):
    start = time.perf_counter()
    val = ls_best_improvement_knapsack(weights, values, capacity)
    elapsed = time.perf_counter() - start

    ls_bi_values.append(val)
    ls_bi_times.append(elapsed)
    ls_bi_similarities.append((val / optimal_value) * 100)

print(f"Average value: {mean(ls_bi_values):.2f}")
print(f"Best value: {max(ls_bi_values)}")
print(f"Average similarity: {mean(ls_bi_similarities):.2f}%")
print(f"Best similarity: {max(ls_bi_similarities):.2f}%")
print(f"Average execution time: {mean(ls_bi_times):.6f}s")


# ---------- Simulated Annealing ----------
print(f"\n[6] Simulated Annealing ({RUNS} runs)")

sa_values = []
sa_times = []
sa_similarities = []

for _ in range(RUNS):
    start = time.perf_counter()
    val = simulated_annealing_knapsack(weights, values, capacity)
    elapsed = time.perf_counter() - start

    sa_values.append(val)
    sa_times.append(elapsed)
    sa_similarities.append((val / optimal_value) * 100)

print(f"Average value: {mean(sa_values):.2f}")
print(f"Best value: {max(sa_values)}")
print(f"Average similarity: {mean(sa_similarities):.2f}%")
print(f"Best similarity: {max(sa_similarities):.2f}%")
print(f"Average execution time: {mean(sa_times):.6f}s")


# ---------- Genetic Algorithm ----------
print(f"\n[7] Genetic Algorithm (pop=50, {2} runs)")

ga_values = []
ga_times = []
ga_similarities = []

for _ in range(2):
    start = time.perf_counter()
    val = genetic_algorithm_knapsack(weights, values, capacity)
    elapsed = time.perf_counter() - start

    ga_values.append(val)
    ga_times.append(elapsed)
    ga_similarities.append((val / optimal_value) * 100)

print(f"Average value: {mean(ga_values):.2f}")
print(f"Best value: {max(ga_values)}")
print(f"Average similarity: {mean(ga_similarities):.2f}%")
print(f"Best similarity: {max(ga_similarities):.2f}%")
print(f"Average execution time: {mean(ga_times):.6f}s")


# =====================================================
# ====================== TSP ==========================
# =====================================================

print("\n" + "="*60)
print("TSP COMPARISON")
print("="*60)

# ---------- Exact Held-Karp ----------
print("\n[1] Exact Held-Karp")

start = time.perf_counter()
optimal_dist = tsp_exact(cities)
optimal_time = time.perf_counter() - start

print(f"Optimal distance: {optimal_dist}")
print(f"Execution time: {optimal_time:.6f}s")


# ---------- Deterministic Greedy ----------
print("\n[2] Deterministic Greedy (Nearest Neighbor)")

start = time.perf_counter()
det_greedy_dist = greedy_tsp_deterministic(cities)
det_greedy_time = time.perf_counter() - start

similarity = (optimal_dist / det_greedy_dist) * 100

print(f"Greedy distance: {det_greedy_dist}")
print(f"Similarity to optimal: {similarity:.2f}%")
print(f"Execution time: {det_greedy_time:.6f}s")


# ---------- Random Greedy ----------
print(f"\n[3] Random Greedy (Top-k, {RUNS} runs)")

greedy_dists = []
greedy_times = []
similarities = []

for _ in range(RUNS):
    start = time.perf_counter()
    d_val = greedy_tsp_random(cities)
    elapsed = time.perf_counter() - start

    greedy_dists.append(d_val)
    greedy_times.append(elapsed)
    similarities.append((optimal_dist / d_val) * 100)

print(f"Average distance: {mean(greedy_dists):.2f}")
print(f"Best distance: {min(greedy_dists):.2f}")
print(f"Average similarity: {mean(similarities):.2f}%")
print(f"Best similarity: {max(similarities):.2f}%")
print(f"Average execution time: {mean(greedy_times):.6f}s")


# ---------- Local Search First Improvement ----------
print(f"\n[4] Local Search - First Improvement ({RUNS} runs)")

ls_fi_dists = []
ls_fi_times = []
ls_fi_similarities = []

for _ in range(RUNS):
    start = time.perf_counter()
    d_val = ls_first_improvement_tsp(cities)
    elapsed = time.perf_counter() - start

    ls_fi_dists.append(d_val)
    ls_fi_times.append(elapsed)
    ls_fi_similarities.append((optimal_dist / d_val) * 100)

print(f"Average distance: {mean(ls_fi_dists):.2f}")
print(f"Best distance: {min(ls_fi_dists):.2f}")
print(f"Average similarity: {mean(ls_fi_similarities):.2f}%")
print(f"Best similarity: {max(ls_fi_similarities):.2f}%")
print(f"Average execution time: {mean(ls_fi_times):.6f}s")


# ---------- Local Search Best Improvement ----------
print(f"\n[5] Local Search - Best Improvement ({RUNS} runs)")

ls_bi_dists = []
ls_bi_times = []
ls_bi_similarities = []

for _ in range(RUNS):
    start = time.perf_counter()
    d_val = ls_best_improvement_tsp(cities)
    elapsed = time.perf_counter() - start

    ls_bi_dists.append(d_val)
    ls_bi_times.append(elapsed)
    ls_bi_similarities.append((optimal_dist / d_val) * 100)

print(f"Average distance: {mean(ls_bi_dists):.2f}")
print(f"Best distance: {min(ls_bi_dists):.2f}")
print(f"Average similarity: {mean(ls_bi_similarities):.2f}%")
print(f"Best similarity: {max(ls_bi_similarities):.2f}%")
print(f"Average execution time: {mean(ls_bi_times):.6f}s")


# ---------- Simulated Annealing ----------
print(f"\n[6] Simulated Annealing ({RUNS} runs)")

sa_dists = []
sa_times = []
sa_similarities = []

for _ in range(RUNS):
    start = time.perf_counter()
    d_val = simulated_annealing_tsp(cities)
    elapsed = time.perf_counter() - start

    sa_dists.append(d_val)
    sa_times.append(elapsed)
    sa_similarities.append((optimal_dist / d_val) * 100)

print(f"Average distance: {mean(sa_dists):.2f}")
print(f"Best distance: {min(sa_dists):.2f}")
print(f"Average similarity: {mean(sa_similarities):.2f}%")
print(f"Best similarity: {max(sa_similarities):.2f}%")
print(f"Average execution time: {mean(sa_times):.6f}s")


# ---------- Genetic Algorithm ----------
print(f"\n[7] Genetic Algorithm (pop=50, {2} runs)")

ga_dists = []
ga_times = []
ga_similarities = []

for _ in range(2):
    start = time.perf_counter()
    d_val = genetic_algorithm_tsp(cities)
    elapsed = time.perf_counter() - start

    ga_dists.append(d_val)
    ga_times.append(elapsed)
    ga_similarities.append((optimal_dist / d_val) * 100)

print(f"Average distance: {mean(ga_dists):.2f}")
print(f"Best distance: {min(ga_dists):.2f}")
print(f"Average similarity: {mean(ga_similarities):.2f}%")
print(f"Best similarity: {max(ga_similarities):.2f}%")
print(f"Average execution time: {mean(ga_times):.6f}s")


print("teacher , i used 2 runs in genetic algorithm because it takes a long time to execute")