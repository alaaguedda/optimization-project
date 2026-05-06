import json
import random

def generate_knapsack(n_items=1500, seed=42):
    random.seed(seed)
    weights = [random.randint(200, 1200) for _ in range(n_items)]
    values = [random.randint(10, 1000) for _ in range(n_items)]
    capacity = 1400

    return {
        "weights": weights,
        "values": values,
        "capacity": capacity
    }


def generate_tsp(n_cities=9, seed=42):
    random.seed(seed)
    cities = [(random.uniform(0, 100), random.uniform(0, 100))
              for _ in range(n_cities)]

    return {
        "cities": cities
    }


def main():
    data = {
        "knapsack": generate_knapsack(),
        "tsp": generate_tsp()
    }

    with open("sample_data.json", "w") as f:
        json.dump(data, f, indent=4)

    print("✅ Sample data generated: sample_data.json")


if __name__ == "__main__":
    main()