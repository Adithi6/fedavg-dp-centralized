import math

def calculate_cumulative_epsilon(noise_multiplier, num_rounds, delta):
    alphas = [1.1, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    best_epsilon = float('inf')
    for alpha in alphas:
        rdp_total = (alpha / (2 * (noise_multiplier ** 2))) * num_rounds
        epsilon = rdp_total + (math.log(1.0 / delta) / (alpha - 1))
        if epsilon < best_epsilon:
            best_epsilon = epsilon
    return best_epsilon

for nm in [10.0, 15.0, 20.0, 25.0]:
    print(f"nm={nm}, rounds=5 -> epsilon={calculate_cumulative_epsilon(nm, 5, 1e-5)}")
