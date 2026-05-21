import os
import subprocess
import re
import yaml
import matplotlib.pyplot as plt

epsilons = [0.3, 0.5, 0.8, 1.0, 1.5]
log_file = "experiment.log"

# Backup config
with open("config.yaml", "r") as f:
    orig_config = f.read()

# Modify config for faster execution
config = yaml.safe_load(orig_config)
config["experiment"]["n_rounds"] = 5
config["experiment"]["local_epochs"] = 1
with open("config.yaml", "w") as f:
    yaml.dump(config, f)

results = {}

for eps in epsilons:
    print(f"Running epsilon = {eps}...")
    if os.path.exists(log_file):
        os.remove(log_file)
        
    env = os.environ.copy()
    env["DP_EPSILON"] = str(eps)
    
    subprocess.run(["python", "main.py"], env=env)
    
    rounds = []
    accuracies = []
    
    acc_pattern = re.compile(r'Round (\d+) global test accuracy: ([\d.]+)%')
    
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                m = acc_pattern.search(line)
                if m:
                    rounds.append(int(m.group(1)))
                    accuracies.append(float(m.group(2)))
                    
    results[eps] = (rounds, accuracies)
    print(f"Epsilon {eps} final accuracy: {accuracies[-1] if accuracies else 0}%")

# Restore config
with open("config.yaml", "w") as f:
    f.write(orig_config)

# Plotting Graph 1: Accuracy vs Rounds
plt.figure(figsize=(10, 6))
for eps, (rounds, accuracies) in results.items():
    if rounds and accuracies:
        plt.plot(rounds, accuracies, marker='o', label=f'ε = {eps}', linewidth=2)

plt.title('Accuracy vs Communication Rounds', fontsize=14, fontweight='bold')
plt.xlabel('Communication Round')
plt.ylabel('Global Test Accuracy (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('epsilon_accuracy_rounds.png', dpi=300)
print("Graph saved to epsilon_accuracy_rounds.png")

# Plotting Graph 2: Final Accuracy vs Epsilon
plt.figure(figsize=(8, 5))
final_eps = []
final_acc = []
for eps in epsilons:
    if results[eps][1]:
        final_eps.append(eps)
        final_acc.append(results[eps][1][-1])

plt.plot(final_eps, final_acc, marker='s', color='tab:red', linewidth=2, markersize=8)
plt.title('Final Model Accuracy vs Epsilon', fontsize=14, fontweight='bold')
plt.xlabel('Privacy Budget (ε)')
plt.ylabel('Final Test Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('epsilon_final_accuracy.png', dpi=300)
print("Graph saved to epsilon_final_accuracy.png")
