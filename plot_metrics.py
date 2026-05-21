import re
import matplotlib.pyplot as plt
from datetime import datetime

log_file = 'experiment.log'

rounds = []
accuracies = []
times = []
cumulative_times = []

# Regex patterns
round_start_pattern = re.compile(r'^(.*?) \| INFO \| Round (\d+)/\d+$')
acc_pattern = re.compile(r'^(.*?) \| INFO \| Round (\d+) global test accuracy: ([\d.]+)%$')

start_time = None
round_start_time = None

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        # Check for round start
        m_start = round_start_pattern.match(line)
        if m_start:
            timestamp_str = m_start.group(1).strip()
            r = int(m_start.group(2))
            
            # If we hit round 1, reset the data for the new run
            if r == 1:
                rounds = []
                accuracies = []
                times = []
                cumulative_times = []
                start_time = None
                
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            if start_time is None:
                start_time = dt
            round_start_time = dt

        # Check for accuracy (end of round)
        m_acc = acc_pattern.match(line)
        if m_acc:
            timestamp_str = m_acc.group(1).strip()
            r = int(m_acc.group(2))
            acc = float(m_acc.group(3))
            
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            
            rounds.append(r)
            accuracies.append(acc)
            
            if round_start_time:
                cost = (dt - round_start_time).total_seconds()
            else:
                cost = 0
            times.append(cost)
            
            if start_time:
                cum_cost = (dt - start_time).total_seconds()
            else:
                cum_cost = 0
            cumulative_times.append(cum_cost)

print(f"Parsed {len(rounds)} rounds from the final run.")
for r, a, c, ct in zip(rounds, accuracies, times, cumulative_times):
    print(f"Round {r}: Acc {a}%, Time {c}s, Cum. Time {ct}s")

if rounds:
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)', color=color, fontweight='bold')
    ax1.plot(rounds, accuracies, marker='o', color=color, label='Accuracy', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Computation Time (s)', color=color, fontweight='bold')  
    ax2.plot(rounds, cumulative_times, marker='s', color=color, linestyle='--', label='Computation Cost', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title('Global Model Accuracy and Computation Cost vs. Communication Rounds', fontsize=14, fontweight='bold')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    # Save the plot
    plt.savefig('metrics_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to metrics_plot.png")
