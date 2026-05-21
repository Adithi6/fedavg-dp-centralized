import os
import subprocess
import re
import math

epsilons = [0.3, 0.5, 0.8, 1.0]
clip_norm = 1.0
log_file = 'experiment.log'

results = []

for idx, eps in enumerate(epsilons):
    print(f"Running trial {idx+1} for epsilon={eps}...")
    
    # Clear log file before run to ensure clean parse
    if os.path.exists(log_file):
        os.remove(log_file)
        
    env = os.environ.copy()
    env["DP_EPSILON"] = str(eps)
    env["DP_CLIP_NORM"] = str(clip_norm)
    
    # Run main.py
    subprocess.run(["python", "main.py"], env=env)
    
    # Parse log file
    noise_std = 0.0
    cum_eps = 0.0
    final_acc = 0.0
    total_time = 0.0
    
    noise_pattern = re.compile(r'noise_multiplier=([\d.]+)')
    cum_eps_pattern = re.compile(r'Round 5 Cumulative Privacy Spent: Epsilon = ([\d.]+)')
    acc_pattern = re.compile(r'Round 5 global test accuracy: ([\d.]+)%')
    time_pattern = re.compile(r'Total time = ([\d.]+)s')
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m_noise = noise_pattern.search(line)
                if m_noise:
                    noise_std = float(m_noise.group(1))
                    
                m_eps = cum_eps_pattern.search(line)
                if m_eps:
                    cum_eps = float(m_eps.group(1))
                    
                m_acc = acc_pattern.search(line)
                if m_acc:
                    final_acc = float(m_acc.group(1))
                    
                m_time = time_pattern.search(line)
                if m_time:
                    total_time = float(m_time.group(1))
                    
    results.append({
        'trial': idx + 1,
        'target_eps': eps,
        'noise_std': noise_std,
        'cum_eps': cum_eps,
        'final_acc': final_acc,
        'total_time': total_time
    })
    
print("Trials completed. Generating LaTeX Table...\n")

latex_table = r"""\begin{frame}{Differential Privacy Epsilon Tuning}

\scriptsize
\begin{center}
\textbf{Table 1: Epsilon Trial-and-Error on Approach 1}
\end{center}

\vspace{0.1cm}

\resizebox{\textwidth}{!}{
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{Trial} & 
\textbf{Target $\epsilon$} & 
\textbf{$\delta$} & 
\textbf{Clip Norm $C$} & 
\textbf{Clients} & 
\textbf{Rounds} & 
\textbf{Noise Std $\sigma$} & 
\textbf{Cumulative $\epsilon$ after 5 Rounds} & 
\textbf{Final Accuracy} & 
\textbf{Total Time} \\
\hline
"""

for res in results:
    latex_table += f"""
{res['trial']} & {res['target_eps']} & $1 \\times 10^{{-5}}$ & {clip_norm} & 10 & 5 & {res['noise_std']:.4f} & {res['cum_eps']:.4f} & {res['final_acc']:.2f}\\% & {res['total_time']:.2f} s \\\\
\\hline
"""

latex_table += r"""
\end{tabular}
}

\vspace{0.2cm}

\scriptsize
\textbf{Observation:} Lower target $\epsilon$ provides stronger privacy but adds higher Gaussian noise, which reduced model accuracy in the initial DP-FedAvg trials.


\vspace{0.1cm}
\scriptsize
\textbf{Next Phase:} After the initial epsilon tuning, the learning model was upgraded to a LeNet-style CNN, which improved the accuracy.


\end{frame}
"""

with open("latex_output.txt", "w") as f:
    f.write(latex_table)

print(latex_table)
