import subprocess


script_path = "run.py"
param_name, param_values = "conservative_weight", [1.0, 5.0, 10.0]

for value in param_values:
    print(f"Calling script {script_path} with {param_name}={value}")
    subprocess.call(['python', script_path, f"--{param_name}", str(value)])