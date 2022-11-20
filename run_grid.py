import subprocess


script_path = "run_crr.py"
additional_arguments = [
    "--data",
    "data/rb_1000_20221118T0816.pkl"
]
param_name, param_values = "actor_lr", [1e-4, 3e-4, 5e-4, 1e-5, 3e-5, 5e-5]

for value in param_values:
    print(f"Calling script {script_path} with {param_name}={value}")
    to_call = ['python', script_path, f"--{param_name}", str(value)]
    to_call.extend(additional_arguments)
    subprocess.call(to_call)
    