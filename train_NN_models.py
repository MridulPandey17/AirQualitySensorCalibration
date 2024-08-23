from multiprocessing import Pool
import subprocess

def run_script(input_value):
    # Example of calling an external script with subprocess
    command = ["python", "NN_unsupervised_all_sensors.py"] + list(str(input_value))
    subprocess.run(command)

if __name__ == '__main__':
    inputs = [11,13,15,18,21,25]  # Example sets of inputs

    with Pool() as p:
        p.map(run_script, inputs)