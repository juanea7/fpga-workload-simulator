"""
FPGA Workload Manager Simulation

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2024
Description : This script simulates the execution of a workload using the FPGA Workload Manager.

"""

import pickle
import random
import sys
import argparse

from incremental_learning import online_models as om
from workloads import generate_workload_from_files
from simulator import WorkloadSimulator


# Parse arguments
parser = argparse.ArgumentParser(description="Simulation of a workload execution with the FPGA Workload Manager based on predictions from incremental learning models trained on-chip")
parser.add_argument("--models_path", required=True, help="Path to the models")
parser.add_argument("--workload_path", required=True, help="Path to the workload")
parser.add_argument("--scheduling_policy", required=True, choices=['FCFS', 'STACK', 'SJF', 'LSIF', 'LOIF', 'CU'], help="Scheduling policy to use")
parser.add_argument("--board", required=True, choices=['ZCU', 'PYNQ', 'AU250'], help="Type of board")
args = parser.parse_args(sys.argv[1:])

# Set a fixed seed for reproducibility
random.seed(42)

#
# Workload Initialization
#

workloads = generate_workload_from_files(args.workload_path, 3, args.board)

#
# Models Initialization
#

# Open models files
with open(args.models_path, 'rb') as file:
    online_models_list = pickle.load(file)

#
# Simulation Initialization
#

simulations = []

# Create a Workload Simulation object for each workload
for i, workload in enumerate(workloads):
    # Create a WorkloadSchedulingSimulation object
    sim = WorkloadSimulator(online_models_list, workload, args.scheduling_policy, args.board)
    # Append to the list of simulations
    simulations.append(sim)

#
# Simulation Execution
#

total_time = 0.0
total_wait_time = 0.0
total_decisions = 0
affected_decisions = 0

# Run each simulation
for i, sim in enumerate(simulations):
    print(f"#{i}")
    total_time_tmp, total_wait_time_tmp, total_decisions_tmp, affected_decisions_tmp = sim.run()
    total_time += total_time_tmp
    total_wait_time += total_wait_time_tmp
    total_decisions += total_decisions_tmp
    affected_decisions += affected_decisions_tmp

print(f"total time: {total_time}")
print(f"total wait time: {total_wait_time}")
print(f"total decisions: {total_decisions}")
print(f"affected decisions: {affected_decisions}")
