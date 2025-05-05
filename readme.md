# FPGA Workload Simulator

This repository contains a software simulator for the FPGA Workload Manager. The original FPGA Workload Manager repository can be found [here](https://github.com/juanea7/fpga-worload-manager.git).


The simulation is performed based on the prediction of the incremental learning models from [this repository](https://github.com/juanea7/fpga-modeling.git). These models need to be pre-trained on the target board beforehand.


## Overview

The FPGA Workload Simulator is designed to simulate the behavior of dynamic workloads running on FPGAs, as if they were executed with the FPGA Workload Manager. It allows users to test and evaluate different workload scenarios without the need for physical FPGA hardware. By using this simulator, users can gain insights into workload management strategies complementing the capabilities of the original FPGA Workload Manager.

## Features

- Simulates FPGA workload management
- Supports various workload scenarios
- Supports different target boards
- Provides performance metrics and analysis
- Easy to configure and extend

## Structure

```text
fpga-workload-simulator/
├── clone_incremental_learning_module.sh     # Script clone IL module
├── models/                                  # Pre-trained models
│   ├── adapt_models.pkl
│   ├── all_models.pkl
│   └── ...                                  # Other model variants
├── simulator/                               # Main simulator implementation
│   ├── __init__.py
│   ├── simulation.py                        # Simulator implementation
│   ├── scheduling_policies.py               # Scheduling policy implementations (FCFS, SJF, etc.)
├── workloads/                               # Workload generation and loading
│   ├── data/                                # Pre-exiting workload files
│   ├── generator.py                         # Utilities to create workload from files
├── workload_simulator.py                    # Entry-point script that runs the simulation of the workload execution
└── README.md                                # You are here
```


## Installation

To install the FPGA Workload Simulator, clone this repository, run the script to clone the Incremental Learning module, and install the requirements:

```bash
git clone https://github.com/juanea7/fpga-workload-simulator.git
cd fpga-workload-simulator
./clone_incremental_learning_module.sh
pip install -r requirements.txt
```

## Usage

To run the simulator, use the following command (with you own arguments):

```bash
python workload_scheduling_simulation.py \
    --models_path "models/adapt_models.pkl" \
    --workload_path "workload" \
    --scheduling_policy "CU" \
    --board "ZCU"
```


