"""
Workload Generator

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : May 2025
Description : This file contains functions for loading binary containing lists of floats or ints,
              as well as generating workloads based on the loaded data.

"""


import random
import struct


def load_binary_float_list(file_path):
    """
    Load a binary file containing a list of floats.
    :param file_path: Path to the binary file.
    :return: List of floats.
    """
    # Open the file at the given file_path in binary read mode
    with open(file_path, "rb") as f:
        float_list = []  # Initialize an empty list to store the floating-point numbers
        while True:
            # Read 4 bytes from the file (size of a single-precision float)
            bytes_read = f.read(4)
            if not bytes_read:  # If no more bytes are read, exit the loop
                break
            # Unpack the 4 bytes into a floating-point number using struct
            float_value = struct.unpack('f', bytes_read)[0]
            # Append the unpacked float to the list
            float_list.append(float_value)
    # Return the list of floating-point numbers
    return float_list


def load_binary_int_list(file_path):
    """
    Load a binary file containing a list of integers.
    :param file_path: Path to the binary file.
    :return: List of integers.
    """
    # Open the file at the given file_path in binary read mode
    with open(file_path, "rb") as f:
        int_list = []  # Initialize an empty list to store the integers
        while True:
            # Read 4 bytes from the file (size of a single-precision float)
            bytes_read = f.read(4)
            if not bytes_read:  # If no more bytes are read, exit the loop
                break
            # Unpack the 4 bytes into an integer using struct
            int_value = struct.unpack('i', bytes_read)[0]
            # Append the unpacked integer to the list
            int_list.append(int_value)
    # Return the list of integers
    return int_list


def load_workload_information(workload_path, workload_id):
    """
    Load workload information from binary files.
    :param workload_path: Path to the directory containing the workload files.
    :param workload_id: Identifier for the workload.
    :return: Dictionary containing inter-arrival times, kernel IDs, and number of executions.
    """
    # Initialize empty lists for inter-arrival times, kernel IDs, and number of executions
    inter_arrival_values = []
    kernel_id_values = []
    num_executions_values = []

    # Load the data from binary files
    inter_arrival_values = load_binary_float_list(f"{workload_path}/inter_arrival_{workload_id}.bin")
    kernel_id_values = load_binary_int_list(f"{workload_path}/kernel_id_{workload_id}.bin")
    num_executions_values = load_binary_int_list(f"{workload_path}/num_executions_{workload_id}.bin")

    # Create a dictionary to store the workload information
    workload_information_dict = {
        "inter_arrival": inter_arrival_values,
        "kernel_id": kernel_id_values,
        "num_executions": num_executions_values
    }

    return workload_information_dict


def generate_workload(workload_information_dict, board):
    """
    Generate a workload based on the information provided in the workload_information_dict and the board type (ZCU or ZCU).
    """

    # Define the number of slots available in the board
    slot_choices = [1,2,4,8] if board == "ZCU" else [1, 2, 4]

    workload = []
    arrival_time = 0.0
    # Insert elements in the workload
    for i in range(len(workload_information_dict["inter_arrival"])):
        # Compute arrival time
        arrival_time += workload_information_dict["inter_arrival"][i]
        # Generate cu
        cu = random.choice(slot_choices)
        # Add kernel to the workload
        workload.append({
            "arrival_ms":round(arrival_time,3),
            "kernel_id": workload_information_dict["kernel_id"][i],
            "num_executions": workload_information_dict["num_executions"][i],
            "cu": cu,
            "start_time" : None,
            "end_time": None,
            "job_remaining_percentage": None,
            "last_update_time": None,
            "last_predicted_time": None,
            "tmp_id" : i
        })

    return workload


def generate_workload_from_files(workload_path, num_subworkloads, board):
    """
    Generate a workload from binary files.
    :param workload_path: Path to the directory containing the workload files.
    :param num_subworkloads: Number of subworkloads to generate.
    :param board: Type of board (ZCU or ZCU).
    :return: List of workloads.
    """
    workloads = []
    for i in range(num_subworkloads):
        # Load workload information from binary files
        workload_info = load_workload_information(workload_path, i)
        # Generate the workload
        workload = generate_workload(workload_info, board)
        workloads.append(workload)
    return workloads


if __name__ == "__main__":
    # Example usage
    workload_path = "data/"
    num_subworkloads = 3
    board = "ZCU"
    workloads = generate_workload_from_files(workload_path, num_subworkloads, board)

    # Print the generated workloads
    for i, workload in enumerate(workloads):
        print(f"Workload {i}: ", workload[:20])