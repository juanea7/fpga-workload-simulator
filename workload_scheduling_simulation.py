"""
Simulation of a workload scheduling system with online learning models

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2024
Description : This script:
                - Simulates a workload scheduling system with online learning models

"""

import struct
import pickle
import random
import pprint
from decimal import Decimal
import time
import ctypes as ct

from incremental_learning import online_models as om


class WorkloadSchedulingSimulation:
    """
    Simulation of a workload scheduling system with online learning models.
    """

    def __init__(self, models, workload, schdeuler, board):
        self.kernel_names = ["aes", "bulk", "crs", "kmp", "knn", "merge", "nw", "queue", "stencil2d", "stencil3d", "strided"]
        self.models = models
        self.workload = workload.copy()
        self.scheduler = schdeuler
        self.waiting_queue = []
        self.running_queue = []
        self.finished_queue = []
        self.current_configuration = {"aes": 0, "bulk": 0, "crs": 0, "kmp": 0, "knn": 0, "merge": 0, "nw": 0, "queue": 0, "stencil2d": 0, "stencil3d": 0, "strided": 0}
        self.free_slots = 8 if board == "ZCU" else 4
        self.current_time = 0.0
        self.time_step = 0.001
        self.are_kernels_executable = False  # Flag to check if there are kernels that can be executed (i.e., new arrivals or finished kernels. Set to False to avoid scheduling kernels when waiting queue has been completely checked)

    def _update_current_time(self):
        """
        Update the current time of the simulation.
        """

        # Check if there are kernels that can be executed (i.e., new arrivals or finished kernels and free slots available)
        if self.are_kernels_executable and self.free_slots > 0:
            # Update the current time to the next time step when there are kernels that can be executed
            self.current_time += self.time_step
        else:
            # Update the current time to closest time between the arrival time of the next kernel and the end time of the first kernel in the running queue
            self.current_time = min(
                self.workload[0]["arrival_ms"] if self.workload else float('inf'),
                self.running_queue[0]["end_time"] if self.running_queue else float('inf')
            )
        if self.current_time == float('inf'):
            print("Simulation error: current time is infinite")

    def _check_new_arrivals(self):
        """
        Check if there are new arrivals in the workload and add them to the waiting queue.
        """
        for kernel in self.workload:
            # Check if the kernel has arrived
            if kernel["arrival_ms"] <= self.current_time:
                # Add the kernel to the waiting queue
                self.waiting_queue.append(kernel)
                # Remove the kernel from the workload
                self.workload.remove(kernel)
                # Indicate that there are kernels that can be executed
                self.are_kernels_executable = True
                # print(f"Kernel {kernel['tmp_id']} arrived at {self.current_time} with {kernel['cu']} CUs")
            else:
                break  # The workload is sorted by arrival time

    def _check_finished_kernels(self):
        """
        Check if there are finished kernels in the running queue and move them to the finished queue.
        """
        for kernel in self.running_queue:
            # Check if the kernel has finished
            if kernel["end_time"] <= self.current_time:
                # Move the kernel to the finished queue
                self.finished_queue.append(kernel)
                # Remove the kernel from the running queue
                self.running_queue.remove(kernel)
                # Update the free slots
                self.free_slots += kernel["cu"]
                # Update the current configuration
                self.current_configuration[self.kernel_names[kernel["kernel_id"]]] -= kernel["cu"]
                # Indicate that there are kernels that can be executed
                self.are_kernels_executable = True
                # print(f"Kernel {kernel['tmp_id']} finished at {self.current_time} with {kernel['cu']} CUs")
                # print(f"Free slots: {self.free_slots}")
                # print(f"Current configuration: {self.current_configuration}")

    def _scheduling_policy(self):
        """
        Schedule the kernels in the waiting queue to the running queue.
        NOTE: All the schedulable kernels are scheduled in the same time step.
        """
        # TODO: Implement the scheduling algorithm

        for kernel in self.waiting_queue:
            # Check if the kernel can be scheduled (i.e., there are free slots available and the kernel has not been scheduled yet(artico3))
            if kernel["cu"] <= self.free_slots and self.current_configuration[self.kernel_names[kernel["kernel_id"]]] == 0:
                # TODO: Change with models prediction
                # kernel["end_time"] = self.current_time + random.choice([1.0,1.5,2.0,2.5,3.0]) * kernel["num_executions"]

                return kernel # Schedule only one kernel per time step

            # Check if there are free slots available
            if self.free_slots == 0:
                break  # Stop scheduling when no free slots are available

        # Indicate that there are no kernels that can be executed
        # Since all the schedulable kernels have been scheduled (wait for arrival of new kernels or finish of running kernels)
        self.are_kernels_executable = False
        return None

    def _execute_kernel(self, kernel):
        """
        Execute the kernel.
        """

        # Create feature (TODO: Make CPU usage more realistic)
        cpu_usage = {"user": 50.0, "kernel": 25.0, "idle": 25.0}
        feature = cpu_usage | self.current_configuration
        feature["Main"] = kernel["kernel_id"]
        tmp_kernel_name = self.kernel_names[kernel["kernel_id"]]
        feature[tmp_kernel_name] = kernel["cu"]

        # Predict the time of execution of the kernel
        time_prediction = round(self.models[-1].predict_one(feature), 3)
        # Update the end time of the kernel
        kernel["end_time"] = self.current_time + kernel["num_executions"] * time_prediction
        # print(feature)
        # print(time_prediction)
        # print(kernel["num_executions"])
        # print(kernel["end_time"])

        # Move the kernel to the running queue
        self.running_queue.append(kernel)
        # Sort the running queue by end time (eases the check of finished kernels)
        self.running_queue.sort(key=lambda x: x['end_time'])
        # Remove the kernel from the waiting queue
        self.waiting_queue.remove(kernel)
        # Update the free slots
        self.free_slots -= kernel["cu"]
        # Update the current configuration
        self.current_configuration[self.kernel_names[kernel["kernel_id"]]] += kernel["cu"]
        # print(f"Kernel {kernel['tmp_id']} started at {self.current_time} with {kernel['cu']} CUs")
        # print(f"Free slots: {self.free_slots}")
        # print(f"Current configuration: {self.current_configuration}")

    def _schedule(self):

        kernel = self._scheduling_policy()
        if kernel:
            self._execute_kernel(kernel)


    def run(self):
        """
        Run the simulation.
        """

        # Variables to measure the time spent in each step
        arrival = 0
        finish = 0
        schedule = 0

        total_start = time.time()

        # Run the simulation while there are kernels in the workload, waiting queue or running queue
        while self.workload or self.waiting_queue or self.running_queue:

            # Update the simulation time
            self._update_current_time()

            # arrival_start = time.time()
            # Check if there are new arrivals
            self._check_new_arrivals()
            # arrival_end = time.time()
            # arrival += arrival_end - arrival_start

            # finish_start = time.time()
            # Check if there are finished kernels
            self._check_finished_kernels()
            # finish_end = time.time()
            # finish += finish_end - finish_start

            # schedule_start = time.time()
            # Schedule the kernels
            self._schedule()
            # schedule_end = time.time()
            # schedule += schedule_end - schedule_start

        total_end = time.time()

        print("Simulation finished")

        simulation_time_sec = self.finished_queue[-1]['end_time'] / 1000
        print(f"Simulation time: {simulation_time_sec} s")
        print(f"Wall-clock time: {total_end - total_start} s")
        # print(f"Arrival time: {arrival}")
        # print(f"Finish time: {finish}")
        # print(f"Schedule time: {schedule}")


def generate_workload(workload_information_dict, board):
    """
    Generate a workload based on the information provided in the workload_information_dict and the board type (ZCU or ZCU).
    """

    # Define the number of slots available in the board
    slot_choices = [1, 2, 4, 8] if board == "ZCU" else [1, 2, 4]

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
            "end_time": None,
            "tmp_id" : i
        })

    return workload

def main():
    # TODO: Add argparse to parse arguments

    # Set a fixed seed for reproducibility
    random.seed(42)

    #
    # Workload Initialization
    #

    # Read workload information from files
    inter_arrival_values = []
    with open("workload/inter_arrival_0.bin", "rb") as file:
        while True:
            value = file.read(4)
            if not value:
                break
            inter_arrival_values.append(struct.unpack('f', value)[0])

    kernel_id_values = []
    with open("workload/kernel_id_0.bin", "rb") as file:
        while True:
            value = file.read(4)
            if not value:
                break
            kernel_id_values.append(struct.unpack('i', value)[0])

    num_executions_values = []
    with open("workload/num_executions_0.bin", "rb") as file:
        while True:
            value = file.read(4)
            if not value:
                break
            num_executions_values.append(struct.unpack('i', value)[0])

    # Generate workload dictionary
    workload_information_dict = {
        "inter_arrival": inter_arrival_values,
        "kernel_id": kernel_id_values,
        "num_executions": num_executions_values
    }

    # Construct the workload
    workload = generate_workload(workload_information_dict, "ZCU")
    # pprint.pprint(workload[:10])

    #
    # Models Initialization
    #

    # Open models files
    with open("models/model_error_figures/adapt_models.pkl", 'rb') as file:
        online_models_list = pickle.load(file)
    # print(type(online_models_list))
    # for model in online_models_list:
    #     print(type(model))

    #
    # Scheduler Initialization
    #

    # Create WorkloadSchedulingSimulation object
    simulation = WorkloadSchedulingSimulation(online_models_list, workload[:100], None, "ZCU")

    # Run simulation
    simulation.run()


if __name__ == "__main__":
    main()