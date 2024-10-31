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

#from incremental_learning import online_models as om


class WorkloadSchedulingSimulation:
    """
    Simulation of a workload scheduling system with online learning models.
    """

    def __init__(self, models, workload, schdeuler, board):
        self.models = models
        self.workload = workload.copy()
        self.scheduler = schdeuler
        self.waiting_queue = []
        self.running_queue = []
        self.finished_queue = []
        self.blocked_ids = []
        self.free_slots = 8 if board == "ZCU" else 4
        # self.current_time = 0.0
        # self.time_step = 0.001
        self.current_time = Decimal('0.0')
        self.time_step = Decimal('0.001')
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
                # Indicate that there are kernels that can be executed
                self.are_kernels_executable = True
                # print(f"Kernel {kernel['tmp_id']} finished at {self.current_time} with {kernel['cu']} CUs")
                # print(f"Free slots: {self.free_slots}")

    def _schedule(self):
        """
        Schedule the kernels in the waiting queue to the running queue.
        NOTE: All the schedulable kernels are scheduled in the same time step.
        """
        # TODO: Implement the scheduling algorithm

        for kernel in self.waiting_queue:
            # Check if the kernel can be scheduled
            if kernel["cu"] <= self.free_slots:
                # TODO: Change with models prediction
                kernel["end_time"] = self.current_time + random.choice([1.0,1.5,2.0,2.5,3.0]) * kernel["num_executions"]
                # Move the kernel to the running queue
                self.running_queue.append(kernel)
                # Sort the running queue by end time (eases the check of finished kernels)
                self.running_queue.sort(key=lambda x: x['end_time'])
                # Remove the kernel from the waiting queue
                self.waiting_queue.remove(kernel)
                # Update the free slots
                self.free_slots -= kernel["cu"]
                # break # Schedule only one kernel per time step
                # print(f"Kernel {kernel['tmp_id']} started at {self.current_time} with {kernel['cu']} CUs")
                # print(f"Free slots: {self.free_slots}")

            # Check if there are free slots available
            if self.free_slots == 0:
                break  # Stop scheduling when no free slots are available

        # Indicate that there are no kernels that can be executed
        # Since all the schedulable kernels have been scheduled (wait for arrival of new kernels or finish of running kernels)
        self.are_kernels_executable = False

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
            # self.current_time += self.time_step

            # print(f"Current time: {current_time}")

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

    # models/model_error_figures/adapt_models.pkl

    # Open models files
    # with open("./model_error_figures/online_models.pkl", 'rb') as file:
    #     online_models = pickle.load(file)

    # print(type(online_models))

    # for model in online_models._models:
    #     print(type(model))


    # def _predict_one(self, features_dict):
    #         """Make a prediction based on given features for each model."""

    #         # Make a prediction for each model
    #         return [model.predict_one(features_dict) for model in self._models]

    # def _predict_one(self, features_dict):
    #         """Make a prediction based on given features for each model."""

    #         # Make a prediction for each model
    #         return [model.predict_one(features_dict) for model in self._models]

    #
    # Scheduler Initialization
    #

    # Create WorkloadSchedulingSimulation object
    simulation = WorkloadSchedulingSimulation(None, workload, None, "ZCU")

    # Run simulation
    simulation.run()


if __name__ == "__main__":
    main()