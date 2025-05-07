"""
Workload Simulator

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : May 2025
Description : This file contains the WorkloadSimulator class, which simulates the execution of a
              workload on a given board using a specified scheduling policy and a list of online learning models.
              The simulation includes the management of a workload, waiting queue, running queue, and finished queue.
              It also provides methods to update the current time, CPU usage, check for new arrivals,
              check for finished kernels, update kernel end times, execute kernels, and schedule them based on the
              specified scheduling policy.

"""


import random
import copy
import time

from .scheduling_policies import *

# Test CSA TODO: Make it more general
import sys

sys.path.insert(1, '/media/juan/HDD/git_repos/fpga-scheduling/')

import csa


class WorkloadSimulator:
    """
    Simulation of a workload scheduling system with online learning models.
    """

    def __init__(self, models, workload, scheduling_policy, board):
        self.kernel_names = ["aes", "bulk", "crs", "kmp", "knn", "merge", "nw", "queue", "stencil2d", "stencil3d", "strided"]
        self.models = models
        self.workload = copy.deepcopy(workload)
        self.waiting_queue = []
        self.running_queue = []
        self.finished_queue = []
        self.current_configuration = {"aes": 0, "bulk": 0, "crs": 0, "kmp": 0, "knn": 0, "merge": 0, "nw": 0, "queue": 0, "stencil2d": 0, "stencil3d": 0, "strided": 0}
        self.free_slots = 8 if board == "ZCU" else 4
        self.kernels_in_execution = 0
        self.current_time = 0.0
        self.time_step = 0.001
        self.are_kernels_executable = False  # Flag to check if there are kernels that can be executed (i.e., new arrivals or finished kernels. Set to False to avoid scheduling kernels when waiting queue has been completely checked)
        self.total_scheduling_decisions = 0
        self.active_scheduling_decisions = 0
        self.tmp_kernel_default_count = 0
        self.cpu_usage = {}
        self.cpu_usage["user"] = random.uniform(30.0, 70.0)
        self.cpu_usage["kernel"] = random.uniform(10.0, 30.0)
        self.cpu_usage["idle"] = 100.0 - self.cpu_usage["user"] - self.cpu_usage["kernel"]

        # Test CSA TODO: Make it more elegant
        self.csa_scheduler = csa.CrowSearchAlgorithm(len(self.kernel_names), self.free_slots, kernel_names=self.kernel_names, models=self.models)

        # Dictionary to map scheduling policies to corresponding methods
        self.scheduling_methods = {
            'FCFS': fist_come_first_served_policy,
            'STACK': stack_policy,
            'SJF': shortest_job_first_policy,
            'LSIF': least_self_interaction_first_policy,
            'LOIF': least_overall_interaction_first_policy,
            'CU': cu_selection_policy,
            'CSA': csa_policy
        }

        # Set the scheduling policy
        if scheduling_policy in self.scheduling_methods:
            self.scheduler = self.scheduling_methods[scheduling_policy]
            print(f"Scheduling policy: {scheduling_policy}")
        else:
            raise ValueError(f"Invalid scheduling policy: {scheduling_policy}")

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

    def _update_cpu_usage(self):
        """
        Update the CPU usage of the simulation.
        """

        # Update the CPU usage
        self.cpu_usage["user"] = random.uniform(30.0, 70.0)
        self.cpu_usage["kernel"] = random.uniform(10.0, 30.0)
        self.cpu_usage["idle"] = 100.0 - self.cpu_usage["user"] - self.cpu_usage["kernel"]

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

        configuration_changed = False
        for kernel in self.running_queue[:]:
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
                self.kernels_in_execution -= 1
                configuration_changed = True
                # print(f"Kernel {kernel['tmp_id']} finished at {self.current_time} with {kernel['cu']} CUs")
                # print(f"Free slots: {self.free_slots}")
                # print(f"Current configuration: {self.current_configuration}")

        # Check whether the configuration has changed
        if configuration_changed:
            # Update the end time of the kernels (since the configuration is different now)
            for kernel in self.running_queue:
                self._update_kernel_end_time(kernel)
            # Sort the running queue by end time (eases the check of finished kernels)
            self.running_queue.sort(key=lambda x: x['end_time'])

    def _update_kernel_end_time(self, kernel):
        """
        Update the end time of the kernel based on the current time and the time of execution of the kernel.
        Every kernel running is updated, even kernels that are starting to run.
        """

        # Create feature (TODO: Make CPU usage more realistic)
        # cpu_usage = self.cpu_usage.copy()
        feature = {**self.cpu_usage, **self.current_configuration}
        feature["Main"] = kernel["kernel_id"]
        tmp_kernel_name = self.kernel_names[kernel["kernel_id"]]
        feature[tmp_kernel_name] = kernel["cu"]

        # Check if the kernel has started to run
        if kernel["start_time"]:
            # Kernel is already running
            # Compute the job performed since last update, from 0 to 1, understading a 1 as the
            # entire job to finish the execution (even if in last update already 50% has been done)
            job_performed_since_last_update = (self.current_time - kernel["last_update_time"]) / (kernel["end_time"] - kernel["last_update_time"])
            # Update the job remaining percentage.
            # Multiplying the job remaining percentage by the job performed since last update
            kernel["job_remaining_percentage"] *= (1 - job_performed_since_last_update)
            kernel["last_update_time"] = self.current_time
        else:
            # Kernel is starting to run
            kernel["start_time"] = self.current_time
            kernel["last_update_time"] = self.current_time
            kernel["job_remaining_percentage"] = 1

        if kernel["job_remaining_percentage"] < 0:
            # This should never happen, but just in case
            print("Error: kernel['job_remaining_percentage'] < 0")
            print("Current time: ", self.current_time)
            print("Arrival time: ", kernel["arrival_time"])
            print("End time: ", kernel["end_time"])
            print("Job percentage completed: ", kernel['job_remaining_percentage'])

        # Compute the remaining executions of the kernel
        remaining_executions = kernel["num_executions"] * kernel["job_remaining_percentage"]

        # Predict the time of execution of the kernel
        time_prediction = round(self.models[-1].predict_one(feature), 3)

        # print("Kernel: ", kernel["tmp_id"])
        # print("Kernel start time: ", kernel["start_time"])
        # print("Current time: ", self.current_time)
        # print("Last update time: ", kernel["last_update_time"])
        # print("Job remaining percentage: ", kernel["job_remaining_percentage"])
        # print("Remaining executions: ", remaining_executions)
        # print("Time prediction: ", time_prediction)
        # print("Kernel prev end time: ", kernel["end_time"])

        # Update the end time of the kernel
        kernel["end_time"] = self.current_time + (remaining_executions * time_prediction) / kernel["cu"]
        kernel["last_predicted_time"] = time_prediction
        # print("Kernel new end time: ", kernel["end_time"])
        # print("\n")

    def _execute_kernel(self, kernel):
        """
        Execute the kernel.
        """

        # Move the kernel to the running queue
        self.running_queue.append(kernel)
        # Remove the kernel from the waiting queue
        self.waiting_queue.remove(kernel)
        # Update the free slots
        self.free_slots -= kernel["cu"]
        self.kernels_in_execution += 1
        # Update the current configuration
        self.current_configuration[self.kernel_names[kernel["kernel_id"]]] += kernel["cu"]

        # print("\nNew set of decisions")

        # Update the end time of the kernels
        for kernel in self.running_queue:
            self._update_kernel_end_time(kernel)

        # Sort the running queue by end time (eases the check of finished kernels)
        self.running_queue.sort(key=lambda x: x['end_time'])

        # print(f"Kernel {kernel['tmp_id']} started at {self.current_time} with {kernel['cu']} CUs")
        # print(f"Free slots: {self.free_slots}")
        # print(f"Current configuration: {self.current_configuration}")

    def _schedule(self):
        """
        Schedule the kernels based on the scheduling policy.
        """

        # Check if there are kernels that can be executed
        if not self.are_kernels_executable or self.free_slots == 0:
            return

        # Grab the kernel to be executed based on the scheduling policy
        kernels = self.scheduler(self)

        # Check if there are kernels to be executed
        if kernels is None:
            return

        # Check if the kernel is a list of kernels or a single kernel
        if isinstance(kernels, np.ndarray):
            # Execute the kernels in the list
            for kernel in kernels:
                self._execute_kernel(kernel)
        else:
            # Execute the kernel
            self._execute_kernel(kernels)

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

            # Update CPU usage
            self._update_cpu_usage()

            arrival_start = time.time()
            # Check if there are new arrivals
            self._check_new_arrivals()
            arrival_end = time.time()
            arrival += arrival_end - arrival_start

            finish_start = time.time()
            # Check if there are finished kernels
            self._check_finished_kernels()
            finish_end = time.time()
            finish += finish_end - finish_start

            schedule_start = time.time()
            # Schedule the kernels
            self._schedule()
            schedule_end = time.time()
            schedule += schedule_end - schedule_start

        total_end = time.time()

        simulation_time_sec = self.finished_queue[-1]['end_time'] / 1000
        print(f"Simulation time: {simulation_time_sec} s")
        print(f"Wall-clock time: {total_end - total_start} s")
        print(f"Arrival time: {arrival}")
        print(f"Finish time: {finish}")
        print(f"Schedule time: {schedule}")

        print("self.tmp_kernel_default_count", self.tmp_kernel_default_count)

        total_wait_time = 0.0
        for kernel in self.finished_queue:
            total_wait_time += kernel["start_time"] - kernel["arrival_ms"]

        if self.total_scheduling_decisions <= 0: # Avoid division by zero
            return simulation_time_sec, total_wait_time, self.total_scheduling_decisions, self.active_scheduling_decisions

        # print("Total scheduling decisions: ", self.total_scheduling_decisions)
        # print("Active scheduling decisions: ", self.active_scheduling_decisions)
        # print("Percentage of active scheduling decisions: ", self.active_scheduling_decisions / self.total_scheduling_decisions * 100)

        return simulation_time_sec, total_wait_time, self.total_scheduling_decisions, self.active_scheduling_decisions
