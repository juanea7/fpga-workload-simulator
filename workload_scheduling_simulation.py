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
import copy

from incremental_learning import online_models as om


class WorkloadSchedulingSimulation:
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

        # Dictionary to map scheduling policies to corresponding methods
        self.scheduling_methods = {
            'FCFS': self._fist_come_first_served_policy,
            'STACK': self._stack_policy,
            'SJF': self._shortest_job_first_policy,
            'LIF': self._least_interaction_first_policy,
            'CU': self._cu_selection
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
        """

        # Create feature (TODO: Make CPU usage more realistic)
        # cpu_usage = self.cpu_usage.copy()
        feature = self.cpu_usage | self.current_configuration
        feature["Main"] = kernel["kernel_id"]
        tmp_kernel_name = self.kernel_names[kernel["kernel_id"]]
        feature[tmp_kernel_name] = kernel["cu"]

        if kernel["start_time"]:
            job_performed_since_last_update = (self.current_time - kernel["last_update_time"]) / (kernel["end_time"] - kernel["last_update_time"])
            kernel["job_remaining_percentage"] *= (1 - job_performed_since_last_update)
            kernel["last_update_time"] = self.current_time
        else:
            kernel["start_time"] = self.current_time
            kernel["last_update_time"] = self.current_time
            kernel["job_remaining_percentage"] = 1

        if kernel["job_remaining_percentage"] < 0:
            print("Error: kernel['job_remaining_percentage'] < 0")
            print("Current time: ", self.current_time)
            print("Arrival time: ", kernel["arrival_time"])
            print("End time: ", kernel["end_time"])
            print("Job percentage completed: ", kernel['job_remaining_percentage'])

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
        kernel = self.scheduler()

        # Execute the kernel
        if kernel:
            self._execute_kernel(kernel)

    def _fist_come_first_served_policy(self):
        """
        Schedule the kernels in the waiting queue to the running queue.
        NOTE: Just one scheduling decition is made in each time step.
        """
        # TODO: Implement the scheduling algorithm

        for kernel in self.waiting_queue:
            # Check if the kernel can be scheduled (i.e., there are free slots available and the kernel has not been scheduled yet(artico3))
            # if kernel["cu"] <= self.free_slots and self.current_configuration[self.kernel_names[kernel["kernel_id"]]] == 0:
            if kernel["cu"] <= self.free_slots and self.current_configuration[self.kernel_names[kernel["kernel_id"]]] == 0:
                # TODO: Change with models prediction
                # kernel["end_time"] = self.current_time + random.choice([1.0,1.5,2.0,2.5,3.0]) * kernel["num_executions"]

                return kernel # Schedule only one kernel per time step

        # Indicate that there are no kernels that can be executed
        # Since all the schedulable kernels have been scheduled (wait for arrival of new kernels or finish of running kernels)
        self.are_kernels_executable = False
        return None

    def _stack_policy(self):
        """
        Schedule the kernels in the waiting queue to the running queue.
        NOTE: Just one scheduling decition is made in each time step.
        """
        # TODO: Implement the scheduling algorithm

        for kernel in self.waiting_queue[::-1]:
            # Check if the kernel can be scheduled (i.e., there are free slots available and the kernel has not been scheduled yet(artico3))
            if kernel["cu"] <= self.free_slots and self.current_configuration[self.kernel_names[kernel["kernel_id"]]] == 0:
                # TODO: Change with models prediction
                # kernel["end_time"] = self.current_time + random.choice([1.0,1.5,2.0,2.5,3.0]) * kernel["num_executions"]

                return kernel # Schedule only one kernel per time step

        # Indicate that there are no kernels that can be executed
        # Since all the schedulable kernels have been scheduled (wait for arrival of new kernels or finish of running kernels)
        self.are_kernels_executable = False
        return None

    def _shortest_job_first_policy(self):
        """
        Schedule the kernels in the waiting queue to the running queue.
        NOTE: Just one scheduling decition is made in each time step.
        """
        # TODO: Implement the scheduling algorithm

        num_kernels_to_compare = 1
        kernels_to_compare = []

        # Get the kernels to compare
        for kernel in self.waiting_queue:
            # Check if the kernel can be scheduled (i.e., there are free slots available and the kernel has not been scheduled yet(artico3))
            if kernel["cu"] <= self.free_slots and self.current_configuration[self.kernel_names[kernel["kernel_id"]]] == 0:

                # Store the kernel to compare
                kernels_to_compare.append(kernel)

                # Check if there are enough kernels to compare
                if len(kernels_to_compare) == num_kernels_to_compare: break

        # Check if there are kernels to compare
        if len(kernels_to_compare) == 0:
            # Indicate that there are no kernels that can be executed
            # Since all the schedulable kernels have been scheduled (wait for arrival of new kernels or finish of running kernels)
            self.are_kernels_executable = False
            return None

        # Increase the total scheduling decisions
        self.total_scheduling_decisions += 1

        # Return the kernel when there is only one kernel to compare
        if len(kernels_to_compare) == 1:
            return kernels_to_compare[0]

        #
        # Compare the kernels
        #

        # Initialize the minimum job time and the kernel to be scheduled
        min_job_time = float('inf')
        min_kernel = None

        # print("hello\n")

        for kernel in kernels_to_compare:
            # Create feature (TODO: Make CPU usage more realistic)
            feature = self.cpu_usage | self.current_configuration
            feature["Main"] = kernel["kernel_id"]
            tmp_kernel_name = self.kernel_names[kernel["kernel_id"]]
            feature[tmp_kernel_name] = kernel["cu"]

            # Predict the time of execution of the kernel
            time_prediction = round(self.models[-1].predict_one(feature), 3)
            job_time = time_prediction * kernel["num_executions"] / kernel["cu"]
            # print("Kernel: ", kernel["tmp_id"])
            # print(feature)
            # print(time_prediction)
            # print(job_time)

            # Update the min kernel
            min_kernel = kernel if job_time < min_job_time else min_kernel
            # Check if the job time is the minimum
            # min_job_time = min(job_time, min_job_time)

        # Increase the active scheduling decisions
        if kernels_to_compare[0] != min_kernel:
            self.active_scheduling_decisions += 1

        # print("Min kernel: ", min_kernel["tmp_id"])

        return min_kernel

    def _least_interaction_first_policy(self):
        """
        Schedule the kernels in the waiting queue to the running queue.
        NOTE: Just one scheduling decition is made in each time step.
        """
        # TODO: Implement the scheduling algorithm

        num_kernels_to_compare = 2
        kernels_to_compare = []

        alone_configuration = {"aes": 0, "bulk": 0, "crs": 0, "kmp": 0, "knn": 0, "merge": 0, "nw": 0, "queue": 0, "stencil2d": 0, "stencil3d": 0, "strided": 0}

        # Get the kernels to compare
        for kernel in self.waiting_queue:
            # Check if the kernel can be scheduled (i.e., there are free slots available and the kernel has not been scheduled yet(artico3))
            if kernel["cu"] <= self.free_slots and self.current_configuration[self.kernel_names[kernel["kernel_id"]]] == 0:

                # Store the kernel to compare
                kernels_to_compare.append(kernel)

                # Check if there are enough kernels to compare
                if len(kernels_to_compare) == num_kernels_to_compare: break

        # Check if there are kernels to compare
        if len(kernels_to_compare) == 0:
            # Indicate that there are no kernels that can be executed
            # Since all the schedulable kernels have been scheduled (wait for arrival of new kernels or finish of running kernels)
            self.are_kernels_executable = False
            return None

        # Increase the total scheduling decisions
        self.total_scheduling_decisions += 1

        # Return the kernel when there is only one kernel to compare
        if len(kernels_to_compare) == 1:
            return kernels_to_compare[0]

        #
        # Compare the kernels
        #

        # Initialize the minimum job time and the kernel to be scheduled
        min_interaction_time = float('inf')
        min_kernel = None

        # print("hello\n")

        for kernel in kernels_to_compare:

            # Current configuration
            # print("current")
            # Create feature (TODO: Make CPU usage more realistic)
            feature = self.cpu_usage | self.current_configuration
            tmp_kernel_name = self.kernel_names[kernel["kernel_id"]]
            feature[tmp_kernel_name] = kernel["cu"]

            acumulated_future_execution_time = 0.0
            acumulated_current_execution_time = 0.0
            # print("new set of decisions")
            if not self.running_queue:
                return kernel # Schedule first kernel if no kernels are running # TODO: Remove
            for in_execution_kernel in self.running_queue:

                feature["Main"] = in_execution_kernel["kernel_id"]

                # Predict the time of execution of the kernel
                future_predicted_time = round(self.models[-1].predict_one(feature), 3)

                tmp_job_performed_since_last_update = (self.current_time - in_execution_kernel["last_update_time"]) / (in_execution_kernel["end_time"] - in_execution_kernel["last_update_time"])
                future_job_remaining_percentage = in_execution_kernel["job_remaining_percentage"] * (1 - tmp_job_performed_since_last_update)

                future_execution_time = future_predicted_time * in_execution_kernel["num_executions"] * future_job_remaining_percentage / in_execution_kernel["cu"]
                # print("Kernel: ", in_execution_kernel["tmp_id"])
                # print(feature)
                # print(future_predicted_time)
                # print(future_execution_time)

                current_execution_time = in_execution_kernel["last_predicted_time"] * in_execution_kernel["num_executions"] * future_job_remaining_percentage / in_execution_kernel["cu"]

                # print("current: ", current_execution_time)
                # print("future: ", future_execution_time)

                acumulated_current_execution_time += current_execution_time
                acumulated_future_execution_time += future_execution_time

            acumulated_interaction_time = (acumulated_future_execution_time - acumulated_current_execution_time) / acumulated_current_execution_time

            # Update the min kernel
            min_kernel = kernel if acumulated_interaction_time < min_interaction_time else min_kernel
            # Check if the job time is the minimum
            min_interaction_time = min(acumulated_interaction_time, min_interaction_time)

            # print("acumulated_current_execution_time", acumulated_current_execution_time)

        # Increase the active scheduling decisions
        if kernels_to_compare[0] != min_kernel:
            self.active_scheduling_decisions += 1
            # print("Decided to change")

        # print("Min kernel: ", min_kernel["tmp_id"])

        return min_kernel

    def _least_interaction_first_policy_old(self):
        """
        Schedule the kernels in the waiting queue to the running queue.
        NOTE: Just one scheduling decition is made in each time step.
        """
        # TODO: Implement the scheduling algorithm

        num_kernels_to_compare = 2
        kernels_to_compare = []

        alone_configuration = {"aes": 0, "bulk": 0, "crs": 0, "kmp": 0, "knn": 0, "merge": 0, "nw": 0, "queue": 0, "stencil2d": 0, "stencil3d": 0, "strided": 0}

        # Get the kernels to compare
        for kernel in self.waiting_queue:
            # Check if the kernel can be scheduled (i.e., there are free slots available and the kernel has not been scheduled yet(artico3))
            if kernel["cu"] <= self.free_slots and self.current_configuration[self.kernel_names[kernel["kernel_id"]]] == 0:

                # Store the kernel to compare
                kernels_to_compare.append(kernel)

                # Check if there are enough kernels to compare
                if len(kernels_to_compare) == num_kernels_to_compare: break

        # Check if there are kernels to compare
        if len(kernels_to_compare) == 0:
            # Indicate that there are no kernels that can be executed
            # Since all the schedulable kernels have been scheduled (wait for arrival of new kernels or finish of running kernels)
            self.are_kernels_executable = False
            return None

        # Increase the total scheduling decisions
        self.total_scheduling_decisions += 1

        # Return the kernel when there is only one kernel to compare
        if len(kernels_to_compare) == 1:
            return kernels_to_compare[0]

        #
        # Compare the kernels
        #

        # Initialize the minimum job time and the kernel to be scheduled
        min_interaction = float('inf')
        min_kernel = None

        # print("hello\n")

        for kernel in kernels_to_compare:

            # Current configuration
            # print("current")
            # Create feature (TODO: Make CPU usage more realistic)
            feature = self.cpu_usage | self.current_configuration
            feature["Main"] = kernel["kernel_id"]
            tmp_kernel_name = self.kernel_names[kernel["kernel_id"]]
            feature[tmp_kernel_name] = kernel["cu"]

            # Predict the time of execution of the kernel
            current_time_prediction = round(self.models[-1].predict_one(feature), 3)
            # print("Kernel: ", kernel["tmp_id"])
            # print(feature)
            # print(current_time_prediction)

            # Alone configuration
            # print("alone")
            # Create feature (TODO: Make CPU usage more realistic)
            feature = self.cpu_usage | alone_configuration
            feature["Main"] = kernel["kernel_id"]
            tmp_kernel_name = self.kernel_names[kernel["kernel_id"]]
            feature[tmp_kernel_name] = kernel["cu"]

            # Predict the time of execution of the kernel
            alone_time_prediction = round(self.models[-1].predict_one(feature), 3)
            # print("Kernel: ", kernel["tmp_id"])
            # print(feature)
            # print(alone_time_prediction)

            # Compute the interaction impact
            interaction_impact = (current_time_prediction - alone_time_prediction) / alone_time_prediction

            # Update the min kernel
            min_kernel = kernel if interaction_impact < min_interaction else min_kernel
            # Check if the job time is the minimum
            min_interaction = min(interaction_impact, min_interaction)

        # Increase the active scheduling decisions
        if kernels_to_compare[0] != min_kernel:
            self.active_scheduling_decisions += 1

        # print("Min kernel: ", min_kernel["tmp_id"])

        return min_kernel

    def _cu_selection(self):
        """
        Schedule the kernels in the waiting queue to the running queue.
        NOTE: Just one scheduling decition is made in each time step.
        """
        # TODO: Implement the scheduling algorithm

        num_kernels_to_compare = 2
        kernels_to_compare = []
        # Get the kernels to compare
        for kernel in self.waiting_queue:
            # Check if the kernel can be scheduled (i.e., there are free slots available and the kernel has not been scheduled yet(artico3))
            if self.free_slots >= 2 and self.current_configuration[self.kernel_names[kernel["kernel_id"]]] == 0:

                # Store the kernel to compare
                kernels_to_compare.append(kernel)

                # Check if there are enough kernels to compare
                if len(kernels_to_compare) == num_kernels_to_compare: break

        # Check if there are kernels to compare
        if len(kernels_to_compare) == 0:
            # Indicate that there are no kernels that can be executed
            # Since all the schedulable kernels have been scheduled (wait for arrival of new kernels or finish of running kernels)
            self.are_kernels_executable = False
            return None

        # Increase the total scheduling decisions
        self.total_scheduling_decisions += 1

        # # Return the kernel when there is only one kernel to compare
        # if len(kernels_to_compare) == 1:
        #     return kernels_to_compare[0]

        #
        # Compare the kernels
        #

        # Initialize the minimum job time and the kernel to be scheduled
        min_interaction_time = float('inf')
        min_kernel = None
        min_cu_option = None

        # print("hello\n")

        for kernel in kernels_to_compare:

            # Set the possible CUs
            slot_choices = [2, 4]
            # Get the available CUs
            cu_available = [slot_choice for slot_choice in slot_choices if slot_choice <= self.free_slots]
            for cu_option in cu_available:

                # Current configuration
                # print("current")
                # Create feature (TODO: Make CPU usage more realistic)
                feature = self.cpu_usage | self.current_configuration
                tmp_kernel_name = self.kernel_names[kernel["kernel_id"]]
                feature[tmp_kernel_name] = cu_option

                cu_acumulated_future_execution_time = 0.0
                cu_acumulated_current_execution_time = 0.0
                # print("new set of decisions")
                if not self.running_queue:
                    # TODO: solve this
                    kernel["cu"] = cu_available[-1]
                    self.tmp_kernel_default_count += 1
                    return kernel # Schedule first kernel if no kernels are running # TODO: Remove

                for in_execution_kernel in self.running_queue:

                    feature["Main"] = in_execution_kernel["kernel_id"]

                    # Predict the time of execution of the kernel
                    future_predicted_time = round(self.models[-1].predict_one(feature), 3)

                    tmp_job_performed_since_last_update = (self.current_time - in_execution_kernel["last_update_time"]) / (in_execution_kernel["end_time"] - in_execution_kernel["last_update_time"])
                    future_job_remaining_percentage = in_execution_kernel["job_remaining_percentage"] * (1 - tmp_job_performed_since_last_update)

                    future_execution_time = future_predicted_time * in_execution_kernel["num_executions"] * future_job_remaining_percentage / in_execution_kernel["cu"]
                    # print("Kernel: ", in_execution_kernel["tmp_id"])
                    # print(feature)
                    # print(future_predicted_time)
                    # print(future_execution_time)

                    current_execution_time = in_execution_kernel["last_predicted_time"] * in_execution_kernel["num_executions"] * future_job_remaining_percentage / in_execution_kernel["cu"]

                    # print("current: ", current_execution_time)
                    # print("future: ", future_execution_time)

                    cu_acumulated_current_execution_time += current_execution_time
                    cu_acumulated_future_execution_time += future_execution_time

                cu_acumulated_interaction_time = (cu_acumulated_future_execution_time - cu_acumulated_current_execution_time) / cu_acumulated_current_execution_time

                # Update min cu option
                min_cu_option = cu_option if cu_acumulated_interaction_time < min_interaction_time else min_cu_option
                # Update the min kernel
                min_kernel = kernel if cu_acumulated_interaction_time < min_interaction_time else min_kernel
                # Check if the job time is the minimum
                min_interaction_time = min(cu_acumulated_interaction_time, min_interaction_time)

                # print("acumulated_current_execution_time", acumulated_current_execution_time)

        # Increase the active scheduling decisions
        if kernels_to_compare[0] != min_kernel or kernels_to_compare[0]["cu"] != min_cu_option:
            self.active_scheduling_decisions += 1
            # print("Decided to change")

        # print("Min kernel: ", min_kernel["tmp_id"])

        return min_kernel


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
    workload_0 = generate_workload(workload_information_dict, "ZCU")

    # Read workload information from files
    inter_arrival_values = []
    with open("workload/inter_arrival_1.bin", "rb") as file:
        while True:
            value = file.read(4)
            if not value:
                break
            inter_arrival_values.append(struct.unpack('f', value)[0])

    kernel_id_values = []
    with open("workload/kernel_id_1.bin", "rb") as file:
        while True:
            value = file.read(4)
            if not value:
                break
            kernel_id_values.append(struct.unpack('i', value)[0])

    num_executions_values = []
    with open("workload/num_executions_1.bin", "rb") as file:
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
    workload_1 = generate_workload(workload_information_dict, "ZCU")

    # Read workload information from files
    inter_arrival_values = []
    with open("workload/inter_arrival_2.bin", "rb") as file:
        while True:
            value = file.read(4)
            if not value:
                break
            inter_arrival_values.append(struct.unpack('f', value)[0])

    kernel_id_values = []
    with open("workload/kernel_id_2.bin", "rb") as file:
        while True:
            value = file.read(4)
            if not value:
                break
            kernel_id_values.append(struct.unpack('i', value)[0])

    num_executions_values = []
    with open("workload/num_executions_2.bin", "rb") as file:
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
    workload_2 = generate_workload(workload_information_dict, "ZCU")
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

    sim_0 = WorkloadSchedulingSimulation(online_models_list, workload_0, "CU", "ZCU")
    sim_1 = WorkloadSchedulingSimulation(online_models_list, workload_1, "CU", "ZCU")
    sim_2 = WorkloadSchedulingSimulation(online_models_list, workload_2, "CU", "ZCU")

    simulations = [sim_0, sim_1, sim_2]

    # Run simulation
    total_time = 0.0
    total_wait_time = 0.0
    total_decisions = 0
    affected_decisions = 0
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


if __name__ == "__main__":
    main()