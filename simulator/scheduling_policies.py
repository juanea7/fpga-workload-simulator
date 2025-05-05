"""
Scheduling Policies for Workload Simulation

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : May 2025
Description : This file contains functions for implementing different scheduling policies,

"""

def fist_come_first_served_policy(self):
    """
    A simple First-Come-First-Served (FCFS) scheduling policy.
    NOTE: Just one scheduling decision is made in each time step.
    """
    # TODO: Implement the scheduling algorithm

    for kernel in self.waiting_queue:
        # Check if the kernel can be scheduled (i.e., there are free slots available and the kernel has not been scheduled yet(artico3))
        if kernel["cu"] <= self.free_slots and self.current_configuration[self.kernel_names[kernel["kernel_id"]]] == 0:
            return kernel # Schedule only one kernel per time step

    # Indicate that there are no kernels that can be executed
    # Since all the schedulable kernels have been scheduled (wait for arrival of new kernels or finish of running kernels)
    self.are_kernels_executable = False
    return None

def stack_policy(self):
    """
    A scheduling policy that schedules the last kernel that arrived to the waiting queue.
    NOTE: Just one scheduling decision is made in each time step.
    """
    # TODO: Implement the scheduling algorithm

    for kernel in self.waiting_queue[::-1]:
        # Check if the kernel can be scheduled (i.e., there are free slots available and the kernel has not been scheduled yet(artico3))
        if kernel["cu"] <= self.free_slots and self.current_configuration[self.kernel_names[kernel["kernel_id"]]] == 0:
            return kernel # Schedule only one kernel per time step

    # Indicate that there are no kernels that can be executed
    # Since all the schedulable kernels have been scheduled (wait for arrival of new kernels or finish of running kernels)
    self.are_kernels_executable = False
    return None

def shortest_job_first_policy(self):
    """
    A scheduling policy that schedules the kernel with the shortest job time.
    NOTE: Just one scheduling decision is made in each time step.
    """
    # TODO: Implement the scheduling algorithm

    num_kernels_to_compare = 2
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

    for kernel in kernels_to_compare:
        # Create feature (TODO: Make CPU usage more realistic)
        feature = {**self.cpu_usage, **self.current_configuration}
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

def least_self_interaction_first_policy(self):
    """
    A scheduling policy that schedules the kernel with the least self interaction
    The interaction on other kernels is not considered.
    NOTE: Just one scheduling decision is made in each time step.
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
        feature = {**self.cpu_usage, **self.current_configuration}
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

def least_overall_interaction_first_policy(self):
    """
    A scheduling policy that schedules the kernel with the least overall interaction
    NOTE: Just one scheduling decision is made in each time step.
    """
    # TODO: Implement the scheduling algorithm

    num_kernels_to_compare = 2
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
    min_interaction_time = float('inf')
    min_kernel = None

    # print("hello\n")

    for kernel in kernels_to_compare:

        # Current configuration
        # Create feature (TODO: Make CPU usage more realistic)
        feature = {**self.cpu_usage, **self.current_configuration}
        tmp_kernel_name = self.kernel_names[kernel["kernel_id"]]
        feature[tmp_kernel_name] = kernel["cu"]

        # Acumulated execution time variables
        acumulated_future_execution_time = 0.0
        acumulated_current_execution_time = 0.0

        if not self.running_queue:
            # Schedule first kernel if no kernels are running # TODO: Remove
            return kernel

        # This loop is used to calculate the interaction the kernel to be scheduled with all the kernels in the running queue
        # The evaluation is done for each kernel in the running queue (i.e., kernel_in_execution)
        # assuming that the kernel to be schedule (kernel_to_schedule) is introduced.
        for in_execution_kernel in self.running_queue:

            # Create possible new feature (using as Main each kernel_in_execution)
            feature["Main"] = in_execution_kernel["kernel_id"]

            # Predict the execution time the kernel_in_execution when the kernel_to_schedule is scheduled
            future_predicted_time = round(self.models[-1].predict_one(feature), 3)

            # Calculate the job performed by the kernel_in_execution since the last update
            tmp_job_performed_since_last_update = (self.current_time - in_execution_kernel["last_update_time"]) / (in_execution_kernel["end_time"] - in_execution_kernel["last_update_time"])
            # Calculate the job remaining percentage of the kernel_in_execution
            future_job_remaining_percentage = in_execution_kernel["job_remaining_percentage"] * (1 - tmp_job_performed_since_last_update)

            # Calculate the execution time of the kernel_in_execution when the kernel_to_schedule IS scheduled
            future_execution_time = future_predicted_time * in_execution_kernel["num_executions"] * future_job_remaining_percentage / in_execution_kernel["cu"]
            # print("Kernel: ", in_execution_kernel["tmp_id"])
            # print(feature)
            # print(future_predicted_time)
            # print(future_execution_time)

            # Calculate the execution time of the kernel_in_execution when the kernel_to_schedule IS NOT scheduled
            current_execution_time = in_execution_kernel["last_predicted_time"] * in_execution_kernel["num_executions"] * future_job_remaining_percentage / in_execution_kernel["cu"]

            # print("current: ", current_execution_time)
            # print("future: ", future_execution_time)

            # Acumulate the execution time of each kernel_in_execution when the kernel_to_schedule is scheduled and when it is not scheduled
            acumulated_current_execution_time += current_execution_time
            acumulated_future_execution_time += future_execution_time

        # Calculate the interaction time of the kernel_to_schedule with all the kernels in the running queue
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

def cu_selection_policy(self):
    """
    A scheduling policy that selects the kernel with the least overall interaction.
    It also selects the amount of CUs.
    NOTE: Just one scheduling decision is made in each time step.
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

        # This loop evaluates every possible CU option for the kernel to be scheduled
        for cu_option in cu_available:

            # Current configuration
            # print("current")
            # Create feature (TODO: Make CPU usage more realistic)
            feature = {**self.cpu_usage, **self.current_configuration}
            tmp_kernel_name = self.kernel_names[kernel["kernel_id"]]
            feature[tmp_kernel_name] = cu_option

            cu_acumulated_future_execution_time = 0.0
            cu_acumulated_current_execution_time = 0.0

            if not self.running_queue:
                # TODO: solve this
                kernel["cu"] = cu_available[-1]
                self.tmp_kernel_default_count += 1
                return kernel # Schedule first kernel if no kernels are running # TODO: Remove

            # This loop is used to calculate the interaction the kernel to be scheduled with all the kernels in the running queue
            # The evaluation is done for each kernel in the running queue (i.e., kernel_in_execution)
            # assuming that the kernel to be schedule (kernel_to_schedule) is introduced.
            for in_execution_kernel in self.running_queue:

                # Create possible new feature (using as Main each kernel_in_execution)
                feature["Main"] = in_execution_kernel["kernel_id"]

                # Predict the execution time the kernel_in_execution when the kernel_to_schedule is scheduled
                future_predicted_time = round(self.models[-1].predict_one(feature), 3)

                # Calculate the job performed by the kernel_in_execution since the last update
                tmp_job_performed_since_last_update = (self.current_time - in_execution_kernel["last_update_time"]) / (in_execution_kernel["end_time"] - in_execution_kernel["last_update_time"])
                # Calculate the job remaining percentage of the kernel_in_execution
                future_job_remaining_percentage = in_execution_kernel["job_remaining_percentage"] * (1 - tmp_job_performed_since_last_update)

                # Calculate the execution time of the kernel_in_execution when the kernel_to_schedule IS scheduled
                future_execution_time = future_predicted_time * in_execution_kernel["num_executions"] * future_job_remaining_percentage / in_execution_kernel["cu"]
                # print("Kernel: ", in_execution_kernel["tmp_id"])
                # print(feature)
                # print(future_predicted_time)
                # print(future_execution_time)

                # Calculate the execution time of the kernel_in_execution when the kernel_to_schedule IS NOT scheduled
                current_execution_time = in_execution_kernel["last_predicted_time"] * in_execution_kernel["num_executions"] * future_job_remaining_percentage / in_execution_kernel["cu"]

                # print("current: ", current_execution_time)
                # print("future: ", future_execution_time)

                # Acumulate the execution time of each kernel_in_execution when the kernel_to_schedule is scheduled and when it is not scheduled
                cu_acumulated_current_execution_time += current_execution_time
                cu_acumulated_future_execution_time += future_execution_time

            # Calculate the interaction time of the kernel_to_schedule with all the kernels in the running queue
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
