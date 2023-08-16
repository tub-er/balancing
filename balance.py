# -*- coding: utf-8 -*-
# Written by Jeremias Hollnagel, TU Berlin, Department of Energy and Resource Management, 2020.

"""
***Balancing tool***

Used to account for differences in supply and demand in modelled energy flows and/or real data.

For each time step these imbalances can be passed onto specifically designated nodes (storage or pass-through)
and distributed among them.

To calculate the share of the imbalance each of the balancing nodes will account for, multiple methods can be chosen:
'equal_split' - meaning each balancing node will receive the same share of the imbalance (not recommended)
'pro_rata'    - each node will receive a share according to its size measured in available flow rates or maximum storage
                capacity (more accurate but requires additional knowledge of the properties of the nodes)

In order to keep any assigned flows within each node's technical limitations, e.g. flow direction, maximum capacity for
storages and available flow rate, flows can be checked against these limitations and adjusted accordingly. This way - if
possible within technical restrictions - a solution that removes any imbalances will always be found.
(If multiple solutions provide a balanced result, the tool will prioritize nodes by their order.)

For storage nodes the available flow rates can be adjusted after each time step based on current fill level in
accordance to each storage's individual flow rate and capacity characteristics ('Speicherkennlinien'). When using
storage nodes, each time step depends on the time step prior.
The storage fill level has to be initialized using the methods of this tool.
For storage nodes make sure corresponding entry and exit nodes belonging to the same storage are named accordingly
(storage_name + 'entry'/'exit'), so that the tool can keep an inventory on the storage.
Storage nodes that allow both entry and exit flows through a single node don't have such a requirement.

Contains functions to infer node characteristics (available flow rates, storage volumes and starting inventory)
from provided data empirically.

Use check_results() function to analyze the results returned by the balancing tool.
"""

# Libraries needed
import pandas as pd
import numpy as np
import time
import random
import pickle
from copy import deepcopy
from math import sin, pi
from datetime import date
import matplotlib.pyplot as plt
from pathlib import Path
import logging as log
from os.path import exists
from functools import reduce
 
log.basicConfig(filename='output/balancing.log', filemode='w', format='%(levelname)s: %(message)s', level=log.INFO)


class Balancing:
    """
    *Balancing object*
    Provides all necessary functions and keeps track of settings and technical restrictions of balancing nodes.
    """

    def __init__(self, settings=None, inv=None, flr=None, cap=None):
        """Initialize Balancing object with settings, inventory, available flow rates and storage volumes..."""
        # Settings
        # use change_settings(settings_dict) to update the settings of Balancing instance
        if settings is None:
            settings = {}
        if inv is None:
            inv = {}
        if flr is None:
            flr = {}
        if cap is None:
            cap = {}
        if not settings:
            self.settings = {
                'method': 'equal_split',
                'share_by': 'flowrate',
                'storage_limit': False,
                'flowrate_limit': False,
                'dynamic_flowrate': False
            }
        else:
            self.settings = settings
        self.inventory = inv  # used to save the current fill level of each storage node
        self.available_flowrates = flr  # flow rates for each balancing node available at the moment
        self.storage_volume = cap  # min fill level and max capacity of each storage node (if using storage AND
        # pass-through nodes set pass-through nodes capacity to infinite or 0 – depending on the model)
        self.available_flowrates_original = flr  # needed for dynamic flow rates
        self.critical_level = 0.5  # critical level

    # ---------- Setup and accessibility functions ---------- #
    def info(self):
        """Show info on available and current settings"""
        info_dict = {
            'method':
                "\n\tTwo methods available to calculate the balancing: "
                "\n\tequal_split - every balancing node is allocated the same share of the imbalance, "
                "\n\tpro_rata - shares are calculated using available capacity or flow rate of each balancing node.\n",
            'share_by':
                "\n\tCalculate shares based on available 'flowrate' or 'capacity' (for pro_rata method only).\n",
            'storage_limit':
                "\n\tIf 'True' minimum and maximum volumes of the balancing nodes will not be exceeded (storages).\n",
            'flowrate_limit':
                "\n\tIf 'True' minimum and maximum flow rates will not be exceeded.\n",
            'dynamic_flowrate':
                "\n\tIf 'True' currently available flow rates will be calculated based on current fill level.\n"
        }
        print('Info:')
        for key, value in info_dict.items():
            print('- ' + key + ':', value)
        print('Current settings:\n', self.settings)
        print("\nUse 'change_settings(settings_dict)' to update the settings.")
        print("\nUse 'show_variables()' to see all global variables in their current state.")

    def show_variables(self):
        """Show all global variables in their current state"""
        print('Min/max available flow rates:')
        print(self.available_flowrates)
        print('\n')
        print('Min/max storage volumes:')
        print(self.storage_volume)
        print('\n')
        print('Current fill levels:')
        print(self.inventory)

    def change_settings(self, settings_dict):
        """adjust the global settings"""
        self.settings.update(settings_dict)
        # log.info('Settings:\t %s', self.settings)

    def set_storage_volume(self, capacities):
        """set the capacities (min&max) for each node by passing a dictionary"""
        self.storage_volume = capacities

    @staticmethod
    def infer_storage_volume(unbalanced_df, balancing_nodes, perc_added=0):
        """Infer capacity as the range between maximum and minimum of the accumulated flows"""
        log.info('Inferring capacities based on the difference between the maximum and minimum of the accumulated in- '
                 'and outflows throughout the year (plus %.1f%%).', perc_added * 100)
        storage_volume_dict = {}
        for node in list(set([node.replace('entry', '').replace('exit', '') for node in balancing_nodes])):
            # for each storage sum up injection and withdrawal
            try:
                series = unbalanced_df[node + 'entry'] + unbalanced_df[node + 'exit']
            except KeyError:
                # handle edge cases (missing nodes)
                try:
                    series = unbalanced_df[node + 'entry']
                except KeyError:
                    try:
                        series = unbalanced_df[node + 'exit']
                    except KeyError:
                        series = unbalanced_df[node]
            # accumulate flows and get min and max
            maximum = series.cumsum().max()
            minimum = series.cumsum().min()
            # storage volume will be the difference between min and max times turnover rate (perc added)
            _range = maximum - minimum
            storage_volume_dict.update({node: {'min': 0, 'max': _range * (1 + perc_added)}})
        return storage_volume_dict
        # instead of using the range we can use min and max respectively. The problem here is, that we cant just add
        # a percentage easily if we want to make the storage a bit bigger with the min/max storage we can just set the
        # inventory at the start to 0 and should have no problems with exceeding storage limits

    def set_available_flowrates(self, flowrates):
        """set the min&max flow rate for each balancing node by passing a dictionary"""
        self.available_flowrates = flowrates

    @staticmethod
    def infer_available_flowrates(unbalanced_df, balancing_nodes, perc_added=0):
        """infer the flow rate cap from the balancing nodes using their min/max flow rate over the time series"""
        log.info('Inferring min and max flow rates based on flow rates in the given data.')
        # get min/max values of all nodes
        pos_limits = dict(unbalanced_df.max())
        neg_limits = dict(unbalanced_df.min())
        flowrates = {}
        # set them as max. flow rates plus/minus percentage
        for node in list(set(balancing_nodes).intersection(set(unbalanced_df.columns))):
            flowrates.update(
                {node: {'min': neg_limits[node] * (1 + perc_added), 'max': pos_limits[node] * (1 + perc_added)}})
        return flowrates

    def set_inventory(self, level=None):
        """Set the initial fill level of the storage (0-100% of capacity) level should always be between 0 and 1
        and STORAGE CAPACITY SHOULD BE SET BEFORE"""
        # manually set the starting fill levels by passing a dictionary of fill levels for the balancing nodes
        if isinstance(level, dict):
            self.inventory = level
        # Otherwise pass a percentage 0-100% of available capacity (above min) that will be applied to all nodes
        elif isinstance(level, float) and 0 < level < 1:
            unique_storage = [node for node in self.storage_volume.keys()]
            self.inventory = {node: self.storage_volume[node]['min'] + (
                    self.storage_volume[node]['max'] - self.storage_volume[node]['min']) * level for node in
                              unique_storage}
        else:
            log.error('Pass a dictionary of fill levels or a percentage (0-1) the levels should be set to...')

    @staticmethod
    def infer_inventory(unbalanced_df, balancing_nodes, perc_added=0):
        """infer the starting inventory of each storage node (only after inferring storage volumes)"""
        log.info('Inferring inventory at the beginning from given dataframe')
        inv_dict = {}
        # sum up flows for all storages cumulatively
        for node in list(set([node.replace('entry', '').replace('exit', '') for node in balancing_nodes])):
            try:
                series = unbalanced_df[node + 'entry'] + unbalanced_df[node + 'exit']
            # handling edge cases (missing nodes)
            except KeyError:
                try:
                    series = unbalanced_df[node + 'entry']
                except KeyError:
                    try:
                        series = unbalanced_df[node + 'exit']
                    except KeyError:
                        series = unbalanced_df[node]
            # set starting inventory to abs(min) * (1 + perc added / 2)
            minimum = series.cumsum().min()
            inv_dict[node] = abs(minimum) * (1 + perc_added / 2)
        return inv_dict

    def infer_inventory_time(self, gas_flows_df):
        """Infer the current fill level for time step 0, using the time of the year / date as an indicator.
        DatetimeIndex has to be provided.
        - only for gas storage
        - (approx. sine curve with max at Oct. 15th and min at April 15th)"""
        d = gas_flows_df.index[0]
        if isinstance(d, pd.Timestamp):
            d1 = d.to_pydatetime().date()
            # d0 = date(d.year, 10, 15) # if net2stroage defined as negative
            d0 = date(d.year, 4, 15)  # if net2storage defined as positive
            delta = (d0 - d1).days
            perc_full = abs(sin(self._linear_mapping(delta, 0, 365, 0, pi) - pi / 2))
            perc_full = self._linear_mapping(perc_full, 0, 1, 0, 1)
            # when using inferred capacity data set to 0 and 100 otherwise a preferred percentage
            log.info('Inferring fill levels using starting dates and maximum capacities.')
            log.info('Levels will be set to %.1f%% based on starting date %s.', perc_full, d1)
            return perc_full
        else:
            log.warning(
                'For inferring the fill level, a DatetimeIndex must be provided. Setting fill level to half full.')
            return 0.5

    def set_critical_level(self, level):
        """Set the critical level of each storage (can be float or dict of storage:float between 0 and 1)"""
        self.critical_level = level

    @staticmethod
    def check_inputs(unbalanced_df, balancing_nodes):
        """check whether all inputs are in the correct format to always ensure useful and predictable results"""
        # Catching invalid arguments
        if not isinstance(unbalanced_df, pd.DataFrame):
            log.error("'unbalanced_df' must be of type dataframe, not %r", type(unbalanced_df))
            return False
        if not isinstance(balancing_nodes, list):
            log.error("'balancing_nodes' must be of type list, not %r", type(balancing_nodes))
            return False
        if not balancing_nodes:
            log.error('List of balancing nodes is empty.')
            return False
        if unbalanced_df.empty:
            log.error('Dataframe is empty.')
            return False
        # Checking for duplicate indices in the dataframe
        duplicates = unbalanced_df.index.duplicated(keep='first')
        if any(duplicates):
            log.error('The following dataframe indices are duplicated. Remove / reindex and try again.\n %s',
                      list(unbalanced_df[duplicates].index))
            return False
        # Checking for duplicate column names in the df
        if len(set(unbalanced_df.columns)) != len(unbalanced_df.columns):
            log.error('There are duplicated column names in the dataframe. Remove / rename and try again.')
            return False
        # Checking for nodes that are not in the dataframe
        missing = set(balancing_nodes) - set(unbalanced_df.columns)
        if missing:
            log.warning('The following nodes from the balancing nodes list are not columns of the dataframe.\n %s '
                        '\nUsing only balancing nodes that are actually in the dataframe...', missing)
        return True

    # ---------- Main 'wrapper' function to call as user ---------- #

    def balance(self, unbalanced_df, balancing_nodes):
        """
        Function to balance a dataframe using only nodes specified in a list.
        - unbalanced_df: pandas.DataFrame of format: (rows = time steps, columns = nodes)
        - balancing_nodes: list with the names of the nodes to be used as the balancing nodes
                           (all other nodes won't be adjusted)
        """
        print('Balancing...')
        log.info('Settings:\n %s', self.settings)
        # check if all inputs are fine
        if not self.check_inputs(unbalanced_df, balancing_nodes):
            print('Input Error: See balancing log for details')
            return None
        # get rid of duplicates and non numeric values
        balanced_df = unbalanced_df.astype(float).fillna(0).copy()
        balancing_nodes = list(set(balancing_nodes).intersection(set(balanced_df.columns)))
        # combine all none balancing nodes for better performance
        b_nodes_df = balanced_df[balancing_nodes]
        rest = balanced_df[list(set(balanced_df.columns) - set(balancing_nodes))]
        combined = rest.sum(axis=1)
        combined.name = 'combined'
        balanced_df = b_nodes_df.join(combined)
        # build a dict from the data for better performance
        df_dictionary = balanced_df.to_dict(orient='records')
        # for keeping track of dynamic flworates we need a dictionary containing the original flowrates
        if self.settings['dynamic_flowrate']:
            self.available_flowrates_original = deepcopy(self.available_flowrates)
        # time tracking
        time_start = time.perf_counter()
        time_array = np.array([])
        # enumerate the rows of the dataframe
        for idx, row in enumerate(balanced_df.index):
            # time tracking
            start_time = time.perf_counter()
            # make a copy of the balancing nodes to make sure they arent changed
            nodes = deepcopy(balancing_nodes)
            # taking the current row from the dict
            flows = df_dictionary[idx]
            log.debug('row: %s, %s', idx, row)
            log.debug('\t%s', flows)
            # getting standard adjustments
            adjustments = self._standard_adjustment(balancing_nodes, sum(flows.values()))
            # checking adjustments against restrictions
            if self.settings['flowrate_limit'] or self.settings['storage_limit']:
                adjustments = self._check_restrictions(adjustments, flows, {}, nodes)
            # getting the flows after subtracting the calculated adjustments
            adjusted_flows = self._adjust(adjustments, flows)
            # updating the fill levels after the adjustments
            if self.settings['storage_limit'] or self.settings['dynamic_flowrate']:
                self._update_inventory(adjusted_flows, nodes)
            # updating the maximum flow rates according to the new fill levels
            if self.settings['dynamic_flowrate']:
                self._update_flowrates(nodes)
            # applying the changes to the current row of the dataframe
            for node in adjusted_flows.keys():
                balanced_df.loc[row, node] = adjusted_flows[node]  # making the changes to the row of the df
            # checking for duplicate flows (have to do this after applying changes for all nodes!
            if self.storage_volume or self.inventory:  # only if we are using storages!
                for ugs in [node.replace('entry', '').replace('exit', '') for node in adjusted_flows.keys()]:
                    try:  # only do this if both 'entry' and 'exit' nodes are found
                        if abs(balanced_df.loc[row, ugs + 'exit']) > abs(balanced_df.loc[row, ugs + 'entry']):
                            balanced_df.loc[row, ugs + 'exit'] += balanced_df.loc[row, ugs + 'entry']
                            balanced_df.loc[row, ugs + 'entry'] = 0
                        elif abs(balanced_df.loc[row, ugs + 'exit']) < abs(balanced_df.loc[row, ugs + 'entry']):
                            balanced_df.loc[row, ugs + 'entry'] += balanced_df.loc[row, ugs + 'exit']
                            balanced_df.loc[row, ugs + 'exit'] = 0
                    except KeyError:
                        pass
            log.debug('\t%s', adjusted_flows)
            # time tracking
            elapsed_time = time.perf_counter() - start_time
            time_array = np.append(time_array, elapsed_time)
            remaining_time = np.mean(time_array) * (len(balanced_df.index) - idx)
            if remaining_time > 60:
                print('remaining time: %.2fmin      ' % (remaining_time / 60), end='\r', flush=True)
            else:
                print('remaining time: %.1fs        ' % remaining_time, end='\r', flush=True)
        # now replace the 'combined' node with the actual data again
        balanced_df = balanced_df[balancing_nodes].join(rest)
        # print the completion time
        time_end = time.perf_counter()
        if time_end - time_start > 60:
            print('Finished after %.0fmin %.0fs     ' % ((time_end - time_start) / 60, (time_end - time_start) % 60))
        else:
            print('Finished after %.1fs           ' % (time_end - time_start))
        # return the results after all rows have been balanced
        return balanced_df

    # ---------- Main functions needed in the balancing ---------- #

    # def imbalance(unbalanced_series, changes): # one-liner that doesn't need a function (except maybe for debugging)
    #    '''calculates the sum of a series of flows (in dict format) and subtracts changes value
    #    (used to keep track of adjustments already done)'''
    #    log.debug('\tchanges: %s', changes)
    #    log.debug('\timbalance: %s', sum(unbalanced_series.values()) - changes)
    #    return sum(unbalanced_series.values()) - changes

    def _standard_adjustment(self, balancing_nodes, imbalance):
        """ Calculate the adjustments needed to balance the dataframe row.
        equal_split: imbalance / number of balancing nodes for all balancing nodes
        pro_rate: imbalance * share of balancing node for all balancing nodes
        """
        if self.settings['method'] == 'pro_rata':
            # Calculate the share for each node and then the adjustments accordingly
            share = self._share(balancing_nodes)
            adjustments = {node: share[node] * imbalance for node in balancing_nodes}
            log.debug('\tproposed adjustments: %s', adjustments)
        elif self.settings['method'] == 'equal_split':
            # Calculate the adjustments with every node receiving the same share
            adjustments = {node: imbalance / len(balancing_nodes) for node in balancing_nodes}
            log.debug('\tproposed adjustments: %s', adjustments)
        elif self.settings['method'] == 'pipe_capacity':
            share = self._share(balancing_nodes)
            adjustments = {node: share[node] * imbalance for node in balancing_nodes}
            log.debug('\tproposed adjustments: %s', adjustments)
        else:
            log.error(f"Invalid setting '{str(self.settings['method'])}'.")
            return None
        return adjustments

    def _check_restrictions(self, adjustments, unbalanced_series, changes, balancing_nodes):
        """ check if any restrictions are exceeded, if so adjust adjustments and go back to step 1 for the rest of the
        nodes """
        # make a deep copy before updating
        adjustments_old = deepcopy(adjustments)
        adjustable_nodes = deepcopy(balancing_nodes)
        # check whether levels will stay within their capacities and adjust accordingly
        adjustments_new, changes, used_nodes1 = self._check_storage_capacity(adjustments, unbalanced_series,
                                                                             adjustable_nodes, changes)
        # check whether flow rates will stay within their limits and adjust accordingly
        adjustments_new, changes, used_nodes2 = self._check_avb_flowrates(adjustments_new, unbalanced_series,
                                                                          adjustable_nodes, changes)
        # get a list of unique nodes that were changed
        used_nodes = list(set(used_nodes1 + used_nodes2))
        # get the sum of the changes made
        sum_of_changes = sum(changes.values())
        # check whether any changes had to be made
        if adjustments_old == adjustments_new:
            log.debug('\toptimal solution found')
            return adjustments_new
        # remove the used_nodes from the list of adjustable nodes
        for node in used_nodes:
            adjustable_nodes.remove(node)
        # check whether all adjustable nodes have already been changed
        if not adjustable_nodes:
            log.debug('\tsuboptimal solution found')
            return adjustments_new
        # if none of the escape conditions are met, redo the adjustments for the left nodes
        log.debug('\tBalancing again using %s', adjustable_nodes)
        adjustments_new.update(
            self._standard_adjustment(adjustable_nodes, sum(unbalanced_series.values()) - sum_of_changes))
        # and check the restrictions again
        return self._check_restrictions(adjustments_new, unbalanced_series, changes, adjustable_nodes)

    def _check_storage_capacity(self, adjustments, unbalanced_series, adjustable_nodes, changes):
        """for each node, check if the fill level after adjustment would be within its capacity,
        return True or changed adjustments dict"""
        used_nodes = []
        if self.settings['storage_limit']:
            # get would-be inventory after balancing
            would_be_inventory = self._check_inventory(unbalanced_series, adjustments)
            current_inventory = deepcopy(self.inventory)
            # get list of unique storages
            unique_nodes = list(set([node.replace('entry', '').replace('exit', '') for node in adjustable_nodes]))
            for node in unique_nodes:
                # for each storage
                if would_be_inventory[node] > self.storage_volume[node]['max']:
                    # if the inventory after adjustment would be higher than allowed inventory
                    log.debug('\texceeding max storage_volume')
                    if node + 'entry' in adjustable_nodes and node + 'exit' in adjustable_nodes:
                        # get the difference between max and current
                        difference = current_inventory[node] + unbalanced_series[node + 'entry'] + unbalanced_series[
                            node + 'exit'] - self.storage_volume[node]['max']
                        # set that as adjustment, using share function
                        share = self._share([node + 'entry', node + 'exit'])
                        adjustments[node + 'entry'] = share[node + 'entry'] * difference
                        adjustments[node + 'exit'] = share[node + 'exit'] * difference
                        # add calculated adjustments to changes dict
                        changes[node + 'entry'] = adjustments[node + 'entry']
                        changes[node + 'exit'] = adjustments[node + 'exit']
                        # and place adjusted nodes on to-remove list
                        used_nodes.append(node + 'entry')
                        used_nodes.append(node + 'exit')
                    # edge cases for missing entry or exit nodes:
                    elif node + 'entry' in adjustable_nodes:
                        difference = current_inventory[node] + unbalanced_series[node + 'entry'] - \
                                     self.storage_volume[node]['max']
                        adjustments[node + 'entry'] = difference
                        changes[node + 'entry'] = adjustments[node + 'entry']
                        used_nodes.append(node + 'entry')
                    elif node + 'exit' in adjustable_nodes:
                        difference = current_inventory[node] + unbalanced_series[node + 'exit'] - \
                                     self.storage_volume[node]['max']
                        adjustments[node + 'exit'] = difference
                        changes[node + 'exit'] = adjustments[node + 'exit']
                        used_nodes.append(node + 'exit')
                    else:
                        difference = current_inventory[node] + unbalanced_series[node] - self.storage_volume[node][
                            'max']
                        adjustments[node] = difference
                        changes[node] = adjustments[node]
                        used_nodes.append(node)
                # If inventory after adjustments would be below min storage level...
                elif would_be_inventory[node] < self.storage_volume[node]['min']:
                    log.debug('\texceeding min storage_volume')
                    # get the difference between minimum and current and set it as adjustment, then same steps as above
                    if node + 'entry' in adjustable_nodes and node + 'exit' in adjustable_nodes:
                        difference = current_inventory[node] + unbalanced_series[node + 'entry'] + unbalanced_series[
                            node + 'exit'] - self.storage_volume[node]['min']
                        share = self._share([node + 'entry', node + 'exit'])
                        adjustments[node + 'entry'] = share[node + 'entry'] * difference
                        adjustments[node + 'exit'] = share[node + 'exit'] * difference
                        changes[node + 'entry'] = adjustments[node + 'entry']
                        changes[node + 'exit'] = adjustments[node + 'exit']
                        used_nodes.append(node + 'entry')
                        used_nodes.append(node + 'exit')
                    # edge cases:
                    elif node + 'entry' in adjustable_nodes:
                        difference = current_inventory[node] + unbalanced_series[node + 'entry'] - \
                                     self.storage_volume[node]['min']
                        adjustments[node + 'entry'] = difference
                        changes[node + 'entry'] = adjustments[node + 'entry']
                        used_nodes.append(node + 'entry')
                    elif node + 'exit' in adjustable_nodes:
                        difference = current_inventory[node] + unbalanced_series[node + 'exit'] - \
                                     self.storage_volume[node]['min']
                        adjustments[node + 'exit'] = difference
                        changes[node + 'exit'] = adjustments[node + 'exit']
                        used_nodes.append(node + 'exit')
                    else:
                        difference = current_inventory[node] + unbalanced_series[node] - self.storage_volume[node][
                            'min']
                        adjustments[node] = difference
                        changes[node] = adjustments[node]
                        used_nodes.append(node)
            if used_nodes:
                log.debug('\tadjustments changed (storage_volume): %s', adjustments)
        return adjustments, changes, used_nodes

    def _check_avb_flowrates(self, adjustments, unbalanced_series, adjustable_nodes, changes):
        """for each node, check if the adjustment would be within its max flow rate,
        return changed adjustments dict, sum of changes made and list of nodes changed"""
        used_nodes = []
        if self.settings['flowrate_limit']:
            would_be_flows = self._adjust(adjustments, unbalanced_series)
            for node in adjustable_nodes:
                # get the min and max flow rates
                # (either one would normally be zero - unless a node can be both pos. and neg.)
                max_flow_rate = self.available_flowrates[node]['max']
                min_flow_rate = self.available_flowrates[node]['min']
                # flows that exceed the minimum flow rate: usually negative because min flow rate is either 0 or < 0
                if would_be_flows[node] < min_flow_rate:
                    log.debug('\tavailable_flowrates min exceeded')
                    # get the difference between current flow and minimum flow rate and set the adjustment to that
                    adjustments[node] = unbalanced_series[node] - min_flow_rate
                    changes[node] = adjustments[node]
                    used_nodes.append(node)
                elif would_be_flows[node] > max_flow_rate:  # flows that exceed the maximum flow rate - usually positive
                    log.debug('\tavailable_flowrates max exceeded')
                    # get the difference between max flow rate and current flow rate and set the adjustment to that
                    adjustments[node] = unbalanced_series[node] - max_flow_rate
                    changes[node] = adjustments[node]
                    used_nodes.append(node)
            if used_nodes:
                log.debug('\tadjustments changed (available_flowrates): %s', adjustments)
        return adjustments, changes, used_nodes

    # ---------- Helper functions needed throughout ---------- #

    @staticmethod
    def _adjust(adjustments, unbalanced_series):
        """get flows of balancing nodes after subtracting adjustments"""
        flows = deepcopy(unbalanced_series)
        for node in adjustments.keys():
            flows[node] = unbalanced_series[node] - adjustments[node]
        log.debug('\tflows after adjustments: %s', flows)
        return flows

    def _share(self, balancing_nodes):
        """ Needed for pro rata method: Calculate the share of each balancing node in the balancing
        according to its available capacity or flow rates """
        share = {}
        if self.settings['share_by'] == 'pipe_capacity':
            
            #print(DE_POINT_WEIGHT[balancing_nodes])
            
            s = DE_POINT_WEIGHT[balancing_nodes].sum()
            
            return {node: w/s for node, w in DE_POINT_WEIGHT[balancing_nodes].iteritems()}
        if self.settings['share_by'] == 'flowrate':
            avb_flowrates = deepcopy(self.available_flowrates)
            # the currently implemented method is using the absolute max available flowrate. 
            # another possible implementation would be using the difference between flow and max...
            flowrate_max = {node: max(abs(value['max']), abs(value['min'])) for node, value in avb_flowrates.items()
                            if node in balancing_nodes}  # changed to use absolute max instead of range!
            s = sum(flowrate_max.values())
            if s == 0:  # Catch division by zero errors!
                for node in balancing_nodes:
                    share[node] = 0
            else:
                for node in balancing_nodes:
                    share[node] = abs(flowrate_max[node]) / s
        log.debug('\tshare dict %s', share)
        return share

    def _check_inventory(self, unbalanced_series, adjustments):
        """Get the current fill levels, subtract the adjustments and return a dictionary containing the results,
        without adjusting the global fill levels"""
        inv = deepcopy(self.inventory)
        would_be_flows = self._adjust(adjustments, unbalanced_series)
        unique_storage = list(set([node.replace('entry', '').replace('exit', '') for node in adjustments.keys()]))
        for node in unique_storage:
            try:
                inv[node] += would_be_flows[node + 'entry']
            except KeyError:
                pass
            try:
                inv[node] += would_be_flows[node + 'exit']
            except KeyError:
                pass
        return inv

    def _update_inventory(self, flows, balancing_nodes):
        """update the global dictionary containing the current fill levels of the storage nodes"""
        unique_storage = list(set([node.replace('entry', '').replace('exit', '') for node in balancing_nodes]))
        # add up injection and withdrawal for each storage
        for node in unique_storage:
            try:
                self.inventory[node] += flows[node + 'entry']
            except KeyError:  # handle edge cases
                log.warning('\tNo node found for %s while trying to update inventory.', node + 'entry')
            try:
                self.inventory[node] += flows[node + 'exit']
            except KeyError:  # handle edge cases
                log.warning('\tNo node found for %s while trying to update inventory.', node + 'exit')
        log.debug('\tLevel after updating: %s', self.inventory)

    @staticmethod
    def _linear_mapping(x, a1, b1, a2, b2):
        """mapping a number x in a range of numbers a1, a2 to a number in another range of numbers b1, b2"""
        return (x - a1) / (b1 - a1) * (b2 - a2) + a2

    def _update_flowrates(self, nodes):
        """Update current min / max available flow rates (injection/withdrawal rates)
        for storage nodes based on their fill level"""
        # storage inventory has to be adjusted beforehand!!
        # critical_level should never be set to 0 or 1!
        c_level = 1
        for node in nodes:
            storage = node.replace('entry', '').replace('exit', '')
            if isinstance(self.critical_level, dict):
                try:
                    c_level = self.critical_level[storage]
                except KeyError:
                    log.warning("Couldn't find critical level for %s" % storage)
            elif isinstance(self.critical_level, float) and 0 <= self.critical_level <= 1:
                c_level = self.critical_level
            # if we have a negative node
            if self.available_flowrates_original[node]['min'] <= 0 \
                    and self.available_flowrates_original[node]['max'] <= 0:
                # making sure no division by zero error if min and max storage is the same
                if self.storage_volume[storage]['min'] != self.storage_volume[storage]['max']:
                    level = (self.inventory[storage] - self.storage_volume[storage]['min']) / \
                            (self.storage_volume[storage]['max'] - self.storage_volume[storage]['min'])
                    if level < 1 - c_level:
                        flr = self._linear_mapping(level, 0, c_level, self.available_flowrates_original[node]['max'],
                                                   self.available_flowrates_original[node]['min'])
                        self.available_flowrates[node]['min'] = flr  # adjust the minimum flow rate accordingly
                    else:
                        self.available_flowrates[node]['min'] = self.available_flowrates_original[node]['min']
                else:
                    self.available_flowrates.update({node: {'min': 0, 'max': 0}})
            # otherwise, if we have a positive node
            elif self.available_flowrates_original[node]['min'] >= 0 \
                    and self.available_flowrates_original[node]['max'] >= 0:
                if self.storage_volume[storage]['min'] != self.storage_volume[storage]['max']:
                    level = (self.inventory[storage] - self.storage_volume[storage]['min']) / (
                            self.storage_volume[storage]['max'] - self.storage_volume[storage]['min'])
                    if level > c_level:
                        flr = self._linear_mapping(level, c_level, 1, self.available_flowrates_original[node]['max'],
                                                   self.available_flowrates_original[node]['min'])
                        self.available_flowrates[node]['max'] = flr  # adjust the maximum flow rate accordingly
                    else:
                        self.available_flowrates[node]['max'] = self.available_flowrates_original[node]['max']
                else:
                    self.available_flowrates.update({node: {'min': 0, 'max': 0}})
            else:
                log.warning('\nno dynamic balancing available for nodes not seperated by entry and exit %s' % node)

    def check_results(self, balanced_df, unbalanced_df, balancing_nodes, number_of_graphs=5, perc_added=0,
                      unit='kWh/h', inventory_by_date=False, save_fig=False):
        """This method can be used to take a closer look at the results and compare unbalanced and balanced data."""
        if save_fig:
            Path("graphs").mkdir(exist_ok=True)
            figname = 'graphs/'
            figname += self.settings['method']
            if self.settings['flowrate_limit']:
                figname += 'flr_lim_'
            if self.settings['storage_limit']:
                figname += 'str_lim_'
            if self.settings['dynamic_flowrate']:
                figname += 'dyn_flr_'
            i = 0

        if len(balancing_nodes) < number_of_graphs:
            number_of_graphs = len(balancing_nodes)
        figsize = (20, 4)
        lw = 0.8  # line width for minima and maxima
        ls = '--'  # line style for minima and maxima
        if len(balanced_df) > 1000:
            s = 0.5  # marker size for scatter plots of flows
        elif len(balanced_df) <= 50:
            s = 20
        else:
            s = 10
        nodes = list(set([node.replace('entry', '').replace('exit', '') for node in balancing_nodes]))
        if number_of_graphs < len(nodes):
            storage_nodes = random.sample(nodes, number_of_graphs)
        else:
            storage_nodes = nodes
        b_nodes = []
        for node in storage_nodes:
            if node + 'entry' in balanced_df.columns:
                b_nodes.append(node + 'entry')
            if node + 'exit' in balanced_df.columns:
                b_nodes.append(node + 'exit')
            if node + 'entry' not in balanced_df.columns and node + 'exit' not in balanced_df.columns:
                b_nodes.append(node)
        plt.figure(figsize=(figsize[0], figsize[1] * 1.5))  # slightly bigger figsize for imbalance
        plt.scatter(unbalanced_df.index, list(unbalanced_df.sum(axis=1)), color='blue', label='unbalanced', s=s)
        plt.scatter(unbalanced_df.index, list(balanced_df.sum(axis=1)), color='orange', label='balanced', s=s)
        plt.title('Sum of flows before and after balancing')
        plt.ylabel(unit)
        plt.legend()
        if save_fig:
            plt.savefig(figname + str(i) + '.png', bbox_inches='tight', pad_inches=0.1)
            i += 1
        else:
            plt.show()
        try:
            unb_ts = [hour.strftime('%Y/%m/%d %H:00') for hour, value in balanced_df.sum(axis=1).iteritems()
                      if round(value) != 0]
        except Exception:  # there is no DatetimeIndex...
            unb_ts = [ind for ind, value in balanced_df.sum(axis=1).iteritems() if round(value) != 0]
        if len(unb_ts) <= 10:
            print('Time steps left unbalanced: %s' % unb_ts)
        else:
            print('Time steps left unbalanced: %s' % len(unb_ts))
            log.info('Time steps left unbalanced: %s' % unb_ts)
        if self.available_flowrates and not self.settings['dynamic_flowrate']:  # and self.settings['flowrate_limit']:
            #print('Plotting comparison of flows after balancing and minimum/maximum flow rates available...')
            if len(balancing_nodes) > 40:
                n = random.sample(balancing_nodes, 40)
                print('Using 40 randomly selected nodes...')
            else:
                n = balancing_nodes
            # get min/max available flowrates and min/max /
            min_avb = [self.available_flowrates[node]['min'] for node in n]
            max_avb = [self.available_flowrates[node]['max'] for node in n]
            min_flw = [balanced_df[node].min() for node in n]
            max_flw = [balanced_df[node].max() for node in n]
            avg_flw_pos = [balanced_df[node].mean() for node in n]
            # avg_flw_neg = [balanced_df[node].mean() for node in n]
            x = [node.replace('entry', ' entry').replace('exit', ' exit') for node in n]
            # plot them all in one graph with nice colors
            plt.figure(figsize=figsize)
            plt.scatter(x, max_avb, color='red', marker='_', label='maximum available flow rates')
            plt.scatter(x, max_flw, color='blue', marker='^', label='maximum flow rates reached')
            plt.scatter(x, avg_flw_pos, color='blue', marker='.', label='average flow rates')
            plt.scatter(x, min_flw, color='blue', marker='v', label='minimum flow rates reached')
            plt.scatter(x, min_avb, color='red', marker='_', label='minimum available flow rates')
            # plt.plot(x, [0]*len(x), color = 'black', lw=lw)
            plt.xticks(rotation=90)
            plt.title('Comparison of minimum/maximum available flow rates and minimum/maximum reached flow rates')
            plt.legend()
            plt.ylabel(unit)
            if save_fig:
                plt.savefig(figname + str(i) + '.png', bbox_inches='tight', pad_inches=0.1)
                plt.close()
                i += 1
            else:
                plt.show()
        if self.storage_volume:
            #print('\nPlotting storage inventory throughout the year for selected nodes before and after balancing...')
            # get the starting inventory
            if inventory_by_date:
                lvl = self.infer_inventory_time(unbalanced_df)
                inv_start = {node: self.storage_volume[node]['min'] + (
                        self.storage_volume[node]['max'] - self.storage_volume[node]['min']) * lvl for node in
                             self.storage_volume.keys()}
            else:
                inv_start = self.infer_inventory(unbalanced_df, balancing_nodes, perc_added)
            for node in storage_nodes:
                try:
                    ts1 = unbalanced_df[node + 'entry'] + unbalanced_df[node + 'exit']
                    ts2 = balanced_df[node + 'entry'] + balanced_df[node + 'exit']
                except KeyError:
                    try:
                        ts1 = deepcopy(unbalanced_df[node + 'entry'])
                        ts2 = deepcopy(balanced_df[node + 'entry'])
                    except KeyError:
                        try:
                            ts1 = deepcopy(unbalanced_df[node + 'exit'])
                            ts2 = deepcopy(balanced_df[node + 'exit'])
                        except KeyError:
                            ts1 = deepcopy(unbalanced_df[node])
                            ts2 = deepcopy(balanced_df[node])
                # start the cumulative sum at the inventory when starting
                ts1.iloc[0] += inv_start[node]
                ts2.iloc[0] += inv_start[node]
                _max = [self.storage_volume[node]['max']] * len(unbalanced_df.index)
                _min = [self.storage_volume[node]['min']] * len(unbalanced_df.index)
                plt.figure(figsize=figsize)
                plt.plot(balanced_df.index, _max, label='full', ls=ls, lw=lw, color='red')
                plt.plot(ts1.cumsum(), label='unbalanced', color='blue', lw=lw)
                plt.plot(ts2.cumsum(), label='balanced', color='orange', lw=lw)
                # plt.plot(ts1.cumsum()-ts2.cumsum(), label='difference', color = 'black', ls ='--', lw=lw)
                plt.plot(balanced_df.index, _min, label='empty', ls=ls, lw=lw, color='red')
                plt.legend()
                plt.ylabel(unit[:3])
                plt.title(node)
                if save_fig:
                    plt.savefig(figname + str(i) + '.png', bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    i += 1
                else:
                    plt.show()
        # add a few lines to show the actual flows of selected nodes before and after balancing
        if self.available_flowrates_original:
            #print('\nPlotting flows throughout the year for selected nodes before and after balancing...')
            for node in b_nodes:
                plt.figure(figsize=figsize)
                _max = [self.available_flowrates_original[node]['max']] * len(balanced_df.index)
                _min = [self.available_flowrates_original[node]['min']] * len(balanced_df.index)
                plt.plot(balanced_df.index, _max, label='max available', ls=ls, lw=lw, color='red')
                plt.scatter(unbalanced_df.index, list(unbalanced_df[node]), label='unbalanced', color='blue', s=s)
                plt.scatter(balanced_df.index, list(balanced_df[node]), label='balanced', color='orange', s=s)
                plt.plot(balanced_df.index, _min, label='min available', ls=ls, lw=lw, color='red')
                plt.title(node.replace('entry', ' entry').replace('exit', ' exit'))
                plt.ylabel(unit)
                plt.legend()
                if save_fig:
                    plt.savefig(figname + str(i) + '.png', bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    i += 1
                else:
                    plt.show()
        elif self.available_flowrates:
            #print('\nPlotting flows throughout the year for selected nodes before and after balancing...')
            for node in random.sample(balancing_nodes, number_of_graphs):
                plt.figure(figsize=figsize)
                _max = [self.available_flowrates[node]['max']] * len(balanced_df.index)
                _min = [self.available_flowrates[node]['min']] * len(balanced_df.index)
                plt.plot(balanced_df.index, _max, label='max available', ls=ls, lw=lw, color='red')
                plt.scatter(unbalanced_df.index, list(unbalanced_df[node]), label='unbalanced', color='blue', s=s)
                plt.scatter(balanced_df.index, list(balanced_df[node]), label='balanced', color='orange', s=s)
                plt.plot(balanced_df.index, _min, label='min available', ls=ls, lw=lw, color='red')
                plt.title(node.replace('entry', ' entry').replace('exit', ' exit'))
                plt.ylabel(unit)
                plt.legend()
                if save_fig:
                    plt.savefig(figname + str(i) + '.png', bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    i += 1
                else:
                    plt.show()
        else:
            #print('\nPlotting flows throughout the year for selected nodes before and after balancing...')
            for node in random.sample(balancing_nodes, number_of_graphs):
                plt.figure(figsize=figsize)
                plt.scatter(unbalanced_df.index, list(unbalanced_df[node]), label='unbalanced', color='blue', s=s)
                plt.scatter(balanced_df.index, list(balanced_df[node]), label='balanced', color='orange', s=s)
                plt.plot(balanced_df.index, [0] * len(balanced_df.index), label='sign change', ls=ls, lw=lw,
                         color='red')
                plt.title(node.replace('entry', ' entry').replace('exit', ' exit'))
                plt.ylabel(unit)
                plt.legend()
                if save_fig:
                    plt.savefig(figname + str(i) + '.png', bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    i += 1
                else:
                    plt.show()



def load_file(filename):
    if exists(filename):
        # print(f'\tloading {filename}...')
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            data.drop(['Unnamed: 0', 'date_time'], axis=1, errors='ignore', inplace=True)
            data.rename(columns={'DEB16': 'DEB1C', 'DEB19': 'DEB1D'}, inplace=True)
            data.reset_index(drop=True, inplace=True)
        return data
    else:
        # print(f'\t{filename} does not exist.')
        return pd.DataFrame()
