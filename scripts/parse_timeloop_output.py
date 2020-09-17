# parse results from log file or xml file
# return generated string
import re
import os
import subprocess
import numpy as np
# import pandas as pd
import xml.etree.ElementTree as ET
# from pathlib import Path

# Output file names.
out_prefix = "timeloop-mapper."
xml_file_name = out_prefix + "map+stats.xml"


def parse_timeloop_stats(dirname, dense):
    file = os.path.join(dirname, xml_file_name)
    tree = ET.parse(file)
    root = tree.getroot()

    # Parse out the problem shape
    problem_dims = root.findall('a')[0].findall('workload_')[0].findall('bounds_')[0].findall('item')
    if len(problem_dims) == 7:
        problem = {'FH': int(problem_dims[0].findall('second')[0].text),
                   'FW': int(problem_dims[1].findall('second')[0].text),
                   'OH': int(problem_dims[2].findall('second')[0].text),
                   'OW': int(problem_dims[3].findall('second')[0].text),
                   'IC': int(problem_dims[4].findall('second')[0].text),
                   'OC': int(problem_dims[5].findall('second')[0].text),
                   'IN': int(problem_dims[6].findall('second')[0].text),
                   }
        macs = problem['FH'] * problem['FW'] * problem['IC'] * problem['OH'] * problem['OW'] * problem['IN'] * problem['OC']
    elif len(problem_dims) == 6:
        problem = {'FH': int(problem_dims[0].findall('second')[0].text),
                   'FW': int(problem_dims[1].findall('second')[0].text),
                   'OH': int(problem_dims[2].findall('second')[0].text),
                   'OW': int(problem_dims[3].findall('second')[0].text),
                   'C': int(problem_dims[4].findall('second')[0].text),
                   'IN': int(problem_dims[5].findall('second')[0].text),
                   }
        macs = problem['FH'] * problem['FW'] * problem['C'] * problem['OH'] * problem['OW'] * problem['IN']
    else:
        raise Exception('Incorrect parsed shape')
    topology = root.findall('engine')[0].findall('topology_')[0]
    # Get the list of storage elements
    levels = topology.findall('levels_')[0]
    num_levels = int(levels.findall('count')[0].text)
    level_ptrs = levels.findall('item')

    networks = topology.findall('networks_')[0]
    num_networks = int(networks.findall('count')[0].text)
    network_ptrs = networks.findall('item')
    assert(num_levels == num_networks + 1)
    # Initialize a dictionary that stores energy breakdown and other statistics
    energy_breakdown_pJ = {}

    arithmetic_level_found = False
    level_energy = {}
    level_cycles = 0

    for level_id in range(len(level_ptrs)):
        level_ptr = level_ptrs[level_id]
        level = level_ptr.findall('px')[0]
        # The XML structure is interesting. Every Level gets a <px>, but
        # only the first object of each type gets a full class_id descriptor.
        # For example, the first model::BufferLevel item will get:
        #    <px class_id="9" class_name="model::BufferLevel" tracking_level="1" version="0" object_id="_1">
        # but subsequent levels will get something like:
        #    <px class_id_reference="9" object_id="_2">
        # with increasing object_ids. We can keep a table of new class_ids as
        # we encounter them, but for now we'll just hack something that works.
        # Is this the Arithmetic level (the only one)?
        if 'class_id' in level.attrib and level.attrib['class_name'] == "model::ArithmeticUnits":
            assert arithmetic_level_found == False
            arithmetic_level_found = True
            cycles = int(level.findall('cycles_')[0].text)
            utilized_instances = float(level.findall('utilized_instances_')[0].text)
            total_instances_list = level.findall('specs_')[0].findall('instances')[0].findall('t_')
            if total_instances_list == []:  # this happens when no mapping is returned by timeloop
                total_instances = 1  # dummy value
            else:
                total_instances = float(level.findall('specs_')[0].findall('instances')[0].findall('t_')[0].text)
            arithmetic_utilization = utilized_instances / total_instances
            energy_breakdown_pJ['MAC'] = {'energy': float(level.findall('energy_')[0].text), 'utilization': arithmetic_utilization}
            if dense:
                level_energy['MAC'] = float(level.findall('energy_')[0].text)
            else:
                level_energy['MAC'] = float(level.findall('sparse_energy_')[0].text)
            continue

        # Continue storage level stat extraction...
            # Level specifications
        specs = level.findall('specs_')[0]
        stats = level.findall('stats_')[0]
        generic_level_specs = specs.findall('LevelSpecs')[0]
        level_name = generic_level_specs.findall('level_name')[0].text
        if dense:
            level_cycles = int(stats.findall('cycles')[0].text)
        else:
            level_cycles = int(stats.findall('sparse_cycles')[0].text)
        # Storage access energy
        reads_per_instance = get_stat(stats, 'reads', int)
        updates_per_instance = get_stat(stats, 'updates', int)
        fills_per_instance = get_stat(stats, 'fills', int)
        accesses_per_instance = reads_per_instance + updates_per_instance + fills_per_instance

        utilized_capacity = get_stat(stats, 'utilized_capacity', int)
        instances = get_stat(stats, 'utilized_instances', int)
        clusters = get_stat(stats, 'utilized_clusters', int)

        total_instances_obj = specs.findall('instances')[0].findall('t_')
        if len(total_instances_obj) == 0:
            total_instances = sum(instances)
        else:
            total_instances = int(total_instances_obj[0].text)

        total_capacity_obj = specs.findall('size')[0].findall('t_')
        if len(total_capacity_obj) == 0:
            total_capacity = sum(utilized_capacity)
        else:
            total_capacity = int(total_capacity_obj[0].text)
        energy_per_access_per_instance = get_stat(stats, 'energy_per_access', float)
        storage_access_energy_in_pJ = energy_per_access_per_instance * accesses_per_instance * instances
        read_energy = energy_per_access_per_instance * reads_per_instance * instances
        if dense:
            level_energy[level_name] = np.sum(storage_access_energy_in_pJ)  # Fixme: should we include more for level energy??
        else:
            sparse_energy = get_stat(stats, 'sparse_energy', float)
            sparse_index_energy = get_stat(stats, 'sparse_index_energy', float)
            sorting_blk_ptr_energy = get_stat(stats, 'sorting_blk_ptr_energy', float)
            sparse_access_energy = (sparse_energy + sparse_index_energy) * instances
            total_sparse_energy_per_datatype = sparse_access_energy + sorting_blk_ptr_energy
            level_energy[level_name] = np.sum(total_sparse_energy_per_datatype)

        assert(level_id >= 1)
        for n in network_ptrs:
            network_name = n.findall('first')[0].text
            network_source = network_name.split(None, 1)[0]
            if network_source == level_name:
                network = n.findall('second')[0].findall('px')[0]
                break

        network_stats = network.findall('stats_')[0]
        num_hops = get_stat(network_stats, 'num_hops', float)
        energy_per_hop_per_instance = get_stat(network_stats, 'energy_per_hop', float)
        ingresses = get_stat(network_stats, 'ingresses', int)
        network_energy_in_pJ = num_hops * ingresses * energy_per_hop_per_instance * instances
        # Add energy
        spatial_add_energy_per_instance = get_stat(network_stats, 'spatial_reduction_energy', float)
        temporal_add_energy_per_instance = get_stat(stats, 'temporal_reduction_energy', float)
        temporal_add_energy = np.nansum(temporal_add_energy_per_instance * instances)
        spatial_add_energy = np.nansum(spatial_add_energy_per_instance * instances)

        # Address generation energy
        address_generation_energy_per_cluster = get_stat(stats, 'addr_gen_energy', float)
        address_generation_energy = np.nansum(address_generation_energy_per_cluster * clusters)

        # Special Case when the memory level is a dummy (capacity = 0)
        if total_capacity == 0:
            utilization = 0
        else:
            utilization = sum((utilized_capacity * instances) / (total_capacity * total_instances))
        # unused, fix the structure later
        energy_breakdown_pJ[level_name] = {
            'cycles': level_cycles,
            'energy': np.nansum(storage_access_energy_in_pJ) + temporal_add_energy + address_generation_energy + np.nansum(network_energy_in_pJ) + spatial_add_energy,
            'storage_access_energy': np.nansum(storage_access_energy_in_pJ),
            'read_energy': np.nansum(read_energy),
            'temporal_add_energy': temporal_add_energy,
            'spatial_add_energy': spatial_add_energy,
            'address_generation_energy': address_generation_energy,
            'network_energy': np.nansum(network_energy_in_pJ),
            'energy_per_access_per_instance': energy_per_access_per_instance,
            'reads_per_instance': reads_per_instance,
            'updates_per_instance': updates_per_instance,
            'fills_per_instance': fills_per_instance,
            'accesses_per_instance': accesses_per_instance,
            'instances': instances,
            'utilization': utilization,
            'num_hops': num_hops,
            'ingresses': ingresses,
            'energy_per_hop_per_instance': energy_per_hop_per_instance}
    energy_pJ = sum([value['energy'] for key, value in energy_breakdown_pJ.items()])
    return level_cycles, level_energy, macs


def get_stat(stats, stat, cast):
    items = stats.findall(stat)[0].findall('PerDataSpace')[0].findall('item')
    count = len(items)
    out = np.array([0] * count, dtype=cast)
    for j in range(count):
        if stat == 'ingresses':
            value = sum([cast(i.text) for i in items[j].findall('item')])
        else:
            value = cast(items[j].text)
        out[j] = value
    return out
