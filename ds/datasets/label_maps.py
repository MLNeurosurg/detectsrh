# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Label mappings for SRH single cell datasets."""


# Dataset class enumeration, NA = 0 (first class)

sc_labels_mrcnn = [
    "n/a", "cell_nuclei", "cell_cytoplasm", "red_blood_cell", "macrophage",
    "axons", "blood_vessel", "chromatin"
]

short_sc_labels_mrcnn = [
    "n/a", "nu", "cyto", "rbc", "mp", "axon", "vessel", "chromatin"
]

sc_label_map_mrcnn = {x: i for (i, x) in enumerate(sc_labels_mrcnn)}
short_sc_label_map_mrcnn = {
    x: i
    for (i, x) in enumerate(short_sc_labels_mrcnn)
}

sc_label_reverse_map_mrcnn = {i: x for x, i in sc_label_map_mrcnn.items()}
short_sc_label_reverse_map_mrcnn = {
    i: x
    for x, i in short_sc_label_map_mrcnn.items()
}

# Name mapping for tensorboard
def sc_label_to_short_label_tb(sc_labels):
    label_to_short_label_map = {
        "n/a": "n/a",
        "cell_nuclei": "nu",
        "cell_cytoplasm": "cyto",
        "red_blood_cell": "rbc",
        "macrophage": "mp",
        "axons": "axon",
        "blood_vessel": "vessel",
        "chromatin": "chromatin"
    }
    return {label: label_to_short_label_map[label] for label in sc_labels}

