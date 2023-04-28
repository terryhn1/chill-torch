import math
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

def isolated_clustering_proportion(x: str, y: str, dataset: Dataset) -> float:
    """
        The isolated clustering proportion, or the IKP, is a metric that determines whether
        the datapoints given is suitable for a scatter plot or a kde plot. When points
        are too cluttered(many surround a particular area on a 2D plane of any class), then it is suitable
        to be using a KDE plot. On the other hand, scatter plots look better when it appears that
        the data is more separated by hue. Therefore, the datapoints is scanned, separating the data
        into a k x k matrix, and the difference of the two most populated hues in the section is taken
        to determine whether the classes are separated sufficiently.

        Args:
            x: A string to a column of discrete values.
            y: A string to a column of discrete values.
            dataset: a ClassificationDataset created from data_loading.
        
        Returns: A float from the range of 0 to 1 representing the isolated clustering proportion.
    """
    x_column, y_column = dataset[x], dataset[y]

    min_x, min_y, max_x, max_y = find_min_max_boundaries(x_column, y_column)
    
    mid_x = math.ceil((max_x - min_x) / 2)
    mid_y = math.ceil((max_y - min_y) / 2)

    boundaries = {'min_x': min_x, 'max_x': max_x,
                  'min_y': min_y, 'max_y': max_y,
                  'mid_x': mid_x, 'mid_y': mid_y}

    sections = create_sections(boundaries)
    labels = dataset[dataset.class_header]
    add_datapoints_to_sections(columns = [x_column, y_column],
                               sections = sections,
                               boundaries = boundaries,
                               labels = labels)

    return find_ikp(sections, dataset)

def find_min_max_boundaries(x_column: pd.Series, y_column: pd.Series) -> Tuple[float]:
    """
        Finds the upper and lower bounds of the datapoints in order to
        separate the data into a k x k matrix.

        Args:
            x_column: pd.Series that contains datapoints for a discrete column.
            y_column: pd.Series that contains datapoints for a discrete column.
        
        Returns: Tuple containing the maximum and minimum x and y boundaries.
    """
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for i in range(len(x_column)):
        min_x = min(min_x, x_column[i])
        max_x = max(max_x, x_column[i])
        
        min_y = min(min_y, y_column[i])
        max_y = max(max_y, y_column[i])
    
    return min_x, min_y, max_x, max_y

def create_sections(boundaries: Dict[str, float]) -> Dict[tuple, defaultdict]:
    """
        Creates the k x k matrix that can be used for sectioning
        each datapoint to a certain area.

        Args:
            boundaries: Dictionary that contains the float values that determine the boundary boxes in the k x k matrix.

        Returns: A dictionary representing the k x k matrix. 
    """
    min_x, max_x = boundaries['min_x'], boundaries['max_x']
    min_y, max_y = boundaries['min_y'], boundaries['max_y']
    mid_x, mid_y = boundaries['mid_x'], boundaries['mid_y']

    sections = {}
    sections[(min_x, mid_x, min_y, mid_y)] = defaultdict(int)
    sections[(min_x, mid_x, mid_y, max_y)] = defaultdict(int)
    sections[(mid_x, max_x, min_y, mid_y)] = defaultdict(int)
    sections[(mid_x, max_x, mid_y, max_y)] = defaultdict(int)

    return sections

def add_datapoints_to_sections(columns: List[pd.Series],
                               sections: Dict[tuple, defaultdict],
                               boundaries: Dict[str, float],
                               labels: pd.Series) -> None:
    """
        Counts up the amount of datapoints for the certain hue associated
        with the datapoint into its section along with the total amount
        of datapoints in each section.

        Args:
            columns: List containing the columns of discrete values.
            sections: A dictionary representing a k x k matrix with boundary boxes.
            boundaries: A dictionary containing float boundaries for the k x k matrix.
            labels: pd.Series containing the hue color, which is the classification label.
        
        Returns: None
    """
    x_col, y_col = columns[0], columns[1]
    min_x, max_x = boundaries['min_x'], boundaries['max_x']
    min_y, max_y = boundaries['min_y'], boundaries['max_y']
    mid_x, mid_y = boundaries['mid_x'], boundaries['mid_y']
    for i in range(len(x_col)):
        # bottom-left section
        if x_col[i] <= mid_x and y_col[i] <= mid_y:
            sections[(min_x, mid_x, min_y, mid_y)][labels[i]] += 1
            sections[(min_x, mid_x, min_y, mid_y)]['total'] += 1
            
        # top-left section
        elif x_col[i] <= mid_x and y_col[i] > mid_y:
            sections[(min_x, mid_x, mid_y, max_y)][labels[i]] += 1
            sections[(min_x, mid_x, mid_y, max_y)]['total'] += 1

        # bottom-right section
        elif x_col[i] > mid_x and y_col[i] <= mid_y:
            sections[(mid_x, max_x, min_y, mid_y)][labels[i]] += 1
            sections[(mid_x, max_x, min_y, mid_y)]['total'] += 1

        # top-right section
        elif x_col[i] > mid_x and y_col[i] > mid_y:
            sections[(mid_x, max_x, mid_y, max_y)][labels[i]] += 1
            sections[(mid_x, max_x, mid_y, max_y)]['total'] += 1

def find_ikp(sections: Dict[tuple, defaultdict],
             dataset: Dataset,
             decline_rate: float = 0.0007) -> float:
    """
        Determines the two highest populated hues in a certain section of the k x k matrix, and
       takes the difference as the determining factor as to whether the section can be identified
       with a single hue color.

        Args:
            sections: A dictionary containing a representation of the k x k matrix.
            dataset: A ClassificationDataset built from data_loading.

        Returns: a float representing the isolated clustering proportion.
    """
    isolated_cluster_proportion_rate = 0
    for section in sections.values():
        max_candidate1 = max_candidate2 = 0
        total_value = section['total']
        if total_value == 0:
            continue
        for hue in dataset.classes:
            hue_value = section[hue]
            population_rate = hue_value / total_value
            if max_candidate1 == 0:
                max_candidate1 = population_rate
            elif max_candidate2 == 0:
                max_candidate2 = population_rate
            elif hue_value > max_candidate1:
                max_candidate2 = max_candidate1
                max_candidate1 = population_rate
            elif hue_value > max_candidate2:
                max_candidate2 = population_rate
        
        differential = (max_candidate1 - max_candidate2) * 100
        weight = total_value // len(dataset) 
            
        
        isolated_cluster_proportion_rate += (differential * weight)
        
    return math.tanh((isolated_cluster_proportion_rate**2) * decline_rate)

def class_bin_relationship(x: str, y: str, dataset: Dataset) -> float:
    """
        The class bin relationship of a set of datapoints determines whether the datapoints
        are suitable to use for a bar plot. For more information gain, a barplot is good
        in the case that there is a significance difference between the discrete values of
        a labeled column or there is a significance difference between the discrete values seen at
        each labeled column for a hue. This allows us to identify whether the datapoints experience
        dependency, and therefore further correlation. The label similarity score and the hue
        similarity score is taken in order to determine the class bin relationship.

        Args:
            x: A string leading to a labeled column, owning less than 20 discrete values.
            y: A string leading to a column with many discrete values.
            dataset: ClassificationDataset created from data_loading.
        
        Returns: Float from range 0.0 to 1.0 representing the class bin relationship score. 
    """

    label_column = dataset[x]
    discrete_column = dataset[y]
    hue_column = dataset[dataset.class_header]
    unique_labels = label_column.unique()
    hue_bin = {label: defaultdict(int) for label in unique_labels}
    hue_relation_bin = defaultdict(list)
    average_value_bin = {label: defaultdict(int) for label in unique_labels} 

    for i in range(len(label_column)):
        hue_bin[label_column[i]][hue_column[i]] += discrete_column[i]
        average_value_bin[label_column[i]][hue_column[i]] += 1
    
    label_similarity_score = find_label_similarity_score(hue_bin = hue_bin,
                                                         average_value_bin = average_value_bin,
                                                         hue_relation_bin = hue_relation_bin)

    hue_similarity_score = find_hue_similarity_score(hue_relation_bin)

    return label_similarity_score + hue_similarity_score

def find_label_similarity_score(hue_bin: Dict[int, defaultdict],
                                average_value_bin: Dict[int: defaultdict],
                                hue_relation_bin: Dict[any, list],
                                precision: int = 5) -> float:
    """
        Label similarity score is defined as the difference between hues on
        the same label/class. A high similarity score means that there is almost
        no difference, resulting in less information gain.

        Args:
            hue_bin: Separates the datapoints into [label, hue] and sums the discrete values.
            average_value_bin: Separates the datapoints into [label, hue] and averages the hue_bin.
            hue_relation_bin: Separates datapoints into hues encountered.
            precision: Float indicating the amount of decimals to round the similarity score up to.
        
            Returns: Float representing the label similarity score.
    """
    label_similarity_score = 0
    for label in hue_bin:

        set_total_hue_bin(hue_bin = hue_bin,
                          average_value_bin= average_value_bin,
                          hue_relation_bin = hue_relation_bin,
                          label = label)
            
        hue_bin[label].clear()
        average_differential = 0
        combination_count = 0

        for hue in average_value_bin:
            for compared_hue in average_value_bin:
                if hue != compared_hue:
                    average_differential += abs(average_value_bin[label][hue] - average_value_bin[label][compared_hue])
                    combination_count += 1
                
        label_similarity_score += round(average_differential / combination_count, precision)
    
    return label_similarity_score

def set_total_hue_bin(hue_bin: Dict[int, defaultdict],
                      average_value_bin: Dict[int, defaultdict],
                      hue_relation_bin: Dict[any, list],
                      label: int) -> None:
    """
        Adds averaged values for future computations along with setting the minimum and maximum
        candidates for average value hues.

        Args:
            hue_bin: Separates the datapoints into [label, hue] and sums the discrete values.
            average_value_bin: Separates the datapoints into [label, hue] and averages the hue_bin.
            hue_relation_bin: Separates datapoints into hues encountered.
            label: Integer representing the label or the x_column value.
        
        Returns: None.
    """
    for hue in average_value_bin:
        average_value_bin[label][hue] = hue_bin[label][hue] / average_value_bin[label][hue]
        hue_relation_bin[hue].append(average_value_bin[label][hue])

        min_max_hue = str(hue) + 'min_max'
        if not hue_relation_bin[min_max_hue]:
            hue_relation_bin[min_max_hue] = [float('inf'), float('-inf')]
        else:
            hue_relation_bin[min_max_hue][0] = min(hue_relation_bin[min_max_hue][0], hue_relation_bin[hue][-1])
            hue_relation_bin[min_max_hue][1] = max(hue_relation_bin[min_max_hue][1], hue_relation_bin[hue][-1])

def find_hue_similarity_score(hue_relation_bin: Dict[any, list]) -> float:
    """
        Hue similarity score is defined as the similarity between hues across different
        labels. A higher score indicates that there is slight or no changes across the labels
        and therefore is not a significant factor for graphing.

        Args:
            hue_relation_bin: Separates datapoints into hues encountered.

        Returns: float representing the hue similarity score.
    """
    hue_similarity_score = 0
    for hue in hue_relation_bin:
        if type(hue) == str:
            continue
        average_val = np.mean(hue_relation_bin[hue])
        bound_diff = hue_relation_bin[(str(hue) + 'min_max')][1] - hue_relation_bin[(str(hue) + 'min_max')][0]
        confidence_interval = 0.4
        range_val = bound_diff * confidence_interval
        min_interval, max_interval = average_val - range_val, average_val + range_val

        count = 0
        for score in hue_relation_bin[hue]:
            if min_interval <= score <= max_interval:
                    count += 1
            
        hue_similarity_score += round(count / len(hue_relation_bin[hue]), 5)
    
    return hue_similarity_score

def population_distribution():
    x = ...

    return x

def correlation_classes():
    x = ...

    return x