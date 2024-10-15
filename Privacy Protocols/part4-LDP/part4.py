import numpy as np
from matplotlib import pyplot as plt
from shapely import geometry, ops

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

""" Helpers """


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


def plot_grid(cell_percentages):
    max_lat = -8.58
    max_long = 41.18
    min_lat = -8.68
    min_long = 41.14

    background_image = plt.imread('porto.png')

    fig, ax = plt.subplots()
    ax.imshow(background_image, extent=[min_lat, max_lat, min_long, max_long], zorder=1)

    rec = [(min_lat, min_long), (min_lat, max_long), (max_lat, max_long), (max_lat, min_long)]
    nx, ny = 4, 5  # number of columns and rows  4,5

    polygon = geometry.Polygon(rec)
    minx, miny, maxx, maxy = polygon.bounds
    dx = (maxx - minx) / nx  # width of a small part
    dy = (maxy - miny) / ny  # height of a small part
    horizontal_splitters = [geometry.LineString([(minx, miny + i * dy), (maxx, miny + i * dy)]) for i in range(ny)]
    vertical_splitters = [geometry.LineString([(minx + i * dx, miny), (minx + i * dx, maxy)]) for i in range(nx)]
    splitters = horizontal_splitters + vertical_splitters

    result = polygon
    for splitter in splitters:
        result = geometry.MultiPolygon(ops.split(result, splitter))

    grids = list(result.geoms)

    for grid_index, grid in enumerate(grids):
        x, y = grid.exterior.xy
        ax.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

        count = cell_percentages[grid_index]
        count = round(count, 2)

        centroid = grid.centroid
        ax.annotate(str(count) + '%', (centroid.x, centroid.y), color='black', fontsize=12,
                    ha='center', va='center', zorder=3)

    plt.show()


# You can define your own helper functions here. #

def calculate_average_error(true_locations, perturbed_locations):
    
    number_of_locations = len(true_locations)
    
    average_error = [abs(perturbed - true) for true, perturbed in zip(true_locations, perturbed_locations)]
    
    return sum(average_error)/number_of_locations

### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    
    all_locations = np.arange(1, 21, dtype=int)
    
    locations_without_val = all_locations[all_locations != val]
    #print(locations_without_val)
    
    p = np.exp(epsilon)/(np.exp(epsilon) + len(all_locations) - 1)
    
    coin_toss = np.random.rand()
    
    if(coin_toss < p):
        return val
    
    else:
        return np.random.choice(locations_without_val)
    

    #print(coin_toss)
    


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    
    percentage_of_taxis_in_locations = [0 for x in range(20)]
    number_of_taxis = len(perturbed_values)
    
    p = np.exp(epsilon)/(np.exp(epsilon) + len(percentage_of_taxis_in_locations) - 1)
    q = (1-p)/(len(percentage_of_taxis_in_locations) - 1)
    
    for i in range(1,21):
        report = perturbed_values.count(i)
        estimate = ((report - (number_of_taxis * q)) / (p - q))
        
        percentage_of_taxis_in_locations[i-1] = (estimate/number_of_taxis) * 100
    
    
    #print(percentage_of_taxis_in_locations)
    
    return percentage_of_taxis_in_locations


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    #estimate_grr(dataset, epsilon)
    
    true_locations = dataset
    pertubed_locations = [perturb_grr(x, epsilon) for x in dataset]
    
    num_of_taxis = len(true_locations)
    true_percentages = [0 for x in range(20)]
    
    for i in range(1,21):
        report = true_locations.count(i)
        true_percentages[i-1] = (report/num_of_taxis) * 100 
    
    perturbed_percentages = estimate_grr(pertubed_locations, epsilon)
    
    return calculate_average_error(true_percentages, perturbed_percentages)


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    
    vector = [0 for x in range(20)]
    
    vector[val-1] = 1
    
    return vector


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    
    p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
    q = 1 / (np.exp(epsilon/2) + 1)
    
    perturbed_vector = [0 for x in range(len(encoded_val))]
    
    for i in range(len(perturbed_vector)):
        probability = np.random.rand()
        
        if(probability < p):
            continue
        
        else:
           val = perturbed_vector[i]
            
           if(val == 0):
              perturbed_vector[i] = 1
                
           elif(val == 1):
               perturbed_vector[i] = 0
            
    return perturbed_vector


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    
    percentage_of_taxis_in_locations = [0 for x in range(20)]
    number_of_taxis = len(perturbed_values)
    
    values = np.array(perturbed_values)
    #print(values)
    
    values_sum = np.sum(values, axis=0)
    #print(values_sum)
    
    p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
    q = 1 / (np.exp(epsilon/2) + 1)
    
    for i in range(1, 20):
        
        report = values_sum[i]
        estimate = ((report - (number_of_taxis * q)) / (p - q))
        
        percentage_of_taxis_in_locations[i-1] = (estimate/number_of_taxis) * 100
    
    return percentage_of_taxis_in_locations


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    
    true_vectors = [encode_rappor(x) for x in dataset]
    
    pertubed_vectors = [perturb_rappor(y, epsilon) for y in true_vectors]
    
    num_of_taxis = len(true_vectors)

    true_sum = np.sum(np.array(true_vectors), axis=0)
    
    true_percentages = (true_sum/num_of_taxis) * 100
    
    #print("true_sum: " + str(sum(true_sum)))
    #print(pertubed_vectors)
    #print("true_percentage: " + str(sum(true_percentages)))
    
    perturbed_percentages = estimate_rappor(pertubed_vectors, epsilon)
        
    return calculate_average_error(true_percentages, perturbed_percentages)
    

# OUE

# TODO: Implement this function!
def encode_oue(val):
    
    vector = [0 for x in range(20)]
    
    vector[val-1] = 1
    
    return vector
    

# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    
    p = 0.5
    q1 = 1 / (np.exp(epsilon/2) + 1)
    q0 = np.exp(epsilon) / (np.exp(epsilon) + 1)
    
    #perturbed_vector = [0 for x in range(len(encoded_val))]
    
    perturbed_vector = np.array(encoded_val.copy())
    #print(perturbed_vector)
    
    zero_idx = np.where(perturbed_vector == 0)[0]
    one_idx = np.where(perturbed_vector == 1)[0] 
    
    probabilities = [q0, p]
    probabilities = np.array(probabilities) / sum(probabilities)
    #print(probabilities)
    
    
    for idx in zero_idx:    
        perturbed_vector[idx] = np.random.choice([0, 1], p=probabilities)
    
    probabilities = [q1, p]
    probabilities = np.array(probabilities) / sum(probabilities)
    perturbed_vector[one_idx] = np.random.choice([0, 1], p=probabilities)
    """
    for i in range(len(perturbed_vector)):
        probability = np.random.rand()
        
        possible_values = [0, 1]
        
        if(encoded_val[i] == 1):
            probabilities = [q1, p]
            
            #normalizing probabilites by dividing their sum from both of them to prevent not sum to 1 error
            probabilities = np.array(probabilities) / sum(probabilities)
            
            perturbed_vector[i] = np.random.choice(possible_values, p=probabilities)
        
        elif(encoded_val[i] == 0):
            
            
            probabilities = np.array(probabilities) / sum(probabilities)
            
            perturbed_vector[i] = np.random.choice(possible_values, p=probabilities)
    """
    #print(encoded_val)
    #print(perturbed_vector)
    return perturbed_vector  


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    
    percentage_of_taxis_in_locations = [0 for x in range(20)]
    number_of_taxis = len(perturbed_values)
    
    values = np.array(perturbed_values)
    #print(values)
    values_sum = np.sum(values, axis=0)
    #print(values_sum)
    
    for i in range(1, 20):
        
        num_of_ones = values_sum[i]
        estimate = 2 * (((np.exp(epsilon) + 1) - num_of_ones) - number_of_taxis) / (np.exp(epsilon) - 1)
        
        percentage_of_taxis_in_locations[i-1] = (estimate/number_of_taxis) * 100
    
    return percentage_of_taxis_in_locations


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    
    true_vectors = [encode_oue(x) for x in dataset]
    
    pertubed_vectors = [perturb_oue(y, epsilon) for y in true_vectors]
    
    num_of_taxis = len(true_vectors)

    true_sum = np.sum(np.array(true_vectors), axis=0)
    
    true_percentages = (true_sum/num_of_taxis) * 100
      
    perturbed_percentages = estimate_oue(pertubed_vectors, epsilon)
        
    return calculate_average_error(true_percentages, perturbed_percentages)


def main():
    dataset = read_dataset("taxi-locations.dat")
    
    
    v1 = perturb_oue(encode_oue(4), 0.5)
    #print(v1)
    """
    v2 = perturb_oue(encode_oue(6), 0.5)
    vl = [v1, v2]
    
    print(estimate_oue(vl, 0.5))
    """
    
    print("GRR EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))
        #percents = estimate_grr(dataset, epsilon)
        #plot_grid(percents)
        

    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))

    print("*" * 50)
    
    print("OUE EXPERIMENT")
    for epsilon in [0.01, 0.1, 0.5, 1, 2]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.3f}".format(epsilon, error))
            
    

if __name__ == "__main__":
    main()
