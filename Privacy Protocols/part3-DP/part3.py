import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random
import csv

from numpy import sqrt, exp



''' Functions to implement '''

# TODO: Implement this function!
def read_dataset(file_path):
    data_dict = {'date': [], 'state': [], 'death': [], 'negative': [], 'positive': []}

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)  

        for row in csv_reader:
            data_dict['date'].append(row[0])
            data_dict['state'].append(row[1])
            data_dict['death'].append(float(row[2]))
            data_dict['negative'].append(float(row[3]))
            data_dict['positive'].append(float(row[4]))

    return data_dict


# TODO: Implement this function!
def get_histogram(dataset, state='TX', year='2020'):
    
    dates = np.array(dataset['date'])
    positives = np.array(dataset['positive'])
    
    #date_index = np.where(np.char.startswith(dates, "2020"))[0]
    
    histogram = []
    
    for i in range(1,13):
        
        if(i < 10):
            month_str = year + "-0" + str(i)
        
        else:
            month_str = year + "-" + str(i)
        
        month_index = np.where(np.equal(dates, month_str))[0]
        month_positives = positives[month_index]
        
        histogram.append(sum(month_positives))
    
    return histogram     
    

# TODO: Implement this function!
def get_dp_histogram(dataset, state, year, epsilon, N):
    
    sensitiviy = N
    noise_param = sensitiviy/epsilon
    
    dates = np.array(dataset['date'])
    positives = np.array(dataset['positive'])
    
    
    histogram = []
    
    sensitiviy = N
    noise_param = sensitiviy/epsilon
    
    for i in range(1,13):
               
        if(i < 10):
            month_str = year + "-0" + str(i)
        
        else:
            month_str = year + "-" + str(i)
            
        month_index = np.where(np.equal(dates, month_str))[0]
        month_positives = positives[month_index]
        
        #print("Before noise: " + str(month_positives))
        
        #calculating and adding the laplace noise
        #month_positives = [(x + (1/(2*(noise_param))) * np.exp(-(x/noise_param))) for x in month_positives]
        month_positives = [abs(x + np.random.laplace(0, noise_param)) for x in month_positives]
        #print("After noise: " + str(month_positives))
        
        
        #print(month_index)
        
        histogram.append(sum(month_positives))
    
    return histogram      
    


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    
    number_of_bins = len(actual_hist)
    
    average_error = [abs(noisy - actual) for noisy, actual in zip(noisy_hist, actual_hist)]
    
    return sum(average_error)/number_of_bins
    
    


# TODO: Implement this function!
def epsilon_experiment(dataset, state, year, eps_values, N):
    
    histogram = get_histogram(dataset, state, year)
    
    result = []
    
    for epsilon in eps_values:
        avg_list = []
        for i in range(10):
            noisy_histogram = get_dp_histogram(dataset, state, year, epsilon, N)
            error = calculate_average_error(histogram, noisy_histogram)
            avg_list.append(error)
            
        avg = sum(avg_list)/len(avg_list)
        result.append(avg)  
    
    return result


# TODO: Implement this function!
def N_experiment(dataset, state, year, epsilon, N_values):
    
    histogram = get_histogram(dataset, state, year)
    
    result = []
    
    for N in N_values:
        avg_list = []
        for i in range(10):
            noisy_histogram = get_dp_histogram(dataset, state, year, epsilon, N)
            error = calculate_average_error(histogram, noisy_histogram)
            avg_list.append(error)
            
        avg = sum(avg_list)/len(avg_list)
        result.append(avg)  
    
    return result
    


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def max_deaths_exponential(dataset, state, year, epsilon):
    
    sensitivity = 1
    
    dates = np.array(dataset['date'])
    states = np.array(dataset['state'])
    deaths = np.array(dataset['death'])
    
    state_deaths = []
    
    for i in range(1,13):
    
        if(i < 10):
            month_str = year + "-0" + str(i)
        
        else:
            month_str = year + "-" + str(i)

        month_index = np.where(np.equal(dates, month_str))[0]
        state_index = np.where(np.equal(states, state))[0]
        common_indices = np.intersect1d(month_index, state_index)
        month_deaths = deaths[common_indices]
        state_deaths.append(month_deaths[0])
    
    #score funtion -> number of deaths in a month
    
    #death_values = [np.exp((epsilon * x)/(2*sensitivity)) for x in state_deaths]

    # Calculate scores with scaling to prevent inf
    max_value = np.max((epsilon * np.array(state_deaths)) / (2 * sensitivity))
    
    #substract the biggest value from the all values to prevent overfow
    death_values = result = np.exp((epsilon * np.array(state_deaths)) / (2 * sensitivity) - max_value) / np.sum(np.exp((epsilon * np.array(state_deaths)) / (2 * sensitivity) - max_value))
    
    
    sum_values = sum(death_values)
         
    probabilities = [(x/sum_values) for x in death_values]
    
    #selecting a random index after exponential mechanism
    idx = np.random.choice(len(probabilities), p=probabilities)   
    
    month = idx + 1
    
    return month 
    
    
# TODO: Implement this function!
def exponential_experiment(dataset, state, year, epsilon_list):
    
    #getting the actual death values
    dates = np.array(dataset['date'])
    states = np.array(dataset['state'])
    deaths = np.array(dataset['death'])
    
    state_deaths = []
    
    for i in range(1,13):
    
        if(i < 10):
            month_str = year + "-0" + str(i)
        
        else:
            month_str = year + "-" + str(i)

        month_index = np.where(np.equal(dates, month_str))[0]
        state_index = np.where(np.equal(states, state))[0]
        common_indices = np.intersect1d(month_index, state_index)
        month_deaths = deaths[common_indices]
        state_deaths.append(month_deaths[0])
    
    max_month = state_deaths.index(max(state_deaths)) + 1
    
    #Experiment
    result = []
    for epsilon in epsilon_list:
      eps_results = []
      for i in range(1000):
        month = max_deaths_exponential(dataset, state, year, epsilon)
        eps_results.append(month)
      num_of_truth = eps_results.count(max_month)
      percent = (num_of_truth/len(eps_results)) * 100
      result.append(percent)
    
    return result
          

# FUNCTIONS TO IMPLEMENT END #


def main():
    filename = "covid19-states-history.csv"
    dataset = read_dataset(filename)
    
    """
    histogram = get_histogram(dataset)

    x_axis = [i + 1 for i in range(12)]
    
    print(x_axis)
    
    plt.bar(x_axis, histogram, color='blue', edgecolor='black')

    plt.xlabel('Month')
    plt.ylabel('Positive Tests')
    plt.title('Positive Test Case For State TX in year 2020')
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(x_axis, months)
    plt.ticklabel_format(axis='y', style='plain')

    plt.show()
    """
    state = "TX"
    year = "2020"

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg = epsilon_experiment(dataset, state, year, eps_values, 2)
    print(error_avg)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])


    print("**** N EXPERIMENT RESULTS ****")
    N_values = [1, 2, 4, 8]
    error_avg = N_experiment(dataset, state, year, 0.5, N_values)
    for i in range(len(N_values)):
        print("N = ", N_values[i], " error = ", error_avg[i])

    #state = "WY"
    year = "2020"

    print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
    #eps_values = [0.0001, 0.001, 0.01, 0.05, 0.1]
    exponential_experiment_result = exponential_experiment(dataset, state, year, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])



if __name__ == "__main__":
    
    main()
