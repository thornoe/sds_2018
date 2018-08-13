# Numeric python
import numpy as np

# Identify outliers
    x = [4 , 9 , 6, 3, 1]
    y = np.array(x)
    y[1]
    y.shape
    # For numpy specifically, you can also use boolean numpy arrays:
    high = y > 5
    y[high]

# Generate data
    height = np.round(np.random.normal(1.75, 0.20, 5000), 2)
    weight = np.round(np.random.normal(60.32, 15, 5000), 2)
    np_baseball = np.column_stack((height, weight))

# Summary statistics
    # Print mean height (first column)
    avg = np.mean(np_baseball[:,0])
    print("Average: " + str(avg))

    # Print median height.
    med = np.median(np_baseball[:,0])
    print("Median: " + str(med))

    # Print out the standard deviation on height.
    stddev = np.std(np_baseball[:,0])
    print("Standard Deviation: " + str(stddev))

    # Print out correlation between first and second column.
    corr = np.corrcoef((np_baseball[:,0],np_baseball[:,1]))
    print("Correlation: " + str(corr))
