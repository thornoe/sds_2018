# Intermediate Python for Data Science (paid course)
https://www.datacamp.com/courses/intermediate-python-for-data-science

# Different plot types with matplotlib
import matplotlib.pyplot as plt

    # Basic plots with matplotlib
        plt.plot(x,y)

    # Scatterplot
        plt.scatter(x, y)
        plt.scatter(x, y, s = z) # z determines the size of the dots, e.g. population
        plt.scatter(x, y, c = col) # colours
        plt.scatter(x, y, alpha = 0.8) # The oppacity of the bubbles can be set from zero to one, where zero is totally transparent, and one is not at all transparent.

    # Histogram
        plt.hist(x, bins=10)

# Beautifying
    plt.xscale('log') # logarithmic scale
    plt.xlabel('xlab')
    plt.ylabel('ylab')
    plt.title('Y on X')
    plt.yticks([0,2,4,6,8,10],
        ['0', '2k', '4k', '6k', '8k', '10k'])
    plt.text(5700, 80, 'China') # Adding a label to a specific scatter
    plt.grid(True)

# Show and clear plot
    plt.show()
    plt.clf() # Clears up the plot
