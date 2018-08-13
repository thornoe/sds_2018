# Functions (a piece of reusable code)
    var1 = [1, 2, 3, 4]
    var2 = True
    var3 = "Happy holiday"

    # Print out type of var1
    print(type(var1))

    # Print out length of var1
    print(len(var1))
    len(var3)

    # Convert var2 to an integer: out2
    out2 = int(var2)

# Help
    help(complex)

# Sorted
    first = [11.25, 18.0, 20.0]
    second = [10.75, 9.50]
    full = first + second

    # Sort full in descending order: full_sorted
    full_sorted = sorted(full,reverse=1)

    # Print out full_sorted
    print(full_sorted)

# String methods
    place = "poolhouse"

    # Print out the number of o's in place
    place.count('o')

# List methods that don't change the list
    areas = [11.25, 18.0, 20.0, 10.75, 9.50]

    # Print out the index of the element 20.0
    areas.index(20.0)

    # Print out how often 9.50 appears in areas
    areas.count(9.5)

# List methods that changes the list
    areas.append(24.5)
    areas.append(15.45)
    areas.reverse()
    print(areas)

# Packages
    import numpy as np
    from numpy import array
