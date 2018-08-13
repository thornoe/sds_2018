# Subsetting lists
    x = ["a", "b", "c", "d"]
    x[1]
    x[-3] # same result!

# List slicing
    x = ["a", "b", "c", "d"]
    x[:2]
    x[2:]
    x[:]

# Subsetting lists of lists
    x = [["a", "b", "c"],
         ["d", "e", "f"],
         ["g", "h", "i"]]
    x[2][0]
    x[2][:2]

# List manipulation and inner workings of lists
    x = ["a", "b", "c", "d"]
    x[1] = "r"
    x[2:] = ["s", "t"]

    y = x[:]
    z = y
    del(z[-3:])
    print(x,y,z)
