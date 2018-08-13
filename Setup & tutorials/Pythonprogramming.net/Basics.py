# ------------------ 1. Introduction ----------------- #
print("Hello world")

Garage = "Ferrari", "Honda", "Porsche", "Toyota"

for each_car in Garage:
    print(each_car)

########################################################
# ---------- 2. Print function and strings ----------- #
print('Say "hi" to the woman')
print("We're going to the store")
print('We\'re going to the store')  # Using the "escape character" backslash

# Concatenation
print('Sel'+'fish')
print("High", 5)
print("High"+str(5))
print(int("8")+5)
print(float("8")+5)

#######################################################
# -------------------- 3. Math ---------------------- #
5/2     # actually converts result to float
4**4    # exponent: "4 to the power of 4"
4^4     # definetely not an exponent

#######################################################
# ------------------- 4. Variables ------------------ #
exampleVar = 55+34
print(exampleVar)

x, y = (exampleVar, 5)
print(x)
print(y)

#######################################################
# ------------------ 5. While-loop ------------------ #
# Usually used for (in)finite iterations
condition = 1   # '=' Defining

while condition < 10:
    print(condition)
    condition += 1  # Equal to "condition = conition + 1"

# while True: #Infinite loop - CTR+C is not reliable for stopping it
#   print('doing stuff')

#######################################################
# ------------------- 6. For-loop ------------------- #
# Usually used for uncertain time frames
exampleList = [1, 5, 6, 2, 1, 10, 15, 20, 18]

for eachNumber in exampleList:  # Creating the eachNumber variable
    print(eachNumber)
    print('Continue program')

for x in range(1, 11):   # Excluding 11
    print(x)

#######################################################
# ---------------- 7. If statement ------------------ #
x = 5   # '=' is a definition
y = 10
z = 5

if x > y:
    print('x is greater than y')
if x < y:
    print('x is less than y')
if z < y > x:
    print('z is bigger than x and z')
if z <= x:
    print('z is less than or equal to x')
if z == x:  # '==' means equals
    print('z is equal to x')
if z != x:  # '==' means equals
    print('z is NOT equal to x')

#######################################################
# -------------- 8. If-else statement --------------- #
# --------------- 8. Else-statement ----------------- #
x = 5
y = 8

if x > y:
    print('x is greater than y')
else:
    print('x is not greater than y')

#######################################################
# -------------- 9. Else-if statement --------------- #
x = 5
y = 10
z = 22

if x > y:
    print('x is greater than y')
elif x > z:     # Short for else-if
    print('x is greater than z')
else:
    print('if and elif never ran...')

#######################################################
# ------------------ 10. Functions ------------------ #
def example():
    print('this code will never run')
    z = 3 + 9
    print(z)


example()

#######################################################
# ------------- 11. Function parameters ------------- #
def simple_addition(num1=5,num2=3):
    answer = num1 + num2
    print('num1 is', num1)
    print(answer)

simple_addition()

simple_addition(3)

simple_addition(num2=5)

#######################################################
# -------------- 12. Function defaults -------------- #
def simple(num1,num2=5):
    print(num1, num2)


simple(2)

def basic_window(width, height, font='TNR',
                 bgc='w', scrollbar=True):
    print(width, height, font, bgc)


basic_window(500, 350, bgc='b')

#######################################################
# ----------- 13. Global/local variables ------------ #
