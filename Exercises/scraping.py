###############################################################
#       Control the pace of your calls using time             #
###############################################################
def ratelimit():
    "A function that handles the rate of your calls."
    time.sleep(1) # sleep one second.
# Reliable requests
def get(url,iterations=10,check_function=lambda x: x.ok):
    """This module ensures that your script does not crash from connection errors.
        that you limit the rate of your calls
        that you have some reliability check
        iterations : Define number of iterations before giving up.
        exceptions: Define which exceptions you accept, default is all.
    """
    for iteration in range(iterations):
        try:
            # add ratelimit function call here
            ratelimit() # !!
            response = session.get(url)
            if check_function(response):
                return response # if succesful it will end the iterations here
        except exceptions as e: #  find exceptions in the request library requests.exceptions
            print(e) # print or log the exception message.
    return None # your code will purposely crash if you don't create a check function later.

###############################################################
#       Control the pace of your calls using time             #
###############################################################
