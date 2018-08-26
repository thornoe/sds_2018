from multiprocessing import Pool

def worker(x):
    return x * (x + 3)

p = Pool(8)  # number of cores (threads) to allocate the operation too

inputs = range(10)
outputs = []

for result in p.imap(worker, inputs):  # <-- why `p.imap` and not just `p.map`?
    outputs.append(result)

p.close()

print(outputs)
