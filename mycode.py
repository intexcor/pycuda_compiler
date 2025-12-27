from compiler import cuda_compile, Array, float32

@cuda_compile
def process(data: Array[float32]):
    for i in range(len(data)):
        if data[i] > 0:
            data[i] = sqrt(data[i])
        else:
            data[i] = data[i] * data[i]

data = process.array([4.0, -2.0, 9.0, -3.0])
process(data)
print(process.to_numpy(data))
print(process.cuda_code)
