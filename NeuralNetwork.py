import numpy
#URL:
#https://www.youtube.com/watch?v=gwitf7ABtK8

#Neural Network Function
def NN(m1, m2, w1, w2, b):
  z = m1 + w1 + m2 * w2 + b
  return sigmoid(z)

#Sigmoid Function
def sigmoid(x)
  return 1/(1+numpy.exp(-x))

w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()

#Red Flower
NN(3,1.5, w1, w2, b)

#Blue Flower
NN(2,1, w1, w2, b)

#Red Flower
NN(2, .5, w1, w2, b)

# Machine Learning Portion:
# Using neural network, computer responds with random phrases:

phrases = ['seems like its', 'I guess', 'I think', 'possibly', 'looks like', 'guessing...']

# Sample Data of Red and Blue Flowers
data = [[3,1.5,1],[2,1,0],[4,1.5,1],[3.5,.5,1],[2,.5,0],[5.5,1,1],[1,1,0]]

rand_data = data[numpy.random.randint(len(data))]

m1 = rand_data[0]
m2 = rand_data[1]

prediction = NN(m1,m2,w1,w2,b)

prediction_text = ["blue", "red"][int(numpy.round(prediction))]
phrase = numpy.random.choice(phrases) + " " + prediction_text
print(phrase)
o = os.system("say" + phrase)

print("It's really" + ["blue", "red"][rand_data[2]])
























)








































