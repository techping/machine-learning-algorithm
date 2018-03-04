import math

class perceptron(object):

    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def predict(self, input_vec):
        return round(self.activator(reduce(lambda a, b: a + b, map(lambda (x, w): x * w, zip(input_vec, self.weights)), 0.0) + self.bias))

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = map(lambda (x, w): w + rate * delta * x, zip(input_vec, self.weights))
        self.bias += rate * delta

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def __str__(self):
        return 'weights\t%s\nbias\t%f' % (self.weights, self.bias)


def activator_(x):
    # return 1 if x > 0 else 0
    return 1 / (1 + math.exp(-x))

p = perceptron(2, activator_)
input_vecs = [[1,1], [0,0], [1,0], [0,1]]
labels = [1, 0, 0, 0]
p.train(input_vecs, labels, 10, 0.1)

print p
print '1, 1 = %d' %p.predict([1,1])
print '0, 0 = %d' %p.predict([0,0])
print '1, 0 = %d' %p.predict([1,0])
print '0, 1 = %d' %p.predict([0,1])