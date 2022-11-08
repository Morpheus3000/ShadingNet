import NetworkParameters as param

class Parameters(object):

	def __init__(self, name):
		self.name = name

	def getWeight(self):

		if self.name == 'first_pipeline':
			return param.weights
		else:
			raise RuntimeError('Unknown weight name %s...\n', self.name)

	def getBias(self):

		if self.name == 'first_pipeline':
			return param.biases
		else:
			raise RuntimeError('Unknown bias name %s...\n', self.name)