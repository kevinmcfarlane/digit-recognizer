import traceback

from datetime import datetime

class Observation:
    """
    A digit from 0 to 9 and its representation in pixels.
    """
    def __init__(self, label, pixels = None):
        self.label = label
        self.pixels = pixels if pixels is not None else []

class DataReader:
    """
    Reads images from a file and transforms them to a form suitable for analyis.
    """
    def observation_factory(self, data):
        """
        Return an Observation instance.

        Args: 
            data (str): A line of comma-delimited input data.

        Returns: 
            Observation: A digit from 0 to 9 and its representation in pixels. 
        """

        comma_separated = data.strip().split(",")
        label = comma_separated[0]
        #pixels = comma_separated[1:]

        # Get pixels as integers (we need to perform calculations later)
        pixels = [int(pixel) for pixel in comma_separated[1:]]

        return Observation(label, pixels)
    
    def  read_observations(self, data_path):
        """
        Read file at specified path and return list of Observation instances.

        Args: 
            data_path (str): Path to observations file.

        Returns: 
            List of Observation instances. 
        """
        
        with open(data_path, "r") as f:
            # Skip header
            next(f)
            data = f.readlines()
            data = map(self.observation_factory, data)

        return list(data)
        
class Distance:
    """
    Determines how different two images are by computing the "distance" between two arrays of pixels.
    This will use some algorithm such that the smaller the nunber the more similar are the images. 
    Distance also corresponds to 'cost' in the ML terminology.
    """
    def between(self, pixels1, pixels2):
        """
        Calculate the distance between two arrays of pixels.

        Args:
            pixels1 (list): The pixels respresenting the first image.
            pixels2 (list): The pixels respresenting the second image.
        """
        pass

class ManhattanDistance(Distance):
    """
    Compares two images pixel by pixel, computing each difference, and adding up their absolute values.
    Identical images will have a distance of zero, and the further apart two pixels are, the higher the distance between the two
    images will be. 
    """
    def between(self, pixels1, pixels2):
        """
        Compute the distance between two images. Identical images will have a distance of zero.

        Args:
            pixels1 (list): The pixels respresenting the first image.
            pixels2 (list): The pixels respresenting the second image.
        """
        assert len(pixels1) == len(pixels2), "Pixels lists should be the same length"

        length = len(pixels1)

        distance = 0

        for i in range(length):
            distance += abs(pixels1[i] - pixels2[i])

        return distance

class Classifier:

    def train(self, training_set):
        pass

    def predict(self, pixels):
        pass

class BasicClassifier(Classifier):
    """
    Classifies an image as a specific digit.
    """
    def __init__(self, distance):
        self.distance = distance

    def train(self, training_set):
        """
        Get a set of images for training.
        """
        self.data = training_set

    def predict(self, pixels):

        """
        Predict the digit that the image corresponds to.

        Args:
            pixels (list): The pixels respresenting the image.
        """
        current_best = None
        shortest = 1000000

        for obs in self.data:
            dist = self.distance.between(obs.pixels, pixels)

            if dist < shortest:
                shortest = dist
                current_best = obs

        return current_best.label

class Evaluator:
    """
    Evaluate a model (or any other model we want to try) by computing the proportion of classifications it gets right.
    """
    def correct(self, validation_set, classifier):
        """
        Compute the percentage of classifications the model gets right.

        Args:
            validation_set (list): The validation images.
            classifier: The classifier instance.
        """
        scores = [self.score(observation, classifier) for observation in validation_set]
        average = sum(scores) / len(scores)

        return average
        
    def score(self, observation, classifier):
        """
        "Score" the prediction by comparing what the classifier predicts with the true value. If they match,
        we record a 1, otherwise we record a 0. By using numbers like this rather than true/false values, we can
        average this out to get the percentage correct.

        Args:
            observation: The observation instance.
            classifier: The classifier instance.
        """
        print("Label: ", observation.label, end = " - ")
        
        if classifier.predict(observation.pixels) == observation.label:
            print("Match")
            return 1.0
        else:
            print("Mismatch")
            return 0.0

def main():
    try:
        t0 = datetime.now()

        distance = ManhattanDistance()
        classifier = BasicClassifier(distance)

        training_path = "trainingsample.csv"
        data_reader = DataReader()
        training = data_reader.read_observations(training_path)
        classifier.train(training)

        validation_path = "validationsample.csv"
        validation = data_reader.read_observations(validation_path)

        correct = Evaluator().correct(validation, classifier)
        print("Correctly classified: {0:.02%}".format(correct))

        print("time elapsed = {0:f} sec.\n".format((datetime.now() - t0).seconds))

    except Exception:
        traceback.print_exc()

if __name__ == "__main__": 
	main() 