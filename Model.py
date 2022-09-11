import keras
from train import train_the_model
from generate import generate_resulting_sequence
from dependencies import make_model


class TextGenerator:
    def __init__(self):
        self.vectorizer = None
        self.model = None
    """
    A model that generates text using the LSTM neural network
    """
    def fit(self, input_dir, model_path, batch_size=32, num_epochs=50, seq_length=50):
        """
        Train the model
        :param input_dir: path for text file
        :param model_path: path where the model will be saved with the name of the model
        :param batch_size: size of one batch during the training
        :param num_epochs: amount of epochs
        :param seq_length: amount of words to remember
        """
        return_the_vectorizer = True
        return_model_path = True
        self.vectorizer, trained_model_path = train_the_model(input_dir, model_path, batch_size, num_epochs, seq_length,
                                                              return_the_vectorizer, return_model_path)

        model = keras.models.load_model(trained_model_path)
        self.model = make_model(1, self.vectorizer.vocab_size)
        self.model.set_weights(model.get_weights())

    def generate(self, length, prefix=None):
        """
        Generate the resulting text
        :param length: length of resulting text
        :param prefix: The first word of the text
        :return:
        """
        if self.vectorizer is None:
            return "Error. You have to train the model at first"

        generated_sequence = generate_resulting_sequence(self.model, self.vectorizer, length, prefix)

        return generated_sequence

