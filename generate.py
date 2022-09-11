import os
# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
import argparse
from dependencies import Vectorizer, generate, make_model


def generate_resulting_sequence(predict_model, vectorizer, length, prefix=None):
    # Get the prefix or generate it randomly
    if prefix is None:
        prefix = vectorizer.get_random_token()
    print(prefix)
    res = generate(model=predict_model, vectorizer=vectorizer, prefix=prefix.lower(), length=length)

    return res


if __name__ == "__main__" :

    # Parse the arguments from cmd
    parser = argparse.ArgumentParser("The sentence generating")
    parser.add_argument("--model", action="store", type=str, required=True, help="Input path to the model file")
    parser.add_argument("--length", action="store", type=int, required=True, help="Output text length")
    parser.add_argument("--prefix", action="store", type=str, required=False, help="The beginning of the sequence")

    args = parser.parse_args()

    length = args.length
    model_file = args.model
    prefix = args.prefix

    # Load the data to initialize a vectorizer
    data_file = "data/rus_text.txt"
    with open(data_file, encoding='utf-8') as input_file:
        text = input_file.read()

    pristine_input, pristine_output = False, False
    vectorizer = Vectorizer(text, True, pristine_input, pristine_output)

    # Initializing the generating model
    model = keras.models.load_model(model_file)
    predict_model = make_model(1, vectorizer.vocab_size)
    predict_model.set_weights(model.get_weights())

    # Generate the resulting sequence
    seq = generate_resulting_sequence(predict_model, vectorizer, length, prefix)

    print(seq)
