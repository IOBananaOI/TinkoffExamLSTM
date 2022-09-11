import os
import argparse
from dependencies import load_data, make_model, train
import sys
# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_the_model(input_dir, model_path, batch_size=32,
                    num_epochs=150, seq_length=50, return_the_vectorizer=False, return_model_path=False):
    # Load text data
    pristine_input, pristine_output = False, False
    x, y, x_val, y_val, vectorizer = load_data(input_dir, True, pristine_input, pristine_output, batch_size, seq_length)

    # Make model
    model = make_model(batch_size, vectorizer.vocab_size)

    # Train model
    train(model, x, y, x_val, y_val, batch_size, num_epochs)

    # Save model to file

    model.save(filepath=model_path)
    print("The model has successfully trained and saved")
    if return_the_vectorizer and return_model_path:
        return vectorizer, model_path


if __name__ == "__main__":

    # Parse the arguments from cmd
    parser = argparse.ArgumentParser("Training the model")
    parser.add_argument("--input-dir", action="store", type=str, required=False, help="Input path to the data file")
    parser.add_argument("--model", action="store", type=str, required=True, help="Directory for the trained model")

    args = parser.parse_args()

    data_file = args.input_dir

    if args.input_dir is None:
        std_in = sys.stdin
        for line in std_in:
            input_dir = line.strip()
            break
    else:
        input_dir = args.input_dir

    model_file = args.model

    train_the_model(input_dir, model_file)

