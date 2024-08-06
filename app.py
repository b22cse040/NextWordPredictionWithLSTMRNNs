from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('models/next_word_lstm.h5')

# Load the Tokenizer
with open('models/tokenizer.pkl', 'rb') as handle:
  tokenizer = pickle.load(handle)

app = Flask(__name__)

def predict_next_word(model, tokenizer, text, max_sequence_len):
  try:
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
      token_list = token_list[-(max_sequence_len - 1):]  # Ensure the sequence length matches max_sequence_len - 1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]  # Extract the index as an integer
    for word, index in tokenizer.word_index.items():
      if index == predicted_word_index:
        return word
    return None
  except Exception as e:
    return str(e)

@app.route('/Home')
def Welcome():
  return "Welcome to Next Word Prediction..."

@app.route('/PredictNextWord', methods=['GET', 'POST'])
def predict_next_word_route():
  word = None
  if request.method == 'POST':
    input_text = request.form['InputText']
    max_sequence_len = model.input_shape[1] + 1
    word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
  return render_template('home.html', word=word)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)