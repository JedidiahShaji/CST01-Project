import re
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # A simple HTML form for input


from transformers import BertTokenizer, BertForSequenceClassification
import torch


# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load('CST02_Model_2.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

def is_valid_sentence(sentence):
    # Check if the input has meaningful text (letters) and not just numbers or gibberish
    if sentence.strip() and re.match(r'[a-zA-Z]{2,}', sentence):  # At least two letters
        return True
    return False



@app.route('/classify', methods=['POST'])
def classify():
    sentence = request.form['sentence']  # Get the user input from the form

    # Validate the sentence
    if not is_valid_sentence(sentence):
        return jsonify({
            'input': sentence,
            'prediction': 'Invalid Input',
            'explanation': 'Please enter a valid sentence containing meaningful text.'
        })

    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # Perform inference
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()


        # Confidence threshold logic- t handle the non cultural sentences and context
        confidence = torch.softmax(logits, dim=1).max().item()
        if confidence < 0.4:  # Adjusting the threshold 
            return jsonify({'input': sentence, 'prediction': 'Unrelated to cultural context', 'explanation': 'The input does not relate to cultural norms or values.'})


        # Convert the prediction index to class label and explanation
        class_info = {
            0: {"label": "Appropriate", "explanation": "The sentence aligns well with cultural norms and is respectful."},
            1: {"label": "Inappropriate", "explanation": "The sentence may be offensive or disrespectful in a cultural context."},
            2: {"label": "Neutral", "explanation": "The sentence is neither strongly respectful nor offensive; it is neutral."}
        }
        predicted_label = class_info[prediction]["label"]
        explanation = class_info[prediction]["explanation"]

        return jsonify({'input': sentence, 'prediction': predicted_label, 'explanation': explanation})
    
    except Exception as e:
        return jsonify({
            'input': sentence,
            'error': 'An error occured during classification',
            'details': str(e)
        })


import csv
import os

from datetime import datetime

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    sentence = data.get('sentence')
    prediction = data.get('prediction')
    feedback = data.get('feedback')


    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save feedback to a persistent CSV file
    file_path = 'feedback.csv'

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Sentence', 'Prediction', 'Feedback'])  # Add headers
        writer.writerow([timestamp, sentence, prediction, feedback])

    return jsonify({'message': 'Feedback received!'}), 200






if __name__ == '__main__':
    app.run(debug=True)
