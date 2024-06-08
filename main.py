# an object of WSGI application 
import os

import pandas as pd
import numpy as np
import torch
import pickle

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import pipeline

from flask import Flask, render_template, request, jsonify, send_file
app = Flask(__name__)   # Flask constructor 

def split_document(document, max_length=512, overlap=50):
    segments = []
    words = document.split()
    current_segment = ""
    for word in words:
        if len(current_segment) + len(word) < max_length - overlap:
            current_segment += word + " "
        else:
            segments.append(current_segment.strip())
            current_segment = word + " "

        segments.append(current_segment.strip())
    return segments

def init_model():
    dsqa = pd.read_csv("data.csv")

    input_ids = []

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    for index, row in dsqa.iterrows():
        text = row["paragraph"]
        # Do something with the age value for each row
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True,max_length=500)
        input_ids.append(ids)

    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    return model, tokenizer, input_ids

model, tokenizer, input_ids = init_model()
  
# A decorator used to tell the application 
# which URL is associated function 
@app.route('/')
def index(): 
    return render_template("index.html")

@app.route("/get-answer", methods=["POST"])
def grammar_check():
    
    question = request.data.decode("utf-8")

    print(question)
    
    # Tokenize the question
    question_ids = tokenizer.encode(question, add_special_tokens=True, truncation=True,max_length=500)

    answers = []
    confidences = []
    
    for ids in input_ids:
        # Combine the question and text into one input
        temp_ids = question_ids + [tokenizer.sep_token_id] + ids
        
        # Segment IDs
        segment_ids = [0] * len(question_ids) + [1] * (len(temp_ids) - len(question_ids))
        
        # Model output
        output = model(torch.tensor([temp_ids]), token_type_ids=torch.tensor([segment_ids]))
        
        # Reconstructing the answer
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)
        tokens = tokenizer.convert_ids_to_tokens(temp_ids)
        answer = ""
        if answer_end >= answer_start:
            for i in range(answer_start+1, answer_end+1):
                if tokens[i][0:2] == "##":
                    answer += tokens[i][2:]
                else:
                    answer += " " + tokens[i]
                    
        if len(answer) < 6:
            continue

        # Compute confidence
        start_confidence = torch.max(output.start_logits).item()
        end_confidence = torch.max(output.end_logits).item()
        confidence = (start_confidence + end_confidence) / 0.2  # Using the average of start and end confidence
        
        answers.append(answer)
        confidences.append(confidence)
        # print("Answer: {} (conf: {})\n".format(answer, confidence))

    if len(answers) > 0:
        sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
        sorted_answers = [answers[i] for i in sorted_indices]
        sorted_confidences = [confidences[i] for i in sorted_indices]

        # Select the second highest confidence
        second_most_confident_answer = sorted_answers[1]
        second_most_confident_confidence = sorted_confidences[1]
        
        # Find the index of the answer with the highest confidence
        most_confident_answer = sorted_answers[0]
        most_confident_confidence = sorted_confidences[0]
        
        return jsonify({"answer": most_confident_answer.capitalize(), "confidence": most_confident_confidence, "alternate": second_most_confident_answer, "alternate_confidence": second_most_confident_confidence})
    else:  
        return jsonify({"answer": "System don't have the answer for your question", "confidence": 1})
  

if __name__=='__main__': 
   app.run() 