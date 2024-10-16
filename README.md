# Sentiment Analysis with Caikit and Hugging Face

## Description

This project demonstrates how to use Caikit, an open-source AI toolkit, to load and infer a Hugging Face sentiment analysis model. The project uses the DistilBERT base uncased model fine-tuned on the SST-2 dataset to analyze text for sentiment.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Code Samples](#code-samples)
- [What does this mean?](#what-does-this-mean)
- [What I Learned](#what-i-learned)
- [Credits](#credits)
- [License](#license)

## Installation

This project was developed using the IBM SkillsBuild lab environment, which provides a preconfigured setup. If you're using the IBM lab, you won't need to install these components on your local machine. However, the project requires:

- Linux/MacOS x86_64 [Provided by the IBM SkillsBuild lab environment]
- Python (v3.8+) [Pre-installed in the lab environment]
- pip (v23.0+)
- Caikit (v0.9.2)

To set up the project in the IBM lab environment:

1. Open the terminal in the lab's IDE
2. Upgrade pip to the latest version:
   ```
   python -m pip install --upgrade pip
   ```
3. Create and activate a virtual environment:
   ```
   python -m venv env
   source env/bin/activate
   ```
4. Install Caikit and other required libraries:
   ```
   pip install caikit
   pip install 'caikit[runtime]'
   pip install 'caikit[nlp]'
   ```
5. Set up the Caikit runtime with the Hugging Face model (follow the specific instructions provided in the lab guide)

Note: If you're trying to reproduce this project outside of the IBM lab environment, you may need to adjust these steps to fit your local setup.

## Usage

To use the sentiment analysis model:

1. Activate the virtual environment: `source env/bin/activate`
2. Run the client application: `python client.py`
3. The application will analyze predefined text samples and display the sentiment analysis results

## Features

- Load and run a Hugging Face sentiment analysis model using Caikit runtime
- Analyze text samples for sentiment using a Python client application
- Display sentiment classification (POSITIVE/NEGATIVE) and confidence score for each text sample

## Code Samples

Here's a snippet from the client application that sends a request to the model:

```python
for text in ["I am not feeling well today!", "Today is a nice sunny day"]:
    input_text_proto = TextInput(text=text).to_proto()
    request = inference_service.messages.HuggingFaceSentimentTaskRequest(
       text_input=input_text_proto
    )
    response = client_stub.HuggingFaceSentimentTaskPredict(
       request, metadata=[("mm-model-id", "text_sentiment")]
    )
    print("Text:", text)
    print("RESPONSE:", response)
```

## What does this mean?

During the project, I observed some interesting results when analyzing single words, particularly gender-related terms:

```
Text: male
RESPONSE: classes {
  class_name: "NEGATIVE"
  confidence: 0.69973886013031006
}

Text: female
RESPONSE: classes {
  class_name: "NEGATIVE"
  confidence: 0.82222908735275269
}
```

Both "male" and "female" were classified as negative, which raises questions about the model's behavior and potential biases.

### Possible Reasons for the Negative Response

1. **Pretrained Model Bias:** The model may have been trained on datasets containing implicit biases, where these terms might have been associated with more negative contexts.

2. **Context-Free Text:** Sentiment analysis models often struggle with isolated words that lack context. "Male" and "female" alone might not provide enough semantic information for accurate classification.

3. **Model Type and Dataset:** The specific Hugging Face model used might be fine-tuned for particular contexts (e.g., product reviews) that don't handle neutral social terms well.

4. **Confidence Threshold:** The model shows high confidence in its negative classification (69.97% for "male" and 82.22% for "female"), indicating a strong belief in the negative sentiment for these words.

### How to Improve Results

1. **Add Context to Input:** Use the words in complete sentences to provide more context, e.g., "I am proud to be a male" or "Being a female is empowering."

2. **Try a Different Model:** Experiment with other Hugging Face models that might be better suited for short or context-limited text, such as:

   - `distilbert-base-uncased-finetuned-sst-2-english`
   - `nlptown/bert-base-multilingual-uncased-sentiment`

3. **Fine-Tune the Model:** If possible, fine-tune the model on a dataset with unbiased sentiment labels for your specific use case.

4. **Use Multiple Sentiment Classes:** Consider using a model that includes a "neutral" class in addition to positive and negative.

### Sample Code to Use a Different Hugging Face Model

```python
from transformers import pipeline

# Load a better-suited Hugging Face model for sentiment analysis
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Test with different inputs
```
