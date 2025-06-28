import torch
from transformers import pipeline
import argparse

def classify_email(text, model_path="./fine_tuned_bert_classifier"):
    """
    Classifies an email text using the fine-tuned BERT model.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU for inference.")
        device = -1
    else:
        print("CUDA is available, using GPU for inference.")
        device = 0

    try:
        # Load the text classification pipeline with the fine-tuned model
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=device
        )
        
        # Get the prediction from the classifier
        result = classifier(text)
        
        predicted_label = result[0]['label']
        confidence_score = result[0]['score']
        
        return predicted_label, confidence_score

    except Exception as e:
        print(f"An error occurred while loading the model or classifying the text: {e}")
        print("Please ensure that a trained model exists in the './fine_tuned_bert_classifier' directory.")
        return None, None

def main():
    """
    Main function to handle command-line arguments for email classification.
    """
    parser = argparse.ArgumentParser(description="Classify an email text using the fine-tuned BERT model.")
    
    # Add an argument for the email text
    parser.add_argument("email_text", type=str, help="The email text to classify, enclosed in quotes.")
    
    args = parser.parse_args()
    
    email_text_to_classify = args.email_text
    
    print(f"Classifying the following email text:{email_text_to_classify}")
    
    predicted_label, confidence = classify_email(email_text_to_classify)
    
    if predicted_label and confidence:
        print(f"\nPrediction complete:")
        print(f"  Predicted Label: {predicted_label}")
        print(f"  Confidence Score: {confidence:.4f}")

if __name__ == "__main__":
    main()
