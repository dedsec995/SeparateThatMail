import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch

def train_and_evaluate_bert(data_path="clean_data.csv", model_name="bert-base-uncased"):
    """
    Loads data, trains a BERT model for text classification, and evaluates it.
    """
    print(f"--- Starting BERT Fine-tuning with {model_name} ---")

    # --- 1. Load the Dataset ---
    print(f"Loading data from '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: '{data_path}' not found. Please ensure the cleaned data is in the correct directory.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Filter out any rows with empty text if not already handled by cleaning script
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']

    if df.empty:
        print("Error: Dataset is empty after loading/cleaning. Cannot proceed with training.")
        return

    print(f"Dataset loaded. Total samples: {len(df)}")
    print("Label distribution:\n", df['label'].value_counts())

    # --- 2. Label Encoding ---
    # Create a mapping from string labels to numerical IDs
    unique_labels = df['label'].unique()
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    num_labels = len(unique_labels)
    print(f"Found {num_labels} unique labels: {unique_labels}")
    print(f"Label to ID mapping: {label_to_id}")

    df['labels'] = df['label'].map(label_to_id)

    # Convert pandas DataFrame to Hugging Face Dataset
    # Hugging Face 'datasets' expects 'text' and 'labels' columns by default for sequence classification
    hf_dataset = Dataset.from_pandas(df[['text', 'labels']])

    # --- 3. Split Dataset into Training and Validation ---
    # Stratify by labels to maintain class distribution in splits
    train_test_split_dataset = hf_dataset.train_test_split(test_size=0.2, stratify_by_column="labels", seed=42)
    
    # Further split test into validation and test (optional, but good practice)
    # If your dataset is small, you might skip a separate test set and just use validation.
    # For now, let's create a validation split from the 20% test_size.
    # This might make the test set too small if overall data is limited.
    # A common split is 80% train, 10% validation, 10% test.
    # We will do 80% train, 20% validation/test for simplicity and then evaluate on the test split.

    train_dataset = train_test_split_dataset["train"]
    eval_dataset = train_test_split_dataset["test"] # This will be our evaluation set

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    # --- 4. Tokenization ---
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Function to tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # --- 5. Model Loading ---
    print(f"Loading pre-trained model for sequence classification: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=id_to_label, # Pass mappings for easier output interpretation
        label2id=label_to_id
    )

    # --- 6. Training Setup ---
    # Define metrics
    accuracy = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1") # Use f1 score for multi-class classification

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate accuracy
        acc = accuracy.compute(predictions=predictions, references=labels)
        
        # Calculate F1-score (macro average is often good for imbalanced classes)
        # 'average=weighted' is also a good option if classes are imbalanced
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted") 
        
        return {"accuracy": acc["accuracy"], "f1_weighted": f1["f1"]}

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        learning_rate=2e-5,          # Common learning rate for fine-tuning BERT
        per_device_train_batch_size=8, # Adjust based on GPU memory. Lower if OOM.
        per_device_eval_batch_size=8,
        num_train_epochs=3,          # Number of passes over the training data
        weight_decay=0.01,           # Regularization
        save_total_limit=2,          # Keep only the last 2 checkpoints
        load_best_model_at_end=True, # Load the best model after training
        metric_for_best_model="f1_weighted", # Or "accuracy"
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none", # You can set this to "wandb" or "tensorboard" for better tracking
        fp16=torch.cuda.is_available() # Enable mixed precision if GPU is available
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # --- 7. Training ---
    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # --- 8. Evaluation ---
    print("\n--- Evaluating the model on the validation set ---")
    eval_results = trainer.evaluate()
    print(eval_results)

    # --- 9. Save the Fine-tuned Model ---
    model_save_path = "./fine_tuned_bert_classifier"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nFine-tuned model and tokenizer saved to: '{model_save_path}'")
    print("You can now load this model for inference.")

    return trainer, tokenizer, model_save_path, id_to_label

if __name__ == "__main__":
    # Check for GPU
    if torch.cuda.is_available():
        print("CUDA is available! Training will use GPU.")
    else:
        print("CUDA is not available. Training will use CPU, which will be significantly slower.")

    # You can change 'bert-base-uncased' to other models like 'distilbert-base-uncased'
    # DistilBERT is smaller and faster, good for starting out.
    trainer_obj, tokenizer_obj, model_path, id_to_label_map = train_and_evaluate_bert(model_name="bert-base-uncased")

    # Example of loading the saved model and making a prediction
    print("\n--- Testing the saved model with a sample text ---")
    from transformers import pipeline

    # Load the pipeline from the saved model
    classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=0 if torch.cuda.is_available() else -1) # 0 for GPU, -1 for CPU

    sample_texts = [
        "Dear candidate, We regret to inform you that your application for the Software Engineer position was not selected at this time.",
        "Congratulations! We are pleased to offer you the position of Data Scientist. Please review the attached offer letter.",
        "Thank you for applying to the Marketing Specialist role at our company. We have received your application and will review it shortly.",
        "Your interview for the Senior Analyst role is scheduled for next Tuesday at 10 AM PST."
    ]

    print("\nPredictions for sample texts:")
    for text in sample_texts:
        result = classifier(text)
        predicted_label = result[0]['label']
        predicted_score = result[0]['score']
        print(f"Text: '{text[:80]}...'")
        print(f"  Predicted Label: {predicted_label} (Confidence: {predicted_score:.4f})")