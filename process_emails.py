
import os.path
import base64
import re
import unicodedata
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
MODEL_PATH = "/home/dedsec995/SeparateThatMail/fine_tuned_bert_classifier"

def clean_text(text):
    """
    Applies a series of cleaning operations to a given text.
    """
    if not isinstance(text, str):
        return "" # Handle non-string inputs (e.g., None, NaN)

    # 1. Decode HTML entities (if any, though BeautifulSoup often handles this)
    # This step is often implicitly handled by BeautifulSoup's get_text()
    # text = html.unescape(text) # Requires import html

    # 2. Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text(separator=' ', strip=True) # strip=True removes leading/trailing whitespace from each line

    # 3. Remove URLs
    clean_text = re.sub(r'http\S+|www\S+|https\S+', '', clean_text, flags=re.MULTILINE)

    # 4. Remove email addresses (optional, but emails often contain unique IDs/names)
    clean_text = re.sub(r'\S*@\S*\s?', '', clean_text)

    # 5. Remove non-alphanumeric characters (keeping spaces and some punctuation for now)
    # This regex keeps letters, numbers, and basic punctuation, and whitespace
    # You might adjust this depending on how much punctuation you want to keep.
    # For general classification, removing most is common.
    # Let's keep alphanumeric and basic spaces initially.
    clean_text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', clean_text) # keeps letters, numbers, spaces, and basic punctuation

    # 6. Convert to lowercase
    clean_text = clean_text.lower()

    # 7. Remove extra whitespaces (tabs, multiple spaces, newlines)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # 8. Remove leading/trailing specific characters that might remain
    clean_text = clean_text.strip('.-_ ') # Remove common leading/trailing artifacts

    # 9. Handle Unicode characters (e.g., accented characters to ASCII equivalents)
    clean_text = unicodedata.normalize('NFKD', clean_text).encode('ascii', 'ignore').decode('utf-8')

    return clean_text

def get_email_body(msg):
    if msg['payload']['mimeType'] == 'text/plain':
        return base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8')
    elif 'parts' in msg['payload']:
        for part in msg['payload']['parts']:
            if part['mimeType'] == 'text/plain':
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
    return ""

def main():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build("gmail", "v1", credentials=creds)

        # Load the fine-tuned model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)

        # Get labels
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        label_map = {label['name']: label['id'] for label in labels}
        if 'accept' not in label_map:
            label_body = {'name': 'accept', 'labelListVisibility': 'labelShow', 'messageListVisibility': 'show'}
            label = service.users().labels().create(userId='me', body=label_body).execute()
            label_map['accept'] = label['id']
        if 'reject' not in label_map:
            label_body = {'name': 'reject', 'labelListVisibility': 'labelShow', 'messageListVisibility': 'show'}
            label = service.users().labels().create(userId='me', body=label_body).execute()
            label_map['reject'] = label['id']


        # List unread messages
        results = service.users().messages().list(userId="me", q="is:unread -has:userlabels").execute()
        messages = results.get("messages", [])

        if not messages:
            print("No unread messages found.")
            return

        print("Processing unread messages...")
        for message in messages:
            msg = service.users().messages().get(userId="me", id=message["id"]).execute()
            email_body = get_email_body(msg)
            
            if not email_body:
                continue

            cleaned_body = clean_text(email_body)
            inputs = tokenizer(cleaned_body, return_tensors="tf", truncation=True, padding=True)
            outputs = model(inputs)
            prediction = tf.argmax(outputs.logits, axis=-1).numpy()[0]
            
            label_name = 'accept' if prediction == 1 else 'reject'
            label_id = label_map[label_name]
            
            # Apply the label and move the email
            service.users().messages().modify(userId='me', id=message['id'], body={'addLabelIds': [label_id], 'removeLabelIds': ['INBOX']}).execute()
            
            headers = msg["payload"]["headers"]
            subject = next(header["value"] for header in headers if header["name"] == "Subject")
            print(f"- Subject: {subject} -> Labeled as: {label_name}")

    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    main()
