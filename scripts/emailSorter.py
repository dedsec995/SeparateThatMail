import os
import base64
import re
import unicodedata
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from bs4 import BeautifulSoup
import torch
from transformers import pipeline

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
TOKEN_PATH = 'token.json'
CREDENTIALS_PATH = 'credentials.json'

def get_gmail_service():
    """Authenticates with the Gmail API and returns a service object."""
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                print(f"Error: '{CREDENTIALS_PATH}' not found.")
                print("Please download your credentials from the Google Cloud Console and place it in the root directory.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


def classify_email(text):
    device = -1
    try:
        classifier = pipeline(
            "text-classification",
            model="dedsec995/BERT_EMAIL_CLASSIFIER",
            tokenizer="dedsec995/BERT_EMAIL_CLASSIFIER",
            device=device
        )
        cleared_text = clean_text(text)
        result = classifier(cleared_text)
        predicted_label = result[0]['label']
        return predicted_label
    except Exception as e:
        print(f"An error occurred while loading the model or classifying the text: {e}")
        return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text(separator=' ', strip=True)
    clean_text = re.sub(r'http\S+|www\S+|https\S+', '', clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r'\S*@\S*\s?', '', clean_text)
    clean_text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', clean_text) # keeps letters, numbers, spaces, and basic punctuation
    clean_text = clean_text.lower()
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    clean_text = clean_text.strip('.-_ ') # Remove common leading/trailing artifacts
    clean_text = unicodedata.normalize('NFKD', clean_text).encode('ascii', 'ignore').decode('utf-8')

    return clean_text

def get_unread_emails(service):
    try:
        results = service.users().messages().list(userId='me', q='is:unread in:inbox').execute()
        messages = results.get('messages', [])
        return messages
    except HttpError as error:
        print(f"An error occurred while fetching emails: {error}")
        return []

def get_email_body(parts):
    if parts:
        for part in parts:
            if part.get('mimeType') == 'text/plain':
                data = part.get('body').get('data')
                return base64.urlsafe_b64decode(data).decode('utf-8')
            elif part.get('mimeType') == 'text/html':
                data = part.get('body').get('data')
                html = base64.urlsafe_b64decode(data).decode('utf-8')
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
            elif part.get('parts'):
                return get_email_body(part.get('parts'))
    return ""

def get_labels(service):
    try:
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        return {label['name']: label['id'] for label in labels}
    except HttpError as error:
        print(f"An error occurred while fetching labels: {error}")
        return {}

def process_emails():
    service = get_gmail_service()
    if not service:
        return

    labels = get_labels(service)
    applied_label_id = labels.get('Applied')
    reject_label_id = labels.get('Rejects')

    if not applied_label_id or not reject_label_id:
        print("Error: 'applied' or 'reject' labels not found in your Gmail account.")
        print("Please create these labels in Gmail and run the script again.")
        return

    messages = get_unread_emails(service)
    if not messages:
        print("No unread emails found.")
        return

    print(f"Found {len(messages)} unread emails. Starting classification...")

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
        payload = msg.get('payload')
        subject = next((header['value'] for header in payload['headers'] if header['name'] == 'Subject'), '')
        body = get_email_body(payload.get('parts'))
        email_text = f"Subject: {subject}\n\n{body}"

        predicted_label = classify_email(email_text)

        if predicted_label in ['applied', 'rejects']:
            label_id_to_apply = applied_label_id if predicted_label == 'applied' else reject_label_id
            print(f"Email classified as '{predicted_label}'. Applying label and moving.")
            service.users().messages().modify(
                userId='me',
                id=message['id'],
                body={
                    'addLabelIds': [label_id_to_apply],
                    'removeLabelIds': ['INBOX']
                }
            ).execute()
        else: # Normal
            print("Email classified as 'normal'. Leaving in inbox.")
            # Mark as read
            # service.users().messages().modify(
            #     userId='me',
            #     id=message['id'],
            #     body={'removeLabelIds': ['UNREAD']}
            # ).execute()


if __name__ == '__main__':
    process_emails()

