import os.path
import base64
from email.message import EmailMessage
import pandas as pd
import re

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, or wish to relogin, delete the file token.json.
# 'https://www.googleapis.com/auth/gmail.modify' For future as it is needed to add/remove labels later.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def authenticate_gmail_api():
    """Authenticates with the Gmail API and returns the service object."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
     
    try:
        service = build("gmail", "v1", credentials=creds)
        print("Successfully authenticated with Gmail API.")
        return service
    except HttpError as error:
        print(f"An error occurred during authentication: {error}")
        return None

def get_message_plain_text(payload):
    """
    Extracts the plain text body from an email payload.
    Handles multipart messages by recursively searching for text/plain parts.
    """
    if "parts" in payload:
        for part in payload["parts"]:
            # Prioritize text/plain over text/html if both exist
            if part["mimeType"] == "text/plain":
                if "data" in part["body"]:
                    return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
            # Recursively check nested parts
            recursive_body = get_message_plain_text(part)
            if recursive_body:
                return recursive_body
        for part in payload["parts"]:
            if part["mimeType"] == "text/html":
                if "data" in part["body"]:
                    # Simple regex to strip HTML tags for general text extraction
                    html_content = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
                    return re.sub(r'<[^>]+>', '', html_content) # Basic HTML stripping
    elif "body" in payload and "data" in payload["body"]:
        # Direct body data (non-multipart)
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
    return "" # No body found

def get_emails_by_label(service, label_name, max_results=500):
    """
    Fetches emails with a specific label and extracts their subject and body.
    """
    email_data = []
     
    try:
        # First, get the label ID
        results = service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
         
        label_id = None
        for label in labels:
            if label["name"].lower() == label_name.lower():
                label_id = label["id"]
                break
         
        if not label_id:
            print(f"Label '{label_name}' not found in your Gmail account. Please create it.")
            return []

        print(f"Fetching emails with label '{label_name}' (ID: {label_id})...")
         
        # List messages with the given label
        response = service.users().messages().list(
            userId="me", 
            labelIds=[label_id], 
            maxResults=max_results
        ).execute()
         
        messages = response.get("messages", [])
         
        if not messages:
            print(f"No messages found with label '{label_name}'.")
            return []

        for i, msg in enumerate(messages):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(messages)} emails for '{label_name}'...")

            msg_details = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
             
            payload = msg_details["payload"]
            headers = payload.get("headers", [])
             
            subject = ""
            for header in headers:
                if header["name"] == "Subject":
                    subject = header["value"]
                    break
             
            body = get_message_plain_text(payload)
             
            # Combine subject and body for the text feature
            full_text = f"Subject: {subject}\n\n{body}".strip()
             
            if full_text: # Only add if there's actual content
                email_data.append({"text": full_text, "label": label_name.lower()})
            else:
                print(f"Skipping empty or unreadable email ID: {msg['id']} from label '{label_name}'")

        print(f"Finished fetching {len(email_data)} emails for label '{label_name}'.")
        return email_data

    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

# NEW FUNCTION to get 'Normal' emails based on a search query
def get_normal_emails(service, max_results=500):
    """
    Fetches emails that are in the inbox, read, and have no user labels.
    """
    email_data = []
    # This query finds all emails in the inbox that are read and have no user-defined labels
    query = "in:inbox is:read has:nouserlabels"
    
    print(f"Fetching 'Normal' emails with query: '{query}'...")
    
    try:
        # List messages matching the query
        response = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = response.get("messages", [])
        
        if not messages:
            print("No messages found matching the 'Normal' criteria.")
            return []

        for i, msg in enumerate(messages):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(messages)} 'Normal' emails...")

            msg_details = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
            
            payload = msg_details["payload"]
            headers = payload.get("headers", [])
            
            subject = ""
            for header in headers:
                if header["name"] == "Subject":
                    subject = header["value"]
                    break
            
            body = get_message_plain_text(payload)
            
            # Combine subject and body for the text feature
            full_text = f"Subject: {subject}\n\n{body}".strip()
            
            if full_text: # Only add if there's actual content
                email_data.append({"text": full_text, "label": "normal"})
            else:
                print(f"Skipping empty or unreadable 'Normal' email ID: {msg['id']}")

        print(f"Finished fetching {len(email_data)} 'Normal' emails.")
        return email_data

    except HttpError as error:
        print(f"An error occurred while fetching 'Normal' emails: {error}")
        return []

def main():
    service = authenticate_gmail_api()
    if not service:
        return

    all_email_data = []
    # MODIFIED: List of labels to fetch by name
    labels_to_fetch = ["Applied", "Rejects"]

    for label in labels_to_fetch:
        data = get_emails_by_label(service, label, max_results=500)
        all_email_data.extend(data)
    
    # MODIFIED: Add the call to the new function for 'Normal' emails
    normal_data = get_normal_emails(service, max_results=500)
    all_email_data.extend(normal_data)

    if all_email_data:
        df = pd.DataFrame(all_email_data)
        output_filename = "data.csv"
        df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully gathered a total of {len(df)} emails and saved to '{output_filename}'")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        print("\nLabel distribution:")
        print(df['label'].value_counts())
    else:
        print("No email data was gathered. Please check your labels and API setup.")

if __name__ == "__main__":
    main()