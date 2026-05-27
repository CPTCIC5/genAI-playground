import os
import json
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")


def get_service():
    """Authenticate and return Gmail service using a local token.json file."""
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def list_messages(service, user_id="me", max_results=10, query=""):
    """List messages from Gmail inbox."""
    try:
        results = (
            service.users()
            .messages()
            .list(userId=user_id, maxResults=max_results, q=query)
            .execute()
        )
        return results.get("messages", [])
    except HttpError as error:
        print(f"An error occurred: {error}")
        return []


def get_message(service, message_id, user_id="me"):
    """Get a specific message by ID."""
    try:
        return (
            service.users()
            .messages()
            .get(userId=user_id, id=message_id)
            .execute()
        )
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


def get_message_content(message):
    """Extract readable content from a Gmail message."""
    payload = message.get("payload", {})
    headers = payload.get("headers", [])

    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
    from_email = next((h["value"] for h in headers if h["name"] == "From"), "")
    to_email = next((h["value"] for h in headers if h["name"] == "To"), "")
    date = next((h["value"] for h in headers if h["name"] == "Date"), "")

    body = ""
    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data")
                if data:
                    body = base64.urlsafe_b64decode(data).decode("utf-8")
                    break
            elif part["mimeType"] == "text/html":
                data = part["body"].get("data")
                if data:
                    body = base64.urlsafe_b64decode(data).decode("utf-8")
    else:
        if payload.get("mimeType") == "text/plain":
            data = payload["body"].get("data")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8")

    return {
        "id": message.get("id"),
        "subject": subject,
        "from": from_email,
        "to": to_email,
        "date": date,
        "body": body,
        "snippet": message.get("snippet", ""),
    }


def send_email(service, to, subject, body, user_id="me", is_html=False):
    """Send an email via Gmail."""
    try:
        message = MIMEText(body, "html" if is_html else "plain")
        message["to"] = to
        message["subject"] = subject

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        return (
            service.users()
            .messages()
            .send(userId=user_id, body={"raw": raw_message})
            .execute()
        )
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


def send_email_with_attachment(service, to, subject, body, attachment_path, user_id="me", is_html=False):
    """Send an email with attachment via Gmail."""
    try:
        message = MIMEMultipart()
        message["to"] = to
        message["subject"] = subject

        msg_body = MIMEText(body, "html" if is_html else "plain")
        message.attach(msg_body)

        if os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                attachment = MIMEText(f.read(), "base64")
                attachment.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{os.path.basename(attachment_path)}"',
                )
                message.attach(attachment)

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        return (
            service.users()
            .messages()
            .send(userId=user_id, body={"raw": raw_message})
            .execute()
        )
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


def delete_message(service, message_id, user_id="me"):
    """Delete a message by ID."""
    try:
        service.users().messages().delete(userId=user_id, id=message_id).execute()
        return True
    except HttpError as error:
        print(f"An error occurred: {error}")
        return False


def mark_as_read(service, message_id, user_id="me"):
    """Mark a message as read."""
    try:
        return (
            service.users()
            .messages()
            .modify(userId=user_id, id=message_id, body={"removeLabelIds": ["UNREAD"]})
            .execute()
        )
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


def mark_as_unread(service, message_id, user_id="me"):
    """Mark a message as unread."""
    try:
        return (
            service.users()
            .messages()
            .modify(userId=user_id, id=message_id, body={"addLabelIds": ["UNREAD"]})
            .execute()
        )
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


def main():
    """Example usage of Gmail functions."""
    service = get_service()

    messages = list_messages(service, max_results=5)
    print(f"Found {len(messages)} messages")

    if messages:
        message = get_message(service, messages[0]["id"])
        if message:
            content = get_message_content(message)
            print(json.dumps(content, indent=2))


if __name__ == "__main__":
    main()
