import os
import requests

# API endpoint
url = "https://chat.genaiconnect.net/api/v1/chat.postMessage" #URL of RocketChat server, keep the same

# Headers with authentication tokens
headers = {
    "Content-Type": "application/json",
    "X-Auth-Token": "6WC3eGckAncEH-Y-j3PJEHyckOdHpyE-UCRrODUb5PI", #Replace with your bot token for local testing or keep it and store secrets in Koyeb
    "X-User-Id": "sFBSPATPjfTt9fJoS" #Replace with your bot user id for local testing or keep it and store secrets in Koyeb
}

# Payload (data to be sent)
payload = {
    "channel": "@BOT-Jiyoon", #Change this to your desired user, for any user it should start with @ then the username
    "text": "This is a direct message from the bot" #This where you add your message to the user
}

# Sending the POST request
response = requests.post(url, json=payload, headers=headers)

# Print response status and content
print(response.status_code)
print(response.json())  
