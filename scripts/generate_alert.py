import os

import requests
from requests.auth import HTTPBasicAuth
import json

url = "https://mlops-units.atlassian.net/rest/api/3/issue"

api_key = os.environ["JIRA_KEY"]

auth = HTTPBasicAuth("stefano.chen@studenti.units.it", api_key)

headers = {
  "Accept": "application/json",
  "Content-Type": "application/json"
}

payload = json.dumps( {
    "fields": {
        "project": {
        "key": "HVP"
    },
    "summary": "Data drift Alert",
    "description": {
      "content": [
        {
          "content": [
            {
              "text": "Data drift detected for the House Value Prediction Model",
              "type": "text"
            }
          ],
          "type": "paragraph"
        }
      ],
      "type": "doc",
      "version": 1
    },
    "issuetype": {
        "name": "Bug"
    }
  }
})

response = requests.request(
   "POST",
   url,
   data=payload,
   headers=headers,
   auth=auth
)