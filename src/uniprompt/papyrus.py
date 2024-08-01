import copy
import json
import os
import time
import msal
import requests

def chat_completion_papyrus(cache_path=None, **kwargs):
            papyrus_endpoint = "https://westus2.papyrus.binginternal.com/chat/completions"
            papyrus_client_id = os.environ.get("PAPYRUS_CLIENT_ID")
            papyrus_client_secret = os.environ.get("PAPYRUS_CLIENT_SECRET")
            verify_authority = "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47"
            verify_scope = "api://3dc78701-ac0b-4114-8173-bf1ea8c2e678/.default"

            client = msal.ConfidentialClientApplication(
                client_id=papyrus_client_id,
                client_credential=papyrus_client_secret,
                authority=verify_authority)
        
            acquire_token_result = client.acquire_token_for_client(scopes=[verify_scope])
            access_token = acquire_token_result["access_token"]

            model_kwargs = copy.deepcopy(kwargs["model_kwargs"])
            headers = {
                "Authorization": "Bearer " + access_token,
                "Content-Type": "application/json",
                "papyrus-model-name": model_kwargs["model"],
                "papyrus-quota-app-id": os.environ.get("PAPYRUS_APP_ID"),
                "papyrus-timeout-ms": "100000"
            }
            del model_kwargs["model"]

            while True:
                try:
                    json_dict = {"messages": kwargs["messages"], **model_kwargs}
                    response = requests.post(papyrus_endpoint, headers=headers, json=json_dict)
                    # print(response.text)
                    return json.loads(response.text)["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"Got error {e}. Sleeping for 5 seconds...")
                    # print(response.text)
                    time.sleep(5)

        


  