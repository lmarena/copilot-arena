import os
import base64
import json
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import aggregation
from src.privacy import PrivacySetting, clean_data
from dotenv import load_dotenv, find_dotenv
import pandas as pd


class FirebaseClient:
    def __init__(self):
        firebase_config_path = "config/firebase_config.json"
        if os.path.exists(firebase_config_path):
            self.cred = credentials.Certificate(firebase_config_path)
        else:
            load_dotenv(find_dotenv())
            encoded_key = os.getenv("FIREBASE_ACCOUNT_KEY")
            config_dict = json.loads(base64.b64decode(encoded_key).decode("utf-8"))
            self.cred = credentials.Certificate(config_dict)
        firebase_admin.initialize_app(self.cred)
        self.db = firestore.client()

    def upload_data(self, collection_name: str, data: dict, privacy: PrivacySetting):
        collection_ref = self.db.collection(collection_name)
        if privacy != PrivacySetting.RESEARCH:
            data = clean_data(data)
        collection_ref.add(data)

    def get_autocomplete_outcomes(
        self, collection_name: str, user_id: str = None, batch_size: int = 1000
    ):
        query = self.db.collection(collection_name)
        if user_id:
            query = query.where(filter=firestore.FieldFilter("userId", "==", user_id))
        dfs = []
        last_doc = None
        while True:
            current_query = query.limit(batch_size)
            if last_doc:
                current_query = current_query.start_after(last_doc)
            outcomes = current_query.get()
            if not outcomes:
                break
            df_batch = pd.DataFrame([x.to_dict() for x in outcomes])
            dfs.append(df_batch)
            last_doc = outcomes[-1]
        if dfs:
            outcomes_df = pd.concat(dfs, ignore_index=True)
        else:
            outcomes_df = pd.DataFrame()
        return outcomes_df

    def get_autocomplete_outcomes_count(
        self, collection_name: str, user_id: str = None
    ):
        query = self.db.collection(collection_name)
        if user_id:
            query = query.where(filter=firestore.FieldFilter("userId", "==", user_id))

        aggregate_query = aggregation.AggregationQuery(query)
        aggregate_query.count(alias="all")

        results = aggregate_query.get()
        return results[0][0].value
