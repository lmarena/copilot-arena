import pandas as pd
from google.cloud import storage
import logging
import io

logger = logging.getLogger(__name__)


class GCPClient:
    def __init__(
        self,
        project_id: str = "code-arena",
        credentials_path: str = "config/firebase_config.json",
    ):
        """
        Initialize GCP client with project and credentials.

        Args:
            project_id: The Google Cloud project ID
            credentials_path: Path to the service account credentials JSON file
        """
        self.storage_client = storage.Client.from_service_account_json(
            credentials_path, project=project_id
        )

    def get_outcomes_df(self, bucket_name: str, file_path: str) -> pd.DataFrame:
        """
        Retrieve and parse the outcomes DataFrame from Google Cloud Storage.

        Args:
            bucket_name: Name of the GCS bucket
            file_path: Path to the CSV file within the bucket

        Returns:
            pandas.DataFrame containing the outcomes data
        """
        try:
            logger.info(f"Fetching outcomes data from gs://{bucket_name}/{file_path}")

            # Get bucket and blob
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(file_path)

            # Download the content as string
            content = blob.download_as_string()

            # Parse CSV content into DataFrame
            df = pd.read_csv(io.BytesIO(content))

            logger.info(f"Successfully loaded {len(df)} records from GCS")
            return df

        except Exception as e:
            logger.error(f"Error fetching outcomes data from GCS: {str(e)}")
            # Return empty DataFrame in case of error
            return pd.DataFrame()
