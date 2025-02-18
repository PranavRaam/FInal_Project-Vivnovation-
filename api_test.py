import requests
import pandas as pd
from time import sleep
from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
import os
from datetime import datetime
from collections import Counter


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PhysicianFinder:
    def __init__(self, file_path: str, output_dir: str = "physician_data"):
        """
        Initialize the PhysicianFinder with the MSA dataset file path and output directory.
        
        Args:
            file_path (str): Path to the MSA dataset CSV file
            output_dir (str): Directory to store JSON output files
        """
        try:
            self.msa_data = pd.read_csv(file_path)
            self.session = self._create_session()
            self.output_dir = output_dir
            self._ensure_output_directory()
        except FileNotFoundError:
            logger.error(f"MSA data file not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"MSA data file is empty: {file_path}")
            raise

    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def get_zip_codes_for_msa(self, msa_name: str) -> List[str]:
        """Retrieve all ZIP codes associated with a given MSA name."""
        mask = self.msa_data["MSA Title"].str.lower() == msa_name.lower()
        zip_codes = self.msa_data[mask]["ZIP"].astype(str).str.zfill(5).unique().tolist()
        return zip_codes

    def get_physicians(self, zip_code: str) -> List[Dict[str, Any]]:
        """Fetch physician details from the NPI API for a given ZIP code."""
        url = "https://npiregistry.cms.hhs.gov/api/"
        params = {
            "version": "2.1",
            "postal_code": zip_code,
            "taxonomy_description": "Physician",
            "limit": 200
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            physicians = data.get("results", [])
            
            for physician in physicians:
                physician['search_zip_code'] = zip_code
                
            sleep(0.5)
            return physicians
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for ZIP {zip_code}: {str(e)}")
            return []

    def fetch_all_physicians(self, zip_codes: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
        """Fetch physicians for multiple ZIP codes concurrently."""
        all_physicians = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_zip = {
                executor.submit(self.get_physicians, zip_code): zip_code 
                for zip_code in zip_codes
            }
            
            for future in as_completed(future_to_zip):
                zip_code = future_to_zip[future]
                try:
                    physicians = future.result()
                    all_physicians.extend(physicians)
                    logger.info(f"Processed ZIP {zip_code}: Found {len(physicians)} physicians")
                except Exception as e:
                    logger.error(f"Error processing ZIP {zip_code}: {str(e)}")
                    
        return all_physicians

    def format_physician_data(self, physician: Dict[str, Any]) -> Dict[str, Any]:
        """Format physician data into a clean JSON structure."""
        basic = physician.get("basic", {})
        addresses = physician.get("addresses", [{}])[0]
        taxonomies = physician.get("taxonomies", [])  # Get all taxonomies

        # Format all specialties/taxonomies
        specialties = [{
            "taxonomy_code": tax.get("code"),
            "taxonomy_desc": tax.get("desc"),
            "primary": tax.get("primary"),
            "state": tax.get("state"),
            "license": tax.get("license")
        } for tax in taxonomies]

        return {
            "npi": physician.get("number"),
            "personal_info": {
                "first_name": basic.get("first_name"),
                "last_name": basic.get("last_name"),
                "middle_name": basic.get("middle_name"),
                "credential": basic.get("credential"),
                "gender": basic.get("gender"),
                "sole_proprietor": basic.get("sole_proprietor")
            },
            "practice_location": {
                "address_line": addresses.get("address_1"),
                "address_line_2": addresses.get("address_2"),
                "city": addresses.get("city"),
                "state": addresses.get("state"),
                "postal_code": addresses.get("postal_code"),
                "telephone": addresses.get("telephone_number"),
                "fax": addresses.get("fax_number")
            },
            "specialties": specialties,
            "search_metadata": {
                "search_zip_code": physician.get("search_zip_code"),
                "retrieved_timestamp": datetime.now().isoformat()
            }
        }

    def analyze_specialties(self, physicians: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze and count all specialties across physicians."""
        specialty_counter = Counter()
        for physician in physicians:
            for taxonomy in physician.get("taxonomies", []):
                if taxonomy.get("desc"):
                    specialty_counter[taxonomy["desc"]] += 1
        return dict(specialty_counter.most_common())

    def save_physician_data(self, msa_name: str, physicians: List[Dict[str, Any]]) -> str:
        """
        Save physician data to a JSON file.
        
        Returns:
            str: Path to the saved JSON file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_msa_name = "".join(c if c.isalnum() else "_" for c in msa_name)
        filename = f"physicians_{safe_msa_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        specialty_counts = self.analyze_specialties(physicians)
        
        formatted_data = {
            "metadata": {
                "msa_name": msa_name,
                "query_timestamp": datetime.now().isoformat(),
                "total_physicians": len(physicians),
                "specialty_distribution": specialty_counts
            },
            "physicians": [self.format_physician_data(doc) for doc in physicians]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        return filepath

def main():
    try:
        finder = PhysicianFinder("cleaned_msa_county_zip.csv")
        
        while True:
            msa_name = input("\nEnter MSA Title (or 'quit' to exit): ").strip()
            if msa_name.lower() == 'quit':
                break
                
            zip_codes = finder.get_zip_codes_for_msa(msa_name)
            if not zip_codes:
                logger.warning("No ZIP codes found for this MSA.")
                continue
                
            logger.info(f"Found {len(zip_codes)} ZIP codes for {msa_name}")
            logger.info("Fetching physician data...")
            
            all_physicians = finder.fetch_all_physicians(zip_codes)
            
            if not all_physicians:
                logger.warning("No physicians found in this MSA.")
                continue
            
            # Analyze specialties
            specialty_counts = finder.analyze_specialties(all_physicians)
            
            # Save data to JSON file
            output_file = finder.save_physician_data(msa_name, all_physicians)
            
            logger.info(f"\nFound {len(all_physicians)} physicians in {msa_name}")
            logger.info("\nSpecialty Distribution:")
            for specialty, count in specialty_counts.items():
                logger.info(f"{specialty}: {count} physicians")
            logger.info(f"\nData saved to: {output_file}")

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
