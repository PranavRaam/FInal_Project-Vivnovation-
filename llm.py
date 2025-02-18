import json
import logging
from typing import Dict, List, Any
from huggingface_hub import InferenceClient

class PhysicianGroupAnalyzer:
    def __init__(self, api_key: str):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key
        )
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename="physician_grouping.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)
    
    def _create_grouping_prompt(self, physicians: List[Dict]) -> str:
        """Create a structured prompt for grouping analysis."""
        physician_details = []
        for p in physicians:
            location = p['practice_location']
            specialties = [s['taxonomy_desc'] for s in p.get('specialties', [])]
            
            detail = f"""
            Physician {len(physician_details) + 1}:
            - Name: {p['personal_info']['first_name']} {p['personal_info']['last_name']}
            - Specialties: {', '.join(specialties)}
            - Location: {location['address_line']}, {location['city']}, {location['state']}
            - Phone: {location['telephone']}
            """
            physician_details.append(detail)
            
        return f"""Analyze these physicians and group them based on:
1. Shared practice locations
2. Common phone numbers
3. Similar specialties at same locations
4. Geographic proximity

Physicians:
{'\n'.join(physician_details)}

Return the grouping as a JSON structure with:
1. Group name/identifier
2. Group type (e.g., "Multi-specialty Practice", "Solo Practice", "Hospital Affiliate")
3. List of physician indices in each group
4. Confidence score (0-1)
5. Reasoning for grouping"""

    def analyze_groups(self, physicians: List[Dict]) -> Dict:
        """Analyze and group physicians based on multiple factors."""
        try:
            # Create batches of 10 physicians for analysis
            batch_size = 10
            all_groups = []
            
            for i in range(0, len(physicians), batch_size):
                batch = physicians[i:i + batch_size]
                prompt = self._create_grouping_prompt(batch)
                
                # Query the model
                response = self.client.chat.completions.create(
                    model="mistralai/Mistral-7B-Instruct-v0.2",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in healthcare organization analysis. Analyze physician data to identify group practices and affiliations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                try:
                    # Parse the response and adjust indices for the batch
                    groups = json.loads(response.choices[0].message.content)
                    for group in groups:
                        group['physician_indices'] = [idx + i for idx in group['physician_indices']]
                        all_groups.append(group)
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse model response for batch {i//batch_size}")
                    continue
                
            return self._consolidate_groups(all_groups)
        except Exception as e:
            self.logger.error(f"Error in group analysis: {str(e)}")
            return []
    
    def _consolidate_groups(self, groups: List[Dict]) -> Dict:
        """Consolidate groups based on overlapping physicians and locations."""
        consolidated = {}
        
        for group in groups:
            group_id = group['group_name']
            if group_id not in consolidated:
                consolidated[group_id] = {
                    'group_name': group['group_name'],
                    'group_type': group['group_type'],
                    'physician_indices': set(group['physician_indices']),
                    'confidence': group['confidence'],
                    'reasoning': group['reasoning']
                }
            else:
                # Merge overlapping groups
                existing = consolidated[group_id]
                existing['physician_indices'].update(group['physician_indices'])
                existing['confidence'] = max(existing['confidence'], group['confidence'])
                
        # Convert sets back to lists for JSON serialization
        for group in consolidated.values():
            group['physician_indices'] = sorted(list(group['physician_indices']))
            
        return list(consolidated.values())

def process_physician_data(input_file: str, output_file: str, api_key: str):
    """Process physician data and add group information."""
    try:
        # Load physician data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Initialize analyzer
        analyzer = PhysicianGroupAnalyzer(api_key)
        
        # Analyze groups
        groups = analyzer.analyze_groups(data['physicians'])
        
        # Add group information to the data
        data['physician_groups'] = groups
        
        # Update individual physician records with their group affiliations
        physician_to_groups = {}
        for group in groups:
            for idx in group['physician_indices']:
                if idx not in physician_to_groups:
                    physician_to_groups[idx] = []
                physician_to_groups[idx].append({
                    'group_name': group['group_name'],
                    'group_type': group['group_type'],
                    'confidence': group['confidence']
                })
        
        for idx, physician in enumerate(data['physicians']):
            physician['group_affiliations'] = physician_to_groups.get(idx, [])
        
        # Save updated data
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        return True
    except Exception as e:
        logging.error(f"Error processing physician data: {str(e)}")
        return False

# Example usage:
if __name__ == "__main__":
    success = process_physician_data(
        input_file="physician_data/physicians_Utuado__PR_MicroSA_20250218_034313.json",
        output_file="physicians_with_groups.json",
        api_key="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"
    )
