import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import pandas as pd
import pydeck as pdk
from huggingface_hub import InferenceClient
from opencage.geocoder import OpenCageGeocode
from opencage.geocoder import InvalidInputError, RateLimitExceededError, UnknownError
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class Location:
    latitude: float
    longitude: float
    confidence: str

@dataclass
class PhysicianData:
    npi: str
    name: str
    specialty: str
    address: str
    city: str
    state: str
    postal_code: str
    phone: str
    latitude: Optional[float]
    longitude: Optional[float]
    group_affiliations: List[Dict] = None

class PhysicianAnalyzer:
    def __init__(self, hf_api_key: str, geocoding_api_key: str):
        self.hf_client = InferenceClient(
            provider="hf-inference",
            api_key=hf_api_key
        )
        self.geocoder = OpenCageGeocode(geocoding_api_key)
        self.rate_limit = 1  # Minimum time between geocoding requests
        self.df: Optional[pd.DataFrame] = None
        self.deck: Optional[pdk.Deck] = None
        self.legend_data = []

    def _setup_logger(self):
        return logger

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

    def _format_address(self, address_components: Dict[str, Any]) -> str:
        address = re.sub(
            r'\s+(?:Suite|Ste|Apt|Unit|#)\s*[\w-]+',
            '',
            str(address_components.get("address_line", "")).strip(),
            flags=re.IGNORECASE
        )
        
        zip_code = address_components.get("postal_code", "")
        zip_str = str(zip_code).strip()
        zip_digits = re.sub(r'\D', '', zip_str)
        formatted_zip = (f"{zip_digits[:5]}-{zip_digits[5:9]}" if len(zip_digits) >= 9 
                        else zip_digits[:5] if len(zip_digits) >= 5 
                        else zip_str)
        
        components = [
            address,
            str(address_components.get("city", "")).strip(),
            str(address_components.get("state", "")).strip(),
            formatted_zip[:5],
            "USA"
        ]
        
        return ", ".join(filter(None, components))

    def _geocode_address(self, address: str) -> Optional[Location]:
        for attempt in range(3):
            try:
                time.sleep(self.rate_limit)
                results = self.geocoder.geocode(address)
                
                if results and len(results):
                    result = results[0]
                    confidence = 'high' if result['confidence'] > 7 else 'low'
                    return Location(
                        latitude=result['geometry']['lat'],
                        longitude=result['geometry']['lng'],
                        confidence=confidence
                    )
                    
                time.sleep(2 * (attempt + 1))
            except RateLimitExceededError:
                logger.warning("Rate limit exceeded, waiting longer...")
                time.sleep(60)
            except (InvalidInputError, UnknownError) as e:
                logger.warning(f"Geocoding error: {str(e)}")
                time.sleep(2 * (attempt + 1))
        return None

    def analyze_groups(self, physicians: List[Dict]) -> Dict:
        """Analyze and group physicians based on multiple factors."""
        try:
            batch_size = 10
            all_groups = []
            
            for i in range(0, len(physicians), batch_size):
                batch = physicians[i:i + batch_size]
                prompt = self._create_grouping_prompt(batch)
                
                response = self.hf_client.chat.completions.create(
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
                    groups = json.loads(response.choices[0].message.content)
                    for group in groups:
                        group['physician_indices'] = [idx + i for idx in group['physician_indices']]
                        all_groups.append(group)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse model response for batch {i//batch_size}")
                    continue
                
            return self._consolidate_groups(all_groups)
        except Exception as e:
            logger.error(f"Error in group analysis: {str(e)}")
            return []

    def _consolidate_groups(self, groups: List[Dict]) -> List[Dict]:
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
                existing = consolidated[group_id]
                existing['physician_indices'].update(group['physician_indices'])
                existing['confidence'] = max(existing['confidence'], group['confidence'])
                
        for group in consolidated.values():
            group['physician_indices'] = sorted(list(group['physician_indices']))
            
        return list(consolidated.values())

    def _process_physician(self, doc: Dict[str, Any]) -> PhysicianData:
        location = doc.get("practice_location", {})
        address = self._format_address(location)
        
        lat = location.get("latitude")
        lon = location.get("longitude")
        
        try:
            lat = float(lat) if lat else None
            lon = float(lon) if lon else None
        except (ValueError, TypeError):
            lat, lon = None, None
        
        if not (lat and lon and -90 <= lat <= 90 and -180 <= lon <= 180):
            if geocoded := self._geocode_address(address):
                lat = geocoded.latitude
                lon = geocoded.longitude
        
        specialties = doc.get("specialties", [])
        specialty_desc = ", ".join(
            str(tax.get("taxonomy_desc", ""))
            for tax in specialties
            if isinstance(tax, dict) and tax.get("taxonomy_desc")
        ) or "Unknown Specialty"
        
        personal_info = doc.get("personal_info", {})
        name = f"{personal_info.get('first_name', '')} {personal_info.get('last_name', '')}".strip() or "Unknown Name"
        
        return PhysicianData(
            npi=str(doc.get("npi", "N/A")),
            name=name,
            specialty=specialty_desc,
            address=address,
            city=str(location.get("city", "Unknown")),
            state=str(location.get("state", "Unknown")),
            postal_code=str(location.get("postal_code", "")),
            phone=str(location.get("telephone", "Unknown")),
            latitude=lat,
            longitude=lon,
            group_affiliations=doc.get("group_affiliations", [])
        )

    def _cluster_points(self, df: pd.DataFrame) -> pd.DataFrame:
        coords = df[['latitude', 'longitude']].values
        clustering = DBSCAN(eps=0.0001, min_samples=1).fit(coords)
        
        df = df.copy()
        df['cluster'] = clustering.labels_
        
        return df.groupby('cluster').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'name': lambda x: f"{len(x)} physicians" if len(x) > 1 else x.iloc[0],
            'specialty': lambda x: "; ".join(sorted(set(x))),
            'color': 'first',
            'address': 'first',
            'phone': 'first',
            'group_affiliations': 'first'
        }).reset_index()

    def process_data(self, input_file: str) -> None:
        """Process physician data with both grouping and visualization."""
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            physicians = data.get("physicians", [])
            if not physicians:
                raise ValueError("No physician data found")
            
            # Analyze groups first
            groups = self.analyze_groups(physicians)
            
            # Update physician records with group affiliations
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
            
            for idx, physician in enumerate(physicians):
                physician['group_affiliations'] = physician_to_groups.get(idx, [])
            
            # Process for visualization
            with ThreadPoolExecutor(max_workers=4) as executor:
                processed_data = list(executor.map(self._process_physician, physicians))
            
            self.df = pd.DataFrame([vars(p) for p in processed_data])
            self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
            self.df['longitude'] = pd.to_numeric(self.df['longitude'], errors='coerce')
            self.df.dropna(subset=['latitude', 'longitude'], inplace=True)
            
            if self.df.empty:
                raise ValueError("No valid coordinates found")
            
            # Save processed data
            output_data = {
                "physicians": physicians,
                "physician_groups": groups
            }
            
            output_path = Path(input_file).with_name("processed_" + Path(input_file).name)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

    def generate_visualization(self) -> None:
        """Generate the visualization after processing data."""
        if self.df is None or self.df.empty:
            raise ValueError("No data available")

        specialties = self.df['specialty'].unique()
        colors = [
            [86, 180, 233, 200],
            [230, 159, 0, 200],
            [0, 158, 115, 200],
            [240, 228, 66, 200],
            [0, 114, 178, 200],
            [213, 94, 0, 200],
            [204, 121, 167, 200]
        ]
        
        while len(colors) < len(specialties):
            colors.extend(colors)
        colors = colors[:len(specialties)]
        
        color_map = dict(zip(specialties, colors))
        self.df['color'] = self.df['specialty'].map(color_map)
        self.df = self._cluster_points(self.df)
        
        layers = [
            pdk.Layer(
                'ScatterplotLayer',
                data=self.df,
                get_position=['longitude', 'latitude'],
                get_color='color',
                get_radius=150,
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                radius_scale=6,
                radius_min_pixels=3,
                radius_max_pixels=20,
                line_width_min_pixels=1,
            ),
            pdk.Layer(
                'ColumnLayer',
                data=self.df,
                get_position=['longitude', 'latitude'],
                get_elevation=80,
                elevation_scale=80,
                radius=30,
                get_fill_color='color',
                pickable=True,
                opacity=0.6,
            ),
            pdk.Layer(
                'TextLayer',
                data=self.df,
                get_position=['longitude', 'latitude'],
                get_text='name',
                get_size=14,
                get_color=[255, 255, 255],
                get_angle=0,
                text_anchor='middle',
                text_baseline='bottom',
                pickable=True,
                get_background_color=[0, 0, 0, 180],
                background_padding=[4, 2, 4, 2],
                offset=[0, -20],
            )
        ]

        view_state = pdk.ViewState(
            latitude=float(self.df['latitude'].mean()),
            longitude=float(self.df['longitude'].mean()),
            zoom=11,
            pitch=35,
            bearing=0
        )

        self.deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style='dark',
            tooltip={
                "html": """
                    <div style="background: rgba(0, 0, 0, 0.8); padding: 10px; border-radius: 5px; color: white;">
                        <div style="font-weight: bold; margin-bottom: 5px;">{name}</div>
                        <div>Specialty: {specialty}</div>
                        <div>Address: {address}</div>
                        <div>Phone: {phone}</div>
                        <div style="margin-top: 5px; border-top: 1px solid #666;">
                            <div style="font-weight: bold; margin-top: 5px;">Group Affiliations:</div>
                            {group_affiliations}
                        </div>
                    </div>
                """
            }
        )
        
        self.legend_data = [{"specialty": s, "color": c} for s, c in color_map.items()]

    def save_visualization(self, output_path: str = "physician_map.html") -> None:
        if not self.deck:
            raise ValueError("No visualization created")
        
        output_path = Path(output_path)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Physician Map</title>
            <style>
                body {{
                    margin: 0;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
                    background-color: #1a1a1a;
                    color: white;
                }}
                #container {{
                    position: relative;
                    height: 100vh;
                }}
                .panel {{
                    background: rgba(0, 0, 0, 0.8);
                    padding: 15px;
                    border-radius: 8px;
                    z-index: 1000;
                    color: white;
                }}
                #controls {{
                    position: absolute;
                    top: 20px;
                    left: 20px;
                    width: 250px;
                }}
                #legend {{
                    position: absolute;
                    bottom: 20px;
                    right: 20px;
                    max-height: 300px;
                    overflow-y: auto;
                }}
                .control-item {{
                    margin: 10px 0;
                }}
                .legend-item {{
                    display: flex;
                    align-items: center;
                    margin: 5px 0;
                }}
                .color-box {{
                    width: 20px;
                    height: 20px;
                    margin-right: 10px;
                    border-radius: 3px;
                }}
            </style>
        </head>
        <body>
            <div id="container">
                <div id="controls" class="panel">
                    <h3>Layer Controls</h3>
                    <div class="control-item">
                        <label>
                            <input type="checkbox" id="scatter" checked> Show Points
                        </label>
                    </div>
                    <div class="control-item">
                        <label>
                            <input type="checkbox" id="column" checked> Show 3D Columns
                        </label>
                    </div>
                    <div class="control-item">
                        <label>
                            <input type="checkbox" id="text" checked> Show Labels
                        </label>
                    </div>
                    <div class="control-item">
                        <label for="pitch">Map Pitch: <span id="pitch-value">35°</span></label>
                        <input type="range" id="pitch" min="0" max="60" value="35">
                    </div>
                </div>
                <div id="legend" class="panel">
                    <h3>Specialties</h3>
                    {''.join(f"""
                    <div class="legend-item">
                        <div class="color-box" style="background-color: rgba({item['color'][0]}, {item['color'][1]}, {item['color'][2]}, 0.8);"></div>
                        <span>{item['specialty']}</span>
                    </div>
                    """ for item in self.legend_data)}
                </div>
                {self.deck.to_html(as_string=True)}
            </div>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    setTimeout(() => {{
                        const deck = document.querySelector('iframe').contentWindow.deck;
                        
                        ['scatter', 'column', 'text'].forEach(control => {{
                            document.getElementById(control).addEventListener('change', function(e) {{
                                const layers = deck.props.layers.map(layer => 
                                    layer.id === control ? {{ ...layer, visible: e.target.checked }} : layer
                                );
                                deck.setProps({{ layers }});
                            }});
                        }});
                        
                        const pitchSlider = document.getElementById('pitch');
                        const pitchValue = document.getElementById('pitch-value');
                        
                        pitchSlider.addEventListener('input', function(e) {{
                            const value = e.target.value;
                            pitchValue.textContent = value + '°';
                            deck.setProps({{
                                viewState: {{
                                    ...deck.viewState,
                                    pitch: parseInt(value)
                                }}
                            }});
                        }});
                    }}, 1000);
                }});
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze and visualize physician groups and locations')
    parser.add_argument('data_file', help='Path to the JSON data file containing physician information')
    parser.add_argument('--hf-api-key', required=True, help='HuggingFace API key')
    parser.add_argument('--geocoding-api-key', required=True, help='OpenCage API key')
    parser.add_argument('--output', default='physician_map.html', help='Output HTML file path (default: physician_map.html)')
    
    args = parser.parse_args()

    try:
        logger.info("Starting physician analysis and mapping process...")
        analyzer = PhysicianAnalyzer(args.hf_api_key, args.geocoding_api_key)
        
        logger.info("Processing data and analyzing groups...")
        analyzer.process_data(args.data_file)
        
        logger.info("Generating visualization...")
        analyzer.generate_visualization()
        
        logger.info(f"Saving visualization to {args.output}...")
        analyzer.save_visualization(args.output)
        
        logger.info("Process completed successfully!")
        logger.info(f"Open {args.output} in a web browser to view the visualization.")
        
    except Exception as e:
        logger.error(f"Error in analysis and visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
