import re
import pandas as pd
import pydeck as pdk
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from opencage.geocoder import OpenCageGeocode
from opencage.geocoder import InvalidInputError, RateLimitExceededError, UnknownError
import time
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import DBSCAN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

class PhysicianMapper:
    def __init__(self, data_path: str, api_key: str):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.deck: Optional[pdk.Deck] = None
        self.legend_data = []
        self.geocoder = OpenCageGeocode(api_key)
        self.request_count = 0
        self.rate_limit = 1  # Minimum time between requests in seconds

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
                time.sleep(self.rate_limit)  # Respect rate limit
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
                time.sleep(60)  # Wait a minute if rate limit exceeded
            except (InvalidInputError, UnknownError) as e:
                logger.warning(f"Geocoding error: {str(e)}")
                time.sleep(2 * (attempt + 1))
        return None

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
            longitude=lon
        )

    def load_data(self) -> None:
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            physicians = data.get("physicians", [])
            if not physicians:
                raise ValueError("No physician data found")
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                processed_data = list(executor.map(self._process_physician, physicians))
            
            self.df = pd.DataFrame([vars(p) for p in processed_data])
            self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
            self.df['longitude'] = pd.to_numeric(self.df['longitude'], errors='coerce')
            self.df.dropna(subset=['latitude', 'longitude'], inplace=True)
            
            if self.df.empty:
                raise ValueError("No valid coordinates found")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

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
            'phone': 'first'
        }).reset_index()

    def generate_visualization(self) -> None:
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
    parser = argparse.ArgumentParser(description='Create an interactive map of physician locations')
    parser.add_argument('data_file', help='Path to the JSON data file containing physician information')
    parser.add_argument('--api-key', required=True, help='OpenCage API key')
    parser.add_argument('--output', default='physician_map.html', help='Output HTML file path (default: physician_map.html)')
    
    args = parser.parse_args()

    try:
        logger.info("Starting physician mapping process...")
        mapper = PhysicianMapper(args.data_file, args.api_key)
        
        logger.info("Loading and processing data...")
        mapper.load_data()
        
        logger.info("Generating visualization...")
        mapper.generate_visualization()
        
        logger.info(f"Saving visualization to {args.output}...")
        mapper.save_visualization(args.output)
        
        logger.info("Process completed successfully!")
        logger.info(f"Open {args.output} in a web browser to view the visualization.")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
