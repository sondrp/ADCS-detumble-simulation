import numpy as np
from datetime import datetime
from orbital_elements import OrbitalElements
from igrf import cartesian_to_lat_long

# Make orbit and prepare values 

R = 6_378_000
alt = 400_000

a = R + alt

orbital_elements = OrbitalElements(R + alt, 0, 0, 56 * np.pi / 180, 0, 0)

h = 100
t_final = 5400
steps = int(t_final / 100)

tout = np.linspace(0, t_final, steps)
mag = np.zeros((steps, 3))

for i, t in enumerate(tout):
  x, _ = orbital_elements.evaluate(t)
  lat, long, alt_km = cartesian_to_lat_long(x)

import json

def parse_igrf(json_file):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract values from JSON
    result = data["geomagnetic-field-model-result"]
    date = result["date"]["value"]
    coordinates = result["coordinates"]
    field_values = result["field-value"]
    
    lat = coordinates["latitude"]["value"]
    lon = coordinates["longitude"]["value"]
    alt = coordinates["geocentric-radius"]["value"] - 6371  # Convert radius to altitude (Earth radius ~6371 km)
    
    X = field_values["north-intensity"]["value"]
    Y = field_values["east-intensity"]["value"]
    Z = field_values["vertical-intensity"]["value"]
    F = field_values["total-intensity"]["value"]
    D = field_values["declination"]["value"]
    I = field_values["inclination"]["value"]

    print(date)
    
    print(lat, lon, alt)

    print(X, Y, Z, F, D, I)

import os

path = "../../../Downloads/IGRF.json"


parse_igrf(path)
