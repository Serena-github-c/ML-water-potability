## Making requests
import requests

url = 'http://localhost:9696/predict'

water_sample = {'ph': 3.71608,
 'hardness': 129.422921,
 'solids': 18630.057858,
 'sulfate': 332.615625,
 'conductivity': 592.885359,
 'turbidity': 4.500656}

requests.post(url, json=water_sample).json()

