import os
import dotenv
import time
from homeassistant_api import Client

from api import CustomAPIClient

dotenv.load_dotenv()

from requests import get

import random 
api_client = CustomAPIClient(os.environ['HASSIO_HOST'], os.environ['HASSIO_TOKEN'])

# Initialize Home Assistant API client
api_client.turn_on("light.habitacion", brightness_pct=100, rgbww_color=[255, 0, 0, 255, 255])

for i in range(10):
    time.sleep(1)
    random_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    api_client.turn_on("light.habitacion", brightness_pct=100, rgbww_color=random_color)


api_client.turn_off("light.habitacion", brightness_pct=0, rgbww_color=[0, 0, 0, 0, 0])

tv = api_client.get_entity(entity_id="media_player.samsung_qn85ca_75_2")  # Replace with your TV entity ID

print(tv["state"]  + " " +  str(time.time()))