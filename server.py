
from flask import Flask, render_template, request, jsonify
import os
from api import CustomAPIClient

app = Flask(__name__)

# Initialize Home Assistant API client
api_client = CustomAPIClient(os.environ['HASSIO_HOST'], os.environ['HASSIO_TOKEN'])
light_entity_id = os.environ.get("LIGHT_ENTITY_ID", "light.habitacion")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/turn_on', methods=['POST'])
def turn_on():
    color = request.json.get('color', [255, 255, 255, 255, 255])
    brightness = request.json.get('brightness', 100)
    api_client.turn_on(entity_id=light_entity_id, brightness_pct=brightness, rgbww_color=color)
    return jsonify({"status": "success"})

@app.route('/turn_off', methods=['POST'])
def turn_off():
    api_client.turn_off(entity_id=light_entity_id)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)