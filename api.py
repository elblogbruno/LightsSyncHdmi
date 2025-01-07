from requests import get, post

class CustomAPIClient:
    def __init__(self, host, token):
        self.url = host
        self.headers = {
            "Authorization": "Bearer " + token,
            "content-type": "application/json",
            "cache-control": "no-cache"
        }

    def get_entity(self, entity_id):
        url =  self.url + "/states/" + entity_id

        response = get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching entity: {response.status_code} - {response.text}")
            return None

    def turn_on(self, entity_id, brightness_pct=100, rgbww_color=None):
        url = self.url + "/services/light/turn_on"
        data = {
            "entity_id": entity_id,
            "brightness_pct": brightness_pct 
        }

        if rgbww_color:
            data["rgbww_color"] = rgbww_color

        try:
            response = post(url, headers=self.headers, json=data)
            if response.status_code == 200:
                print(f"Light turned on: {response.text}")
            else:
                print(f"Error turning on light: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error turning on light: {e} - {url}")

    def turn_off(self, entity_id, brightness_pct=100, rgbww_color=None):
        url = self.url + "/services/light/turn_off"
        data = {
            "entity_id": entity_id, 
        } 

        try:
            response = post(url, headers=self.headers, json=data)
            if response.status_code == 200:
                print(f"Light turned off: {response.text}")
            else:
                print(f"Error turning off light: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error turning off light: {e} - {url}")