"""Light and TV control functions."""
from utils import retry_on_error


class LightController:
    """Controller for managing lights and TV state."""

    def __init__(self, api_client, light_entity_id, media_player_entity_id):
        """
        Initialize the LightController.

        Args:
            api_client: CustomAPIClient instance
            light_entity_id: Entity ID of the light
            media_player_entity_id: Entity ID of the media player/TV
        """
        self.api_client = api_client
        self.light_entity_id = light_entity_id
        self.media_player_entity_id = media_player_entity_id

    @retry_on_error(max_attempts=3)
    def is_tv_on(self):
        """
        Check if the TV is on.

        Returns:
            bool: True if TV is on, False otherwise
        """
        import time

        tv = self.api_client.get_entity(entity_id=self.media_player_entity_id)
        if not tv:
            return False

        print(tv["state"] + " " + str(time.time()))
        return tv["state"] == "on"

    @retry_on_error(max_attempts=3)
    def turn_on_light(self, brightness_pct=100, rgb_color=None):
        """
        Turn on the light with specified color and brightness.

        Args:
            brightness_pct: Brightness percentage (0-100)
            rgb_color: RGB color as a list [R, G, B], defaults to red
        """
        if rgb_color is None:
            rgb_color = [255, 0, 0]
        self.api_client.turn_on(
            entity_id=self.light_entity_id,
            brightness_pct=brightness_pct,
            rgb_color=rgb_color
        )

    @retry_on_error(max_attempts=3)
    def turn_off_light(self):
        """Turn off the light."""
        self.api_client.turn_off(entity_id=self.light_entity_id)

    @retry_on_error(max_attempts=3)
    def set_light_color(self, target_color, brightness_pct, rgbww_values=None):
        """
        Set the light to a specific color and brightness.

        Args:
            target_color: RGB color as a list [R, G, B]
            brightness_pct: Brightness percentage (0-100)
            rgbww_values: Optional RGBWW values [WW1, WW2]
        """
        if rgbww_values is None:
            rgbww_values = [255, 255]

        rgb_color = target_color
        print(f"Setting light color to: {rgb_color} with brightness: {brightness_pct}%")
        self.api_client.turn_on(
            entity_id=self.light_entity_id,
            brightness_pct=brightness_pct,
            rgb_color=rgb_color
        )
