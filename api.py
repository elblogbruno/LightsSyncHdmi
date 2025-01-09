from requests import get, post
from asyncio import Queue
import uuid

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

    def turn_on(self, entity_id, brightness_pct=100, rgb_color=None):
        url = self.url + "/services/light/turn_on"
        data = {
            "entity_id": entity_id,
            "brightness_pct": brightness_pct 
        }

        if rgb_color:
            data["rgb_color"] = rgb_color

        try:
            response = post(url, headers=self.headers, json=data)
            if response.status_code == 200:
                print(f"Light turned on: {response.text}")
            else:
                print(f"Error turning on light: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error turning on light: {e} - {url}")

    def turn_off(self, entity_id):
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

import time
import websockets
import asyncio
import threading
import json

class CustomWebsocketClient: 
    def __init__(self, host_websocket, token, entities):
        self.host = host_websocket
        self.token = token 
        self.entities = entities
        self.entities_status = {}
        self.websocket = None
        self.outgoing_queue = Queue()
        self.incoming_queue = Queue()
        self.pending_responses = {}
        self._running = False
        self._tasks = []

    async def init_socket(self):
        self._running = True
        try:
            async with websockets.connect(self.host) as websocket:
                self.websocket = websocket

                # Autenticación
                auth_msg = {'type': 'auth', 'access_token': self.token}
                await self.websocket.send(json.dumps(auth_msg))
                
                while True:
                    auth_response = await self.websocket.recv()
                    data = json.loads(auth_response)
                    if data.get('type') == 'auth_ok':
                        print("Authentication successful")
                        break
                    elif data.get('type') == 'auth_invalid':
                        print("Authentication failed")
                        return

                # Suscripción a eventos
                await self.websocket.send(json.dumps({
                    'id': 1, 
                    'type': 'subscribe_events',
                    'event_type': 'state_changed'
                }))

                print("Subscribed to events")

                # Crear y guardar referencias a las tareas
                send_task = asyncio.create_task(self._send_loop())
                receive_task = asyncio.create_task(self._receive_loop())
                self._tasks = [send_task, receive_task]

                # Esperar a que cualquiera de las tareas termine
                done, pending = await asyncio.wait(
                    self._tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancelar las tareas pendientes
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            print(f"WebSocket connection error: {e}")
        finally:
            self._running = False
            self.websocket = None
            # Limpiar las colas
            while not self.outgoing_queue.empty():
                await self.outgoing_queue.get()
            while not self.incoming_queue.empty():
                await self.incoming_queue.get()

    async def _send_loop(self):
        try:
            while self._running:
                try:
                    message = await asyncio.wait_for(
                        self.outgoing_queue.get(), 
                        timeout=1.0
                    )
                    if message is None:
                        break
                    await self.websocket.send(json.dumps(message))
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            print(f"Send loop error: {e}")

    async def _receive_loop(self):
        try:
            while self._running:
                try:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    
                    if 'type' in data:
                        if data['type'] == 'event':
                            await self._handle_state_event(data)
                        elif data['type'] == 'result':
                            if 'id' in data and data['id'] in self.pending_responses:
                                self.pending_responses[data['id']].set_result(data)
                except Exception as e:
                    print(f"Receive loop error: {e}")
                    break
        finally:
            self._running = False

    async def _handle_state_event(self, data):
        try:
            entity_data = data['event']['data']
            entity_id = entity_data['entity_id']
            if entity_id in self.entities:
                print(f"Entity: {entity_id} - State: {entity_data['new_state']['state']}")
                self.entities_status[entity_id] = entity_data['new_state']['state']
        except Exception:
            pass

    async def send_command(self, message):
        if not hasattr(self, '_msg_id_counter'):
            self._msg_id_counter = 2
        else:
            self._msg_id_counter += 1
        msg_id = self._msg_id_counter
        message['id'] = msg_id
        
        # Crear un Future para la respuesta
        future = asyncio.get_event_loop().create_future()
        self.pending_responses[msg_id] = future
        
        # Enviar mensaje
        await self.outgoing_queue.put(message)
        
        # Esperar respuesta con timeout
        try:
            response = await asyncio.wait_for(future, timeout=5.0)
            return response
        except asyncio.TimeoutError:
            print(f"Timeout waiting for response to message {msg_id}")
        finally:
            self.pending_responses.pop(msg_id, None)

    async def turn_on(self, entity_id, brightness_pct=100, rgb_color=None):
        if entity_id not in self.entities or not self.websocket:
            return

        message = {
            "type": "call_service",
            "domain": "light",
            "service": "turn_on",
            "service_data": {"brightness_pct": brightness_pct},
            "target": {"entity_id": entity_id},
            "return_response": False
        }
        if rgb_color:
            message["service_data"]["rgb_color"] = rgb_color

        response = await self.send_command(message)
        print(f"Turn on response: {response}")
        return response

    async def turn_off(self, entity_id):
        if entity_id not in self.entities or not self.websocket:
            return

        message = { 
            "type": "call_service",
            "domain": "light",
            "service": "turn_off",
            "service_data": {},
            "target": {"entity_id": entity_id},
            "return_response": False
        }

        response = await self.send_command(message)
        print(f"Turn off response: {response}")
        return response