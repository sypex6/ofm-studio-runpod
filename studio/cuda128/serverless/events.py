"""Observe ComfyUI events for one client; never infer completion from a file."""
import json
import threading
from urllib.parse import urlencode

class ComfyEvents:
    def __init__(self, base, client_id):
        self.socket=None
        self.stop=threading.Event()
        self.thread=None
        self.prompts={}
        self.lock=threading.Lock()
        try:
            import websocket
            self.socket=websocket.create_connection(base.replace('http://','ws://')+'/ws?'+urlencode({'clientId':client_id}),timeout=2)
            self.thread=threading.Thread(target=self.read,daemon=True)
            self.thread.start()
        except Exception:
            print('ComfyUI events unavailable; using durable history',flush=True)

    def accept(self, message):
        if not isinstance(message,str):return
        event=json.loads(message)
        data=event.get('data',{})
        prompt=data.get('prompt_id')
        if not prompt:return
        with self.lock:
            state=self.prompts.setdefault(prompt,{'outputs':{}})
            kind=event.get('type')
            if kind=='executing' and data.get('node') is not None:
                state['node']=str(data['node'])
            elif kind=='progress':
                state['progress']=(data.get('value'),data.get('max'))
            elif kind=='executed':
                state['outputs'][str(data['node'])]=data.get('output',{})
            elif kind=='execution_success':state['completed']=True
            elif kind in ('execution_error','execution_interrupted'):
                state['error']=str(data.get('exception_type') or kind)+' at node '+str(data.get('node_id'))

    def read(self):
        import websocket
        while not self.stop.is_set():
            try:self.accept(self.socket.recv())
            except websocket.WebSocketTimeoutException:continue
            except Exception:return

    def state(self, prompt):
        with self.lock:
            state=self.prompts.get(prompt,{})
            return {**state,'outputs':dict(state.get('outputs',{}))}

    def close(self):
        self.stop.set()
        if self.socket is not None:
            try:self.socket.close()
            except Exception:pass
        if self.thread is not None:self.thread.join(timeout=3)
