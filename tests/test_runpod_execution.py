"""Comfy completion/output contract: partial files never mean successful video."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch
from studio.cuda128.serverless import handler as worker
from studio.cuda128.serverless.events import ComfyEvents

class ExecutionTests(unittest.TestCase):
    def test_waits_for_history_completion_and_accepts_output_event_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'output/studio/job/result.mp4'
            path.parent.mkdir(parents=True);path.write_bytes(b'complete video')
            video={'filename':'result.mp4','subfolder':'studio/job','type':'output'}
            responses=[]
            for value in ({},{'prompt':{'status':{'completed':False},'outputs':{'751':{'gifs':[video]}}}},
                          {'prompt':{'status':{'completed':True},'outputs':{}}}):
                response=Mock();response.json.return_value=value;responses.append(response)
            accepted=Mock(status_code=200);accepted.json.return_value={'prompt_id':'prompt'}
            observer=Mock();observer.state.side_effect=[{'node':'273','outputs':{}},
                {'node':'751','outputs':{'751':{'gifs':[video]}}}]
            with patch.object(worker,'ROOT',root),patch.object(worker,'ComfyEvents',return_value=observer),\
                 patch.object(worker,'validate_graph'),patch.object(worker.requests,'get',side_effect=responses),\
                 patch.object(worker.requests,'post',return_value=accepted),patch.object(worker.time,'sleep') as sleep:
                result=worker.execute({'751':{'class_type':'VHS_VideoCombine'}},'751',{},Mock())
            self.assertEqual(result,path.resolve());sleep.assert_called_once_with(2)
            observer.close.assert_called_once()

    def test_terminal_encoder_event_delivers_even_when_history_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'output/studio/job/result.mp4'
            path.parent.mkdir(parents=True);path.write_bytes(b'complete video')
            response=Mock();response.json.return_value={}
            accepted=Mock(status_code=200);accepted.json.return_value={'prompt_id':'prompt'}
            observer=Mock();observer.state.return_value={'node':'751','outputs':{'751':{'gifs':[
                {'filename':'result.mp4','subfolder':'studio/job','type':'output'}]}}}
            with patch.object(worker,'ROOT',root),patch.object(worker,'ComfyEvents',return_value=observer),\
                 patch.object(worker,'validate_graph'),patch.object(worker.requests,'get',return_value=response),\
                 patch.object(worker.requests,'post',return_value=accepted),patch.object(worker.time,'sleep') as sleep:
                self.assertEqual(worker.execute({'751':{'class_type':'VHS_VideoCombine'}},'751',{},Mock()),path.resolve())
            sleep.assert_not_called()

    def test_event_state_separates_prompts_and_only_explicit_success_completes(self):
        events=ComfyEvents.__new__(ComfyEvents)
        events.prompts={};events.lock=threading.Lock()
        def event(kind,prompt,**data):events.accept(json.dumps({'type':kind,'data':{'prompt_id':prompt,**data}}))
        event('executed','other',node='751',output={'gifs':['other user']})
        event('executing','mine',node='273')
        event('executed','mine',node='751',output={'gifs':['mine']})
        self.assertFalse(events.state('mine').get('completed',False))
        self.assertEqual(events.state('mine')['outputs']['751']['gifs'],['mine'])
        event('execution_success','mine')
        self.assertTrue(events.state('mine')['completed'])
        self.assertFalse(events.state('other').get('completed',False))

    def test_event_failure_interrupts_without_accepting_a_partial_video(self):
        registry=Mock();registry.json.return_value={}
        history=Mock();history.json.return_value={}
        accepted=Mock(status_code=200);accepted.json.return_value={'prompt_id':'prompt'}
        observer=Mock();observer.state.return_value={'error':'Interrupted at node 273'}
        with patch.object(worker,'ComfyEvents',return_value=observer),patch.object(worker,'validate_graph'),\
             patch.object(worker.requests,'get',side_effect=[registry,history]),\
             patch.object(worker.requests,'post',return_value=accepted) as post:
            with self.assertRaisesRegex(RuntimeError,'Interrupted'):
                worker.execute({},'751',{},Mock())
        self.assertTrue(any(call.args[0].endswith('/interrupt') for call in post.call_args_list))
        observer.close.assert_called_once()

if __name__=='__main__':unittest.main()
