"""Offline graph checks usable in GitHub Actions before Docker build."""
import json
from pathlib import Path
import unittest

from studio.cuda128.serverless.graph import native, wrapper, validate_graph

PARAMS = dict(width=480, height=832, fps=30, steps=4, seed=12345, prompt='A person walking in soft light')


class GraphTests(unittest.TestCase):
    def test_native_long_video_links_continuation_and_trims_overlap(self):
        graph, output = native('person.png', 'motion.mp4', PARAMS, 449, 'studio/test/result')
        animate = [(key, n) for key, n in graph.items() if n['class_type'] == 'WanAnimateToVideo']
        self.assertEqual(len(animate), 7)
        self.assertNotIn('continue_motion', animate[0][1]['inputs'])
        for (before, _), (_, current) in zip(animate, animate[1:]):
            self.assertEqual(current['inputs']['video_frame_offset'], [before, 5])
            self.assertIn('continue_motion', current['inputs'])
        self.assertEqual(graph[output]['class_type'], 'VHS_VideoCombine')
        trim = [n['inputs'] for n in graph.values() if n['class_type'] == 'GetImageRangeFromBatch']
        self.assertEqual(sum(n['num_frames'] for n in trim), 449)
        self.assertNotIn('easy whileLoopStart', [n['class_type'] for n in graph.values()])

    def test_native_short_video_no_extra_blocks(self):
        graph, output = native('x.png', 'y.mp4', PARAMS, 49, 'studio/short/result')
        self.assertEqual(sum(n['class_type'] == 'WanAnimateToVideo' for n in graph.values()), 1)
        self.assertEqual(next(n['inputs']['num_frames'] for n in graph.values() if n['class_type']=='GetImageRangeFromBatch'),49)
        self.assertEqual(graph[output]['inputs']['frame_rate'], 30)

    def test_wrapper_preserves_pose_color_uni3c_and_exact_lora_strengths(self):
        graph, output = wrapper('identity.png', 'ref.mp4', PARAMS, 149, 'studio/wrapped/result')
        classes = {n['class_type'] for n in graph.values()}
        self.assertTrue({'TSPoseDataSmoother', 'TSColorMatch', 'WanVideoUni3C_embeds'} <= classes)
        self.assertFalse({'GetNode', 'SetNode', 'SimpleMath+', 'PreviewAny'} & classes)
        self.assertEqual(graph['493']['inputs']['text'], PARAMS['prompt'])
        self.assertEqual(graph['273']['inputs']['seed'], PARAMS['seed'])
        self.assertIs(graph['273']['inputs']['force_offload'], True)
        self.assertEqual(graph['273']['inputs']['scheduler'], 'dpm++_sde')
        self.assertEqual(graph['270']['inputs']['num_frames'],149)
        self.assertEqual(graph['270']['inputs']['width'],480)
        self.assertEqual(graph['270']['inputs']['height'],832)
        self.assertAlmostEqual(graph['354']['inputs']['strength_4'], 0.6)
        self.assertEqual(graph['76']['inputs']['image'],'identity.png')
        self.assertEqual(graph['75']['inputs']['frame_load_cap'],149)
        self.assertEqual(graph[output]['inputs']['filename_prefix'],'studio/wrapped/result')
        self.assertEqual(graph[output]['inputs']['audio'],['75',2])
        for node in graph.values():
            for value in node['inputs'].values():
                if isinstance(value,list) and len(value)==2:
                    self.assertIn(value[0],graph)

    def test_registry_validation_fails_missing_node_or_model(self):
        graph={'1':{'class_type':'Loader','inputs':{'model':'missing'}}}
        with self.assertRaisesRegex(ValueError,'Missing ComfyUI node'):
            validate_graph(graph,{})
        info={'Loader':{'input':{'required':{'model':[['present'],{}]}}}}
        with self.assertRaisesRegex(ValueError,'Unavailable model'):
            validate_graph(graph,info)

    def test_manifest_covers_every_reachable_model_and_sidecar(self):
        entries=json.loads(Path('studio/cuda128/serverless/models.json').read_text())
        names={Path(m['path']).name for m in entries}
        for builder in (native,wrapper):
            graph,_=builder('x.png','y.mp4',PARAMS,149,'studio/test/result')
            for node in graph.values():
                for key,value in node['inputs'].items():
                    if key in ('model','model_name','unet_name','vae_name','clip_name','vitpose_model','yolo_model','lora_name') or key.startswith('lora_'):
                        if isinstance(value,str) and value.endswith(('.safetensors','.onnx')):
                            self.assertIn(value,names)
        self.assertIn('vitpose_h_wholebody_data.bin',names)
        self.assertTrue(all(m['revision']!='main' for m in entries))


if __name__ == '__main__':
    unittest.main()
