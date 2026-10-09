"""Job inputs must be discoverable by ComfyUI's flat file selectors and cleaned."""
import json
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
import unittest
from unittest.mock import Mock, patch

from studio.cuda128.serverless import handler as worker


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'input').mkdir()
        self.unrelated = self.root / 'input' / 'other_job.png'
        self.unrelated.write_bytes(b'other job')
        self.params = dict(width=480, height=832, fps=30, steps=4, seed=0, prompt='movement', duration=1)
        self.job = {'input': {'image_url': 'https://example.org/image',
                              'video_url': 'https://example.org/video'}}

    def test_both_workflows_use_discoverable_files_and_clean_only_their_job(self):
        for profile in ('animate-ki', 'animate-wrapper'):
            with self.subTest(profile=profile):
                def fetch(url, target, limit):
                    target.write_bytes(b'input bytes')
                def execute(graph, output, job, progress):
                    # The pinned ComfyUI LoadImage schema does os.listdir(input)
                    # and isfile(), without recursively looking in subfolders.
                    available = [p.name for p in (self.root / 'input').iterdir() if p.is_file()]
                    selected = {k: v for k, v in graph.items()
                                if v['class_type'] in ('LoadImage', 'VHS_LoadVideo')}
                    info = {
                        'LoadImage': {'input': {'required': {'image': [[n for n in available if n.endswith('.png')], {}]}}},
                        'VHS_LoadVideo': {'input': {'required': {'video': [[n for n in available if n.endswith('.mp4')], {}]}}},
                    }
                    worker.validate_graph(selected, info)
                    for node in selected.values():
                        for value in node['inputs'].values():
                            self.assertNotIn('/', value)
                            self.assertTrue((self.root / 'input' / value).is_file())
                    target = self.root / 'output' / graph[output]['inputs']['filename_prefix']
                    target = target.with_suffix('.mp4')
                    target.parent.mkdir(parents=True)
                    target.write_bytes(b'video result')
                    return target
                probe = CompletedProcess([], 0, stdout=json.dumps({
                    'streams': [{'width': 480, 'height': 832}], 'format': {'duration': '1'}}))
                delivery = Mock()
                delivery.json.return_value = {'url': 'https://example.org/result.mp4'}
                with patch.object(worker, 'ROOT', self.root), \
                        patch.object(worker, 'checked_input', return_value=(profile, self.params, 'https://example.org/upload')), \
                        patch.object(worker, 'fetch', side_effect=fetch), \
                        patch.object(worker.subprocess, 'run', return_value=probe), \
                        patch.object(worker, 'execute', side_effect=execute), \
                        patch.object(worker, 'public_url', side_effect=lambda value: value), \
                        patch.object(worker.requests, 'put', return_value=delivery), \
                        patch.object(worker.requests, 'post'):
                    result = worker.handler(self.job, progress=lambda *args: None)
                self.assertEqual(result['workflow'], profile)
                self.assertEqual(list((self.root / 'input').iterdir()), [self.unrelated])
                self.assertEqual(self.unrelated.read_bytes(), b'other job')
                self.assertFalse(list((self.root / 'output' / 'studio').iterdir()))

    def test_download_failure_cleans_partial_inputs(self):
        def fetch(url, target, limit):
            target.write_bytes(b'partial')
            raise RuntimeError('download interrupted')
        with patch.object(worker, 'ROOT', self.root), \
                patch.object(worker, 'checked_input', return_value=('animate-ki', self.params, 'https://example.org/upload')), \
                patch.object(worker, 'fetch', side_effect=fetch), \
                patch.object(worker.requests, 'post'):
            with self.assertRaisesRegex(RuntimeError, 'download interrupted'):
                worker.handler(self.job, progress=lambda *args: None)
        self.assertEqual(list((self.root / 'input').iterdir()), [self.unrelated])


if __name__ == '__main__':
    unittest.main()
