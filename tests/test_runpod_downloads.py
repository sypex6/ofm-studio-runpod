"""Exercise resumable downloads against HTTP, including corrupt and ignored ranges."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from studio.cuda128.serverless.prepare_models import download

PAYLOAD = b'verified model bytes' * 100


class HTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.calls.append(dict(self.headers))
        mode = self.server.mode
        if mode == 'forbidden':
            self.send_response(403)
            self.end_headers()
            return
        start = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
        if mode == 'ignore':
            start = 0
        payload = (b'X' * len(PAYLOAD) if mode == 'corrupt' else PAYLOAD)[start:]
        self.send_response(206 if start else 200)
        if start:
            self.send_header('Content-Range', f'bytes {start + (1 if mode == "bad-range" else 0)}-{len(PAYLOAD)-1}/{len(PAYLOAD)}')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), HTTPHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / 'model.bin'
        self.model = dict(path='model.bin', url=f'http://127.0.0.1:{self.server.server_port}/model',
                          size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest())
        self.server.mode = 'normal'
        self.server.calls = []

    def partial(self, fingerprint=None):
        self.target.with_suffix('.bin.part').write_bytes(PAYLOAD[:100])
        self.target.with_suffix('.bin.part.json').write_text(json.dumps(fingerprint or self.model['sha256']))

    def test_direct_download_without_cache_or_token_leak(self):
        download(self.model, self.target, token='private-test-value', retry_delay=0)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertEqual([p.name for p in self.target.parent.iterdir()], ['model.bin'])
        self.assertNotIn('Authorization', self.server.calls[0])

    def test_resume_existing_partial_and_atomic_promotion(self):
        self.partial()
        download(self.model, self.target, retry_delay=0)
        self.assertEqual(self.server.calls[0]['Range'], 'bytes=100-')
        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_range_ignored_restarts_instead_of_appending(self):
        self.partial()
        self.server.mode = 'ignore'
        download(self.model, self.target, retry_delay=0)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_different_model_does_not_resume_old_partial(self):
        self.partial('another-checksum')
        download(self.model, self.target, retry_delay=0)
        self.assertNotIn('Range', self.server.calls[0])

    def test_corrupt_bytes_never_become_a_model(self):
        self.server.mode = 'corrupt'
        with self.assertRaises(RuntimeError):
            download(self.model, self.target, attempts=2, retry_delay=0)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_suffix('.bin.part').exists())

    def test_invalid_range_does_not_overwrite_partial(self):
        self.partial()
        self.server.mode = 'bad-range'
        with self.assertRaises(RuntimeError):
            download(self.model, self.target, attempts=1, retry_delay=0)
        self.assertEqual(self.target.with_suffix('.bin.part').read_bytes(), PAYLOAD[:100])
        self.assertFalse(self.target.exists())

    def test_auth_failure_is_not_retried(self):
        self.server.mode = 'forbidden'
        with self.assertRaisesRegex(RuntimeError, 'HTTP 403'):
            download(self.model, self.target, retry_delay=0)
        self.assertEqual(len(self.server.calls), 1)

    def test_insufficient_space_fails_before_network(self):
        with patch('studio.cuda128.serverless.prepare_models.shutil.disk_usage') as usage:
            usage.return_value.free = 0
            with self.assertRaisesRegex(RuntimeError, 'Insufficient Volume space'):
                download(self.model, self.target, retry_delay=0)
        self.assertFalse(self.server.calls)

    def test_global_download_uses_no_disk_guard_rename_or_fsync(self):
        marker = self.target.with_suffix('.bin.complete.json')
        marker.write_text('old readiness')
        with patch('studio.cuda128.serverless.prepare_models.shutil.disk_usage',
                   side_effect=AssertionError('object storage is not local disk')), \
                patch.object(Path, 'replace', side_effect=AssertionError('atomic rename')), \
                patch('os.fsync', side_effect=AssertionError('POSIX fsync')):
            download(self.model, self.target, retry_delay=0, object_storage=True)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertFalse(marker.exists())
        self.assertEqual([p.name for p in self.target.parent.iterdir()], ['model.bin'])

    def test_global_partial_resumes_directly(self):
        self.target.write_bytes(PAYLOAD[:100])
        self.target.with_suffix('.bin.download.json').write_text(json.dumps(self.model['sha256']))
        download(self.model, self.target, retry_delay=0, object_storage=True)
        self.assertEqual(self.server.calls[0]['Range'], 'bytes=100-')
        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_global_corrupt_download_never_gets_ready_marker(self):
        self.server.mode = 'corrupt'
        with self.assertRaises(RuntimeError):
            download(self.model, self.target, attempts=1, retry_delay=0, object_storage=True)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_suffix('.bin.complete.json').exists())


if __name__ == '__main__':
    unittest.main()
