"""Global Volume readiness never writes; partial uploads cannot become ready."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from studio.cuda128.serverless.prepare_models import prepare
from studio.cuda128.serverless.publish_global import publish


class GlobalVolumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source'
        self.destination = Path(self.temp.name) / 'global'
        self.source.mkdir()
        self.payload = b'verified weights'
        self.model = dict(path='diffusion_models/model.bin', size=len(self.payload),
                          sha256=hashlib.sha256(self.payload).hexdigest(),
                          url='https://example.org/model', profiles=['animate-ki'])
        origin = self.source / self.model['path']
        origin.parent.mkdir()
        origin.write_bytes(self.payload)

    def check(self):
        original = Path.read_text
        def read(path, *args, **kwargs):
            if path.name == 'models.json':
                return json.dumps([self.model])
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read), \
                patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir on reader')), \
                patch.object(Path, 'write_text', side_effect=AssertionError('write on reader')), \
                patch.dict('sys.modules', {'filelock': SimpleNamespace(
                    FileLock=lambda *args, **kwargs: self.fail('lock on reader'))}), \
                patch('studio.cuda128.serverless.prepare_models.shutil.disk_usage',
                      side_effect=AssertionError('object storage capacity is not disk capacity')):
            prepare(self.destination, ['animate-ki'], check=True, readonly=True)

    def test_publish_and_check_without_posix_operations(self):
        with patch.object(Path, 'replace', side_effect=AssertionError('rename')), \
                patch('os.link', side_effect=AssertionError('hardlink')):
            publish(self.source, self.destination, ['animate-ki'], [self.model])
        self.check()

    def test_partial_upload_rejected_and_recovered(self):
        publish(self.source, self.destination, ['animate-ki'], [self.model])
        target = self.destination / self.model['path']
        marker = target.with_suffix('.bin.complete.json')
        with patch('studio.cuda128.serverless.publish_global.shutil.copyfile',
                   side_effect=OSError('interrupted')):
            target.write_bytes(b'partial')
            with self.assertRaises(OSError):
                publish(self.source, self.destination, ['animate-ki'], [self.model])
        self.assertFalse(marker.exists())
        with self.assertRaises(SystemExit):
            self.check()
        publish(self.source, self.destination, ['animate-ki'], [self.model])
        self.check()

    def test_wrong_source_rejected_before_destination_changes(self):
        (self.source / self.model['path']).write_bytes(b'corrupt')
        with self.assertRaises(RuntimeError):
            publish(self.source, self.destination, ['animate-ki'], [self.model])
        self.assertFalse(self.destination.exists())

    def test_missing_global_directory_does_not_get_created(self):
        with self.assertRaises(SystemExit):
            self.check()
        self.assertFalse(self.destination.exists())

    def test_cannot_download_to_readonly_volume(self):
        with self.assertRaises(ValueError):
            prepare(self.destination, ['animate-ki'], readonly=True)

    def test_source_and_destination_must_be_separate(self):
        for target in (self.source, self.source / 'models', self.source.parent):
            with self.assertRaises(ValueError):
                publish(self.source, target, ['animate-ki'], [self.model])

    def test_direct_preparation_creates_readiness_without_posix_operations(self):
        original = Path.read_text
        def read(path, *args, **kwargs):
            if path.name == 'models.json':
                return json.dumps([self.model])
            return original(path, *args, **kwargs)
        def fetch(model, target, token, object_storage=False):
            self.assertTrue(object_storage)
            self.assertFalse(target.with_suffix('.bin.complete.json').exists())
            target.write_bytes(self.payload)
        with patch.object(Path, 'read_text', read), \
                patch.object(Path, 'replace', side_effect=AssertionError('rename')), \
                patch.dict('sys.modules', {'filelock': SimpleNamespace(
                    FileLock=lambda *args, **kwargs: self.fail('lock on writer'))}), \
                patch('studio.cuda128.serverless.prepare_models.shutil.disk_usage',
                      side_effect=AssertionError('disk guard on object storage')), \
                patch('studio.cuda128.serverless.prepare_models.download', side_effect=fetch):
            prepare(self.destination, ['animate-ki'], global_write=True)
        self.check()


if __name__ == '__main__':
    unittest.main()
