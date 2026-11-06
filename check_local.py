"""Local validation tests only. Never creates a model or emulates a DPU."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import deploy

class LocalChecks(unittest.TestCase):
    def test_python38_syntax(self):
        ast.parse((deploy.ROOT/'deploy.py').read_text(),feature_version=(3,8))

    def test_fingerprint_query_formats(self):
        self.assertEqual(deploy.fingerprints('{"fingerprint": "0x101000016010407"}'),{0x101000016010407})
        self.assertEqual(deploy.fingerprints('fingerprint = 0x101000016010407'),{0x101000016010407})
        self.assertEqual(deploy.fingerprints('{"fingerprint": "0x0"}'),set())

    def test_hash_and_atomic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            deploy.write_json(path,{'status':'CHECK_ONLY'})
            self.assertEqual(json.loads(path.read_text()),{'status':'CHECK_ONLY'})
            self.assertEqual(len(deploy.sha(path)),64)
            self.assertFalse(path.with_name('report.json.tmp').exists())

    def test_numeric_layout_validation(self):
        # Numeric fixtures exercise preprocessing only, not an inference backend.
        patch=np.arange(138*11*11,dtype=np.float32).reshape(138,11,11)/(138*11*11)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'numeric-validation.npy'
            np.save(str(path),patch)
            actual=deploy.load_patch(path,(1,11,11,138),np)
            self.assertEqual(actual.shape,(1,11,11,138))
            self.assertTrue(actual.flags.c_contiguous)
            self.assertTrue(np.array_equal(actual[0,3,5,:],patch[:,3,5]))
            for invalid in (np.ones((3,11,11),dtype=np.float32),
                            np.full((138,11,11),np.nan),np.full((138,11,11),2.0)):
                np.save(str(path),invalid)
                with self.assertRaises(ValueError): deploy.load_patch(path,(1,138,11,11),np)

    def test_reference_identity_and_tolerance(self):
        reference={'model_sha256':'same-model','input_sha256':'same-input',
                   'prediction':'water','logit':-1.0,'tolerance':0.125}
        deploy.baseline_check(dict(reference),reference)
        for changes in ({'model_sha256':'different'}, {'input_sha256':'different'},
                        {'prediction':'oil'},{'logit':-0.5}):
            result=dict(reference);result.update(changes)
            with self.assertRaises(RuntimeError): deploy.baseline_check(result,reference)

if __name__=='__main__':
    unittest.main(verbosity=2)
