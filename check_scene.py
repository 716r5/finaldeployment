"""Algorithm/file tests. No inference backend is emulated or substituted."""
import ast
import tempfile
import json
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import scene
from queue_app import read_queue
from regions import extract as extract_regions
from hsi_contract import USABLE_BANDS,normalise_band

class SceneChecks(unittest.TestCase):
    def test_python38_syntax(self):
        for name in ('deploy.py','scene.py','prepare_scene.py','hsi_contract.py'):
            ast.parse((scene.ROOT/name).read_text(),feature_version=(3,8))

    def test_dense_centers_and_border(self):
        rows,cols=scene.centers(600,400)
        self.assertEqual((len(rows),len(cols)),(590,390))
        self.assertEqual((rows[0],rows[-1],cols[0],cols[-1]),(5,594,5,394))
        with self.assertRaises(ValueError): scene.centers(600,400,0)

    def test_input_contract_band_order(self):
        self.assertEqual(len(USABLE_BANDS),138)
        self.assertEqual(len(set(USABLE_BANDS)),138)
        self.assertEqual((USABLE_BANDS[0],USABLE_BANDS[-1]),(3,198))

    def test_normalization_constant_and_nonfinite(self):
        output,_=normalise_band(np.ones((11,11),dtype=np.float32),np)
        self.assertTrue(np.array_equal(output,np.zeros((11,11))))
        with self.assertRaises(ValueError): normalise_band(np.full((11,11),np.nan),np)

    def test_quantization_layout_and_center(self):
        # Numeric arithmetic fixtures, not a fake scene/model prediction.
        cube=np.arange(13*15*138,dtype=np.float32).reshape(13,15,138)
        cube/=cube.size
        patch=cube[1:12,2:13]
        for shape in ((1,138,11,11),(1,11,11,138)):
            output=np.empty(shape,dtype=np.int8)
            temporary=np.empty(shape,dtype=np.float32)
            scene.quantize_into(cube,6,7,shape,128,temporary,output)
            expected=np.clip(np.rint(patch*128),-128,127).astype(np.int8)
            if shape[1]==138: expected=np.transpose(expected,(2,0,1))
            self.assertTrue(np.array_equal(output[0],expected))
            self.assertTrue(output.flags.c_contiguous)

    def test_once_per_scene_input_conversion_matches_patch_conversion(self):
        cube=np.linspace(0,1,13*15*138,dtype=np.float32).reshape(13,15,138)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            quantized,reused=scene.input_cache(cube,output,128.0,{'cube_sha256':'numeric-fixture-only'})
            self.assertFalse(reused)
            for shape in ((1,138,11,11),(1,11,11,138)):
                converted=np.empty(shape,dtype=np.int8)
                expected=np.empty(shape,dtype=np.int8)
                scene.copy_patch(quantized,6,7,shape,converted)
                scene.quantize_into(cube,6,7,shape,128.0,np.empty(shape,dtype=np.float32),expected)
                self.assertTrue(np.array_equal(converted,expected))
            del quantized
            quantized,reused=scene.input_cache(cube,output,128.0,{'cube_sha256':'numeric-fixture-only'})
            self.assertTrue(reused)
            del quantized

    def test_metrics_ignore_unclassified(self):
        classes=np.array([[0,1,255],[1,0,255]],dtype=np.uint8)
        labels=np.array([[0,1,1],[0,1,0]],dtype=np.uint8)
        value=scene.metrics(classes,labels)
        self.assertEqual(value['evaluated_centers'],4)
        self.assertEqual(value['accuracy'],0.5)
        self.assertEqual(value['oil_f1'],0.5)

    def test_empty_support_has_no_fabricated_score(self):
        value=scene.metrics(np.full((2,2),255,dtype=np.uint8),np.zeros((2,2),dtype=np.uint8))
        self.assertIsNone(value['accuracy']);self.assertIsNone(value['oil_recall'])

    def test_png_writer_preserves_actual_bytes(self):
        rgb=np.arange(11*13*3,dtype=np.uint16).reshape(11,13,3).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bytes-only.png'
            scene.write_png(path,rgb)
            self.assertTrue(np.array_equal(np.asarray(Image.open(path)),rgb))

    def test_regions_diagonal_bridge_and_coordinates(self):
        values=np.array([[1,0,1,0,0],[0,1,0,0,0],[0,0,0,0,1]],dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            result=extract_regions(values,Path(directory)/'region-check.csv')
            self.assertEqual(result['oil_region_count'],2)
            largest=result['largest_oil_regions'][0]
            self.assertEqual(largest['pixel_count'],3)
            self.assertEqual((largest['row_min'],largest['col_min'],largest['row_max'],largest['col_max']),(0,0,1,2))
            self.assertAlmostEqual(largest['centroid_row'],1/3)
            self.assertEqual(largest['centroid_col'],1.0)

    def test_region_empty_and_solid(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'region-check.csv'
            self.assertEqual(extract_regions(np.zeros((5,7),dtype=np.uint8),path)['oil_region_count'],0)
            result=extract_regions(np.ones((5,7),dtype=np.uint8),path)
            self.assertEqual(result['oil_region_count'],1)
            self.assertEqual(result['largest_oil_regions'][0]['pixel_count'],35)

    def test_queue_enumeration_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'queue.json'
            scene.write_json(path,{'contract':scene.CONTRACT,'scenes':[
                {'directory':'first','source_name':'first-real-file.mat'},
                {'directory':'second','source_name':'second-real-file.mat'}]})
            _,entries=read_queue(path,check_inputs=False)
            self.assertEqual(len(entries),2)
            scene.write_json(path,{'contract':scene.CONTRACT,'scenes':[{'directory':'../outside','source_name':'invalid-path'}]})
            with self.assertRaises(ValueError):read_queue(path,check_inputs=False)

if __name__=='__main__':
    unittest.main(verbosity=2)
