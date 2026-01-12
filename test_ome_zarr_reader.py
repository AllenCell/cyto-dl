#!/usr/bin/env python
"""
Test OmeZarrReader with zarr-3 compatibility
"""
import sys
import tempfile
import os
import numpy as np
import zarr
import shutil

# Add cyto_dl to path
sys.path.insert(0, '/home/user/cyto-dl')

def create_test_ome_zarr(store_path, use_v3=True):
    """Create a test OME-Zarr file"""
    zarr_format = 3 if use_v3 else 2

    # Create root group
    root = zarr.open_group(store_path, mode='w', zarr_format=zarr_format)

    # Create multi-resolution data
    # Level 0: full resolution
    data_0 = np.random.rand(1, 3, 100, 100).astype('float32')
    arr_0 = root.create_array('0', data=data_0, chunks=(1, 1, 50, 50))

    # Level 1: half resolution
    data_1 = np.random.rand(1, 3, 50, 50).astype('float32')
    arr_1 = root.create_array('1', data=data_1, chunks=(1, 1, 25, 25))

    # Add OME-Zarr metadata
    multiscales_version = '0.5' if use_v3 else '0.4'
    root.attrs['multiscales'] = [{
        'version': multiscales_version,
        'name': 'test',
        'axes': [
            {'name': 't', 'type': 'time'},
            {'name': 'c', 'type': 'channel'},
            {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'x', 'type': 'space', 'unit': 'micrometer'}
        ],
        'datasets': [
            {'path': '0'},
            {'path': '1'}
        ]
    }]

    root.attrs['omero'] = {
        'channels': [
            {'label': 'channel1', 'color': '00FFFF', 'window': {'start': 0, 'end': 1, 'min': 0, 'max': 1}},
            {'label': 'channel2', 'color': 'FF00FF', 'window': {'start': 0, 'end': 1, 'min': 0, 'max': 1}},
            {'label': 'channel3', 'color': 'FFFF00', 'window': {'start': 0, 'end': 1, 'min': 0, 'max': 1}}
        ],
        'name': 'test_image'
    }

    return store_path


def test_ome_zarr_reader(use_v3=True):
    """Test the OmeZarrReader class"""
    tmpdir = tempfile.mkdtemp()
    store_path = os.path.join(tmpdir, 'test_ome.zarr')

    try:
        # Import here to catch any import errors
        try:
            from cyto_dl.image.io.ome_zarr_reader import OmeZarrReader
            from upath import UPath as Path
        except ImportError as e:
            print(f"⊘ SKIP: Missing dependency: {e}")
            return None

        print(f"\nTesting OmeZarrReader with zarr {'v3' if use_v3 else 'v2'}...")

        # Create test data
        create_test_ome_zarr(store_path, use_v3=use_v3)
        print(f"  Created test OME-Zarr at {store_path}")

        # Test basic reading
        reader = OmeZarrReader(level=0)
        img_obj = reader.read(store_path)
        print(f"  ✓ Read image object: {type(img_obj)}")

        # Test getting data
        img_data, metadata = reader.get_data(img_obj)
        print(f"  ✓ Got data with shape: {img_data.shape}")
        print(f"  ✓ Expected shape: (3, 100, 100) - First dim removed by [0] indexing")

        if img_data.shape != (3, 100, 100):
            print(f"  ✗ ERROR: Shape mismatch! Expected (3, 100, 100), got {img_data.shape}")
            return False

        # Test level 1
        reader1 = OmeZarrReader(level=1)
        img_obj1 = reader1.read(store_path)
        img_data1, metadata1 = reader1.get_data(img_obj1)
        print(f"  ✓ Level 1 data shape: {img_data1.shape}")

        if img_data1.shape != (3, 50, 50):
            print(f"  ✗ ERROR: Level 1 shape mismatch! Expected (3, 50, 50), got {img_data1.shape}")
            return False

        # Test channel selection
        reader_ch = OmeZarrReader(level=0, channels=[0, 1])
        img_obj_ch = reader_ch.read(store_path)
        img_data_ch, metadata_ch = reader_ch.get_data(img_obj_ch)
        print(f"  ✓ Channel selection data shape: {img_data_ch.shape}")

        if img_data_ch.shape != (2, 100, 100):
            print(f"  ✗ ERROR: Channel selection shape mismatch! Expected (2, 100, 100), got {img_data_ch.shape}")
            return False

        # Test channel selection by name
        reader_ch_name = OmeZarrReader(level=0, channels=['channel1', 'channel3'])
        img_obj_ch_name = reader_ch_name.read(store_path)
        img_data_ch_name, metadata_ch_name = reader_ch_name.get_data(img_obj_ch_name)
        print(f"  ✓ Channel selection by name data shape: {img_data_ch_name.shape}")

        if img_data_ch_name.shape != (2, 100, 100):
            print(f"  ✗ ERROR: Channel by name shape mismatch! Expected (2, 100, 100), got {img_data_ch_name.shape}")
            return False

        # Test verify_suffix
        if not reader.verify_suffix(store_path):
            print(f"  ✗ ERROR: verify_suffix failed for .zarr file")
            return False
        print(f"  ✓ verify_suffix works correctly")

        print(f"  ✓ All OmeZarrReader tests passed for zarr {'v3' if use_v3 else 'v2'}!")
        return True

    except Exception as e:
        print(f"  ✗ ERROR in OmeZarrReader test: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    print("="*60)
    print("Testing OmeZarrReader with zarr-3 compatibility")
    print("="*60)

    test_v3 = test_ome_zarr_reader(use_v3=True)
    test_v2 = test_ome_zarr_reader(use_v3=False)

    print("\n" + "="*60)
    print("Summary:")
    if test_v3 is not None:
        print(f"  OmeZarrReader with zarr v3: {'✓ PASS' if test_v3 else '✗ FAIL'}")
    else:
        print(f"  OmeZarrReader with zarr v3: ⊘ SKIP (dependencies not available)")

    if test_v2 is not None:
        print(f"  OmeZarrReader with zarr v2: {'✓ PASS' if test_v2 else '✗ FAIL'}")
    else:
        print(f"  OmeZarrReader with zarr v2: ⊘ SKIP (dependencies not available)")
    print("="*60)
