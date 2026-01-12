#!/usr/bin/env python
"""
Test script to check zarr-3 compatibility with ome-zarr and bioio
"""
import tempfile
import os
import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import shutil

def test_ome_zarr_basic():
    """Test basic ome-zarr reading with zarr-3"""
    tmpdir = tempfile.mkdtemp()
    store_path = os.path.join(tmpdir, 'test_ome.zarr')

    try:
        print(f"zarr version: {zarr.__version__}")

        # Create a proper OME-Zarr structure
        root = zarr.open_group(store_path, mode='w')

        # Create multi-resolution data
        # Level 0: full resolution
        data_0 = np.random.rand(1, 3, 100, 100).astype('float32')  # (T, C, Y, X)
        arr_0 = root.create_array('0', data=data_0, chunks=(1, 1, 50, 50))

        # Level 1: half resolution
        data_1 = np.random.rand(1, 3, 50, 50).astype('float32')
        arr_1 = root.create_array('1', data=data_1, chunks=(1, 1, 25, 25))

        # Add OME-Zarr metadata
        root.attrs['multiscales'] = [{
            'version': '0.4',
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
                {'label': 'channel1', 'color': '00FFFF'},
                {'label': 'channel2', 'color': 'FF00FF'},
                {'label': 'channel3', 'color': 'FFFF00'}
            ]
        }

        print(f"Created OME-Zarr at {store_path}")
        print(f"Root attrs: {dict(root.attrs)}")

        # Try to read with ome-zarr Reader
        reader = Reader(parse_url(store_path))
        nodes = list(reader())

        print(f"\nFound {len(nodes)} nodes")

        if len(nodes) > 0:
            node = nodes[0]
            print(f"\nNode type: {type(node)}")
            print(f"Node has data: {hasattr(node, 'data')}")

            if hasattr(node, 'data'):
                print(f"Data type: {type(node.data)}")
                print(f"Data length: {len(node.data) if hasattr(node.data, '__len__') else 'N/A'}")

                # Try accessing different levels
                for level in range(min(2, len(node.data))):
                    print(f"\nLevel {level}:")
                    level_data = node.data[level]
                    print(f"  Type: {type(level_data)}")
                    print(f"  Is dask array: {hasattr(level_data, 'compute')}")

                    if hasattr(level_data, 'compute'):
                        computed = level_data.compute()
                        print(f"  Computed shape: {computed.shape}")
                        print(f"  Computed dtype: {computed.dtype}")
                    elif hasattr(level_data, 'shape'):
                        print(f"  Shape: {level_data.shape}")
                        print(f"  Dtype: {level_data.dtype}")

            if hasattr(node, 'metadata'):
                print(f"\nMetadata: {node.metadata}")

        print("\n✓ ome-zarr basic test passed")
        return True

    except Exception as e:
        print(f"\n✗ Error in ome-zarr test: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_direct_zarr_access():
    """Test direct zarr-3 access patterns"""
    tmpdir = tempfile.mkdtemp()
    store_path = os.path.join(tmpdir, 'test_direct.zarr')

    try:
        print("\n" + "="*60)
        print("Testing direct zarr-3 access")
        print("="*60)

        # Create a zarr group
        group = zarr.open_group(store_path, mode='w')

        # Create an array
        data = np.random.rand(10, 10).astype('float32')
        arr = group.create_array('test_array', data=data)

        print(f"Created array type: {type(arr)}")
        print(f"Array shape: {arr.shape}")

        # Read it back
        group_read = zarr.open_group(store_path, mode='r')
        arr_read = group_read['test_array']

        print(f"Read array type: {type(arr_read)}")
        print(f"Read array shape: {arr_read.shape}")
        print(f"Can access with [:]: {arr_read[:].shape}")

        print("\n✓ Direct zarr-3 access test passed")
        return True

    except Exception as e:
        print(f"\n✗ Error in direct zarr test: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    print("Testing zarr-3 compatibility\n")

    test1 = test_direct_zarr_access()
    test2 = test_ome_zarr_basic()

    print("\n" + "="*60)
    print("Summary:")
    print(f"  Direct zarr access: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"  OME-Zarr reading:   {'✓ PASS' if test2 else '✗ FAIL'}")
    print("="*60)
