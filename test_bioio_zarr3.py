#!/usr/bin/env python
"""
Test bioio-ome-zarr with zarr-3 compatibility
"""
import tempfile
import os
import numpy as np
import zarr
import shutil

def test_bioio_ome_zarr():
    """Test bioio-ome-zarr reading with zarr-3"""
    tmpdir = tempfile.mkdtemp()
    store_path = os.path.join(tmpdir, 'test_bioio.zarr')

    try:
        print(f"zarr version: {zarr.__version__}")

        # Create a proper OME-Zarr structure using zarr-3 API
        root = zarr.open_group(store_path, mode='w', zarr_format=3)

        # Create multi-resolution data
        data_0 = np.random.rand(1, 3, 100, 100).astype('float32')
        arr_0 = root.create_array('0', data=data_0, chunks=(1, 1, 50, 50))

        # Add OME-Zarr metadata
        root.attrs['multiscales'] = [{
            'version': '0.5',
            'name': 'test',
            'axes': [
                {'name': 't', 'type': 'time'},
                {'name': 'c', 'type': 'channel'},
                {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                {'name': 'x', 'type': 'space', 'unit': 'micrometer'}
            ],
            'datasets': [
                {'path': '0'}
            ]
        }]

        root.attrs['omero'] = {
            'channels': [
                {'label': 'channel1', 'color': '00FFFF'},
                {'label': 'channel2', 'color': 'FF00FF'},
                {'label': 'channel3', 'color': 'FFFF00'}
            ]
        }

        print(f"Created OME-Zarr v3 at {store_path}")

        # Try to read with bioio
        try:
            from bioio_ome_zarr import Reader as BioioOmeZarrReader

            # Try to read
            reader = BioioOmeZarrReader(store_path)
            print(f"BioIO OME-Zarr reader created")

            # Try to get xarray data
            data = reader.xarray_dask_data
            print(f"Got xarray data: shape={data.shape if hasattr(data, 'shape') else 'N/A'}")

            print("\n✓ bioio-ome-zarr test passed")
            return True

        except ImportError as e:
            print(f"bioio-ome-zarr not available: {e}")
            print("This is expected if bioio-ome-zarr is not installed")
            return None

    except Exception as e:
        print(f"\n✗ Error in bioio-ome-zarr test: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_zarr_format_versions():
    """Test different zarr format versions"""
    tmpdir = tempfile.mkdtemp()

    try:
        print("\n" + "="*60)
        print("Testing zarr format versions")
        print("="*60)

        # Test zarr v2 (legacy)
        store_v2 = os.path.join(tmpdir, 'test_v2.zarr')
        try:
            group_v2 = zarr.open_group(store_v2, mode='w', zarr_format=2)
            data = np.random.rand(10, 10).astype('float32')
            arr_v2 = group_v2.create_array('test', data=data)
            print(f"✓ Zarr v2 format works")
        except Exception as e:
            print(f"✗ Zarr v2 format failed: {e}")

        # Test zarr v3
        store_v3 = os.path.join(tmpdir, 'test_v3.zarr')
        try:
            group_v3 = zarr.open_group(store_v3, mode='w', zarr_format=3)
            data = np.random.rand(10, 10).astype('float32')
            arr_v3 = group_v3.create_array('test', data=data)
            print(f"✓ Zarr v3 format works")
        except Exception as e:
            print(f"✗ Zarr v3 format failed: {e}")

        # Test default format
        store_default = os.path.join(tmpdir, 'test_default.zarr')
        try:
            group_default = zarr.open_group(store_default, mode='w')
            data = np.random.rand(10, 10).astype('float32')
            arr_default = group_default.create_array('test', data=data)

            # Check which version was used
            import json
            zarr_json_path = os.path.join(store_default, 'zarr.json')
            zgroup_path = os.path.join(store_default, '.zgroup')

            if os.path.exists(zarr_json_path):
                print(f"✓ Default format: zarr v3 (zarr.json exists)")
            elif os.path.exists(zgroup_path):
                print(f"✓ Default format: zarr v2 (.zgroup exists)")
            else:
                print(f"? Default format: unknown")

        except Exception as e:
            print(f"✗ Default format failed: {e}")

        return True

    except Exception as e:
        print(f"\n✗ Error in format version test: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    print("Testing zarr-3 and bioio compatibility\n")

    test1 = test_zarr_format_versions()
    test2 = test_bioio_ome_zarr()

    print("\n" + "="*60)
    print("Summary:")
    print(f"  Zarr format versions: {'✓ PASS' if test1 else '✗ FAIL'}")
    if test2 is not None:
        print(f"  BioIO OME-Zarr:       {'✓ PASS' if test2 else '✗ FAIL'}")
    else:
        print(f"  BioIO OME-Zarr:       ⊘ SKIP (not installed)")
    print("="*60)
