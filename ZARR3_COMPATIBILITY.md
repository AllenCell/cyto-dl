# Zarr-3 Compatibility Summary

## Overview
This document summarizes the zarr-3 compatibility status for cyto-dl and the changes made to ensure full compatibility.

## Current Status: ✅ COMPATIBLE

cyto-dl is now fully compatible with zarr-3 format.

## Changes Made

### 1. Updated Dependencies
- Added explicit `zarr>=3.0.0` dependency in `pyproject.toml`
- Updated `ome-zarr>=0.12.0` to ensure zarr-3 compatibility
- Current versions being used:
  - `zarr==3.1.5`
  - `ome-zarr==0.12.2`
  - `bioio-ome-zarr==3.1.0`

### 2. Testing Performed

#### Basic zarr-3 API
- ✅ zarr.open_group() with zarr_format=3
- ✅ Creating and reading zarr v3 arrays
- ✅ zarr v3 is now the default format

#### ome-zarr with zarr-3
- ✅ ome_zarr.reader.Reader works with zarr-3
- ✅ Multi-resolution OME-Zarr reading
- ✅ Metadata access and channel selection
- ✅ Dask array computation

#### bioio-ome-zarr with zarr-3
- ✅ bioio_ome_zarr.Reader works with zarr-3
- ✅ xarray data access
- ✅ Multi-dimensional image reading

## Code Compatibility

### OmeZarrReader (cyto_dl/image/io/ome_zarr_reader.py)
The existing code is fully compatible with zarr-3:
```python
# Line 39: This pattern works correctly with zarr-3
data = img_obj.data[self.level].compute()[0]
```

- `img_obj.data[self.level]` returns a dask array (works with zarr-3)
- `.compute()` computes the dask array to numpy array
- `[0]` removes the time dimension

### BioIOImageLoaderd (cyto_dl/image/io/bioio_loader.py)
The bioio-based loader is fully compatible as bioio>=3.0.0 supports zarr-3.

## Migration Notes

### For Users
- No code changes required in user scripts
- Existing zarr-2 files can still be read
- New zarr files will be created in v3 format by default
- To create v2 format explicitly, use: `zarr.open_group(path, zarr_format=2)`

### For Developers
- Default format is now zarr-3
- Both v2 and v3 formats are supported for reading
- When creating new OME-Zarr files, use format version 0.5 for best zarr-3 compatibility

## Known Issues

### Minor Warning
You may see this warning when reading some OME-Zarr files:
```
version mismatch: detected: FormatV04, requested: FormatV05
```

This is benign and does not affect functionality. It occurs when reading older OME-Zarr files (format v0.4) with newer ome-zarr library that expects v0.5.

## Testing

### Automated Tests
Test scripts have been created to verify zarr-3 compatibility:
- `test_zarr3_compat.py` - Tests basic zarr-3 and ome-zarr reading
- `test_bioio_zarr3.py` - Tests bioio-ome-zarr with both zarr v2 and v3
- `test_ome_zarr_reader.py` - Tests the OmeZarrReader class

Run tests with:
```bash
python test_zarr3_compat.py
python test_bioio_zarr3.py
```

### Manual Testing
To test with your own zarr files:
```python
from cyto_dl.image.io import OmeZarrReader

reader = OmeZarrReader(level=0)
img_obj = reader.read('/path/to/file.zarr')
img_data, metadata = reader.get_data(img_obj)
print(f"Loaded image with shape: {img_data.shape}")
```

## Dependencies

Minimum versions required for zarr-3 compatibility:
- `zarr>=3.0.0`
- `ome-zarr>=0.12.0`
- `bioio>=3.0.0`
- `bioio-base>=3.0.0`
- `bioio-ome-zarr>=3.1.0`

All dependencies are now properly specified in `pyproject.toml`.

## Conclusion

cyto-dl is fully compatible with zarr-3. The codebase required minimal changes (only dependency version updates) as the existing code was already using the correct API patterns.
