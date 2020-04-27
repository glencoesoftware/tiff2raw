[![AppVeyor status](https://ci.appveyor.com/api/projects/status/github/tiff2raw)](https://ci.appveyor.com/project/gs-jenkins/tiff2raw)

# TIFF Converter

Python tool that converts TIFF files to an intermediary raw format.

## Requirements

* Python 3.5+

## Usage

Basic usage is:

    tiff2raw write_tiles /path/to/input.tiff /path/to/tile/directory

Please see `tiff2raw write_tiles --help` for detailed information.

Output tile width and height can optionally be specified; default values are
detailed in `--help`.

A directory structure containing the pyramid tiles at all resolutions and
macro/label images will be created.  Be mindful of available disk space and
available RAM, as larger TIFF files can result in >20 GB of tiles and
require GBs of free RAM.

Use of a n5 (the default) or zarr `--file_type` will result in losslessly
compressed output.  These are the only formats that are currently
supported by the downstream `raw2ometiff`.

## Performance

This package is __highly__ sensitive to underlying hardware as well as
the following configuration options:

 * `--max_workers`
 * `--tile_width`
 * `--tile_height`
 * `--batch_size`

On systems with significant I/O bandwidth, particularly SATA or
NVMe based storage, we have found sharply diminishing returns with worker
counts > 4.  There are significant performance gains to be had utilizing
larger tile sizes but be mindful of the consequences on the downstream
workflow.  You may find increasing the batch size on systems with very
high single core performance to give modest performance gains.

In general, expect to need to tune the above settings and measure
relative performance.

## License

The TIFF converter is distributed under the terms of the GPL license.
Please see `LICENSE.txt` for further details.
