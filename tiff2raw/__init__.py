#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Glencoe Software, Inc. All rights reserved.
#
# This software is distributed under the terms described by the LICENSE.txt
# file you can find at the root of the distribution bundle.  If the file is
# missing please request a copy by contacting info@glencoesoftware.com

import logging
import math
import os

import numpy as np
import zarr

from datetime import datetime
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from threading import BoundedSemaphore

from kajiki import PackageLoader
from skimage.transform import resize
from tifffile import imread


log = logging.getLogger(__name__)

# Minimum size of the largest XY dimension in the smallest resolution,
# when calculating the number of resolutions to generate.
MIN_SIZE = 256

# Scaling factor in X and Y between any two consecutive resolutions. */
PYRAMID_SCALE = 2

# Version of the bioformats2raw layout.
LAYOUT = 1


class MaxQueuePool(object):
    """This Class wraps a concurrent.futures.Executor
    limiting the size of its task queue.
    If `max_queue_size` tasks are submitted, the next call to submit will
    block until a previously submitted one is completed.

    Brought in from:
      * https://gist.github.com/noxdafox/4150eff0059ea43f6adbdd66e5d5e87e

    See also:
      * https://www.bettercodebytes.com/
            theadpoolexecutor-with-a-bounded-queue-in-python/
      * https://pypi.org/project/bounded-pool-executor/
      * https://bugs.python.org/issue14119
      * https://bugs.python.org/issue29595
      * https://github.com/python/cpython/pull/143
    """
    def __init__(self, executor, max_queue_size, max_workers=None):
        if max_workers is None:
            max_workers = max_queue_size
        self.pool = executor(max_workers=max_workers)
        self.pool_queue = BoundedSemaphore(max_queue_size)

    def submit(self, function, *args, **kwargs):
        """Submits a new task to the pool, blocks if Pool queue is full."""
        self.pool_queue.acquire()

        future = self.pool.submit(function, *args, **kwargs)
        future.add_done_callback(self.pool_queue_callback)

        return future

    def pool_queue_callback(self, _):
        """Called once task is done, releases one queue slot."""
        self.pool_queue.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.pool.__exit__(exception_type, exception_value, traceback)


class WriteTiles(object):

    def __init__(
        self, tile_width, tile_height, resolutions, file_type, max_workers,
        dimension_order, input_path, output_path
    ):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.resolutions = resolutions
        self.file_type = file_type
        self.max_workers = max_workers
        self.dimension_order = dimension_order
        self.input_path = input_path
        self.slide_directory = output_path

        os.mkdir(self.slide_directory)

    def __enter__(self):
        self.input = imread(self.input_path)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def write_metadata(self):
        '''write metadata to a OME-XML file'''

        timestamp = datetime.now()  # TODO
        xml_values = {
            'image': {
                'name': "",  # TODO
                'acquisitionDate': timestamp.isoformat(),
                'description': "",  # TODO
                'pixels': {
                    'sizeX': self.input.shape[-1],
                    'sizeY': self.input.shape[-2],
                    'physicalSizeX': 1,  # TODO
                    'physicalSizeY': 1   # TODO
                }
            },
        }
        loader = PackageLoader()
        template = loader.import_("tiff2raw.resources.ome_template")
        xml = template(xml_values).render()
        ome_xml_file = os.path.join(self.slide_directory, "METADATA.ome.xml")
        with open(ome_xml_file, "w") as omexml:
            omexml.write(xml)

    def create_dataset(self, resolution, width, height):
        tile_directory = os.path.join(
            self.slide_directory, "data.%s" % self.file_type
        )
        self.zarr_store = zarr.DirectoryStore(tile_directory)
        if self.file_type == "n5":
            self.zarr_store = zarr.N5Store(tile_directory)
        self.zarr_group = zarr.group(store=self.zarr_store)
        shape = ([1] * (5 - len(self.input.shape))) + list(self.input.shape)
        # numpy dimension order is reversed
        dimension_order = self.dimension_order[::-1]
        shape[dimension_order.index('X')] = width
        shape[dimension_order.index('Y')] = height
        self.zarr_group.create_dataset(
            '0/%d' % resolution, shape=shape,
            chunks=(1, 1, 1, self.tile_height, self.tile_width),
            dtype=self.input.dtype,
        )
        return tile_directory

    def get_zct_coords(self, order, zSize, cSize, tSize, num, index):
        '''ported from FormatTools.getZCTCoords()'''
        # check DimensionOrder
        if not order.startswith('XY') and not order.startswith('YX'):
            raise ValueError('Invalid dimension order: %s' % order)
        iz = order.index('Z') - 2
        ic = order.index('C') - 2
        it = order.index('T') - 2
        if iz < 0 or iz > 2 or ic < 0 or ic > 2 or it < 0 or it > 2:
            raise ValueError('Invalid dimension order: %s' % order)

        # check SizeZ
        if zSize <= 0:
            raise ValueError('Invalid Z size: %d' % zSize)

        # check SizeC
        if cSize <= 0:
            raise ValueError('Invalid C size: %d' % cSize)

        # check SizeT
        if tSize <= 0:
            raise ValueError('Invalid T size: %d' % tSize)

        # check image count
        if num <= 0:
            raise ValueError("Invalid image count: %d" % num)
        if num != zSize * cSize * tSize:
            # if this happens, there is probably a bug in metadata population
            # either one of the ZCT sizes, or the total number of images
            # or else the input file is invalid
            raise ValueError(
                'ZCT size vs image count mismatch (sizeZ=%d, sizeC=%d, '
                'sizeT=%d, total=%d)' % (
                    zSize, cSize, tSize, num
                )
            )
        if index < 0 or index >= num:
            raise ValueError("Invalid image index: %d/%d" % (index, num))

        # assign rasterization order
        if iz == 0:
            len0 = zSize
        elif ic == 0:
            len0 = cSize
        else:
            len0 = tSize

        if iz == 1:
            len1 = zSize
        if ic == 1:
            len1 = cSize
        else:
            len1 = tSize

        v0 = int(index % len0)
        v1 = int(index / len0 % len1)
        v2 = int(index / len0 / len1)
        # z
        if iz == 0:
            z = v0
        elif iz == 1:
            z = v1
        else:
            z = v2
        # c
        if ic == 0:
            c = v0
        elif ic == 1:
            c = v1
        else:
            c = v2
        # t
        if it == 0:
            t = v0
        elif it == 1:
            t = v1
        else:
            t = v2

        return (z, c, t)

    def get_grid_position(self, resolution, plane):
        '''ported from Converter.getGridPosition()'''
        z = self.zarr_group['0/%d' % resolution]
        # numpy dimension order is reversed
        dimension_order = self.dimension_order[::-1]
        plane_count = np.prod(self.input.shape[0:-2])
        z_index = dimension_order.index('Z')
        c_index = dimension_order.index('C')
        t_index = dimension_order.index('T')
        zct = self.get_zct_coords(
            self.dimension_order,
            z.shape[z_index], z.shape[c_index], z.shape[t_index],
            plane_count, plane
        )
        indexes = [0, 0, 0]
        indexes[z_index] = zct[0]
        indexes[c_index] = zct[1]
        indexes[t_index] = zct[2]
        return indexes

    def write_tiles(self):
        '''write the slide's pyramid as a set of tiles'''
        size_x = self.input.shape[-1]
        size_y = self.input.shape[-2]
        # numpy dimension order is reversed
        dimension_order = self.dimension_order[::-1]
        plane_count = np.prod(self.input.shape[0:-2])
        if self.resolutions is None:
            resolutions = 0
            width = size_x
            height = size_y
            while (width > MIN_SIZE) or (height > MIN_SIZE):
                self.create_dataset(resolutions, width, height)
                width = math.ceil(width / PYRAMID_SCALE)
                height = math.ceil(height / PYRAMID_SCALE)
                resolutions += 1
            resolutions = range(resolutions)
        else:
            resolutions = range(self.resolutions)
            width = size_x
            height = size_y
            for resolution in resolutions:
                self.create_dataset(resolution, width, height)
                width = math.ceil(width / PYRAMID_SCALE)
                height = math.ceil(height / PYRAMID_SCALE)

        def write_tile(
            pixels, resolution, plane, x_start, y_start, tile_width,
            tile_height
        ):
            path = '0/%d' % resolution
            indexes = self.get_grid_position(resolution, plane)
            if resolution > 0:
                pixels = resize(
                    pixels,
                    tuple([
                        math.ceil(v / PYRAMID_SCALE) for v in pixels.shape
                    ]),
                    preserve_range=True
                )
                log.debug('Resized pixels to %s' % repr(pixels.shape))
                x_start = math.ceil(x_start / PYRAMID_SCALE)
                y_start = math.ceil(y_start / PYRAMID_SCALE)
            x_end = x_start + tile_width
            y_end = y_start + tile_height
            try:
                z = self.zarr_group[path]
                z[
                    indexes[0], indexes[1], indexes[2],
                    y_start:y_end, x_start:x_end
                ] = pixels
                log.debug(
                    "Wrote tile %r at [%s]%r[%d:%d, %d:%d]" % (
                        pixels.shape, path, indexes, x_start, x_end,
                        y_start, y_end
                    )
                )
            except Exception:
                log.error(
                    "Failed to write tile %r at [%s]%r[%d:%d, %d:%d]" % (
                        pixels.shape, path, indexes, x_start, x_end,
                        y_start, y_end
                    ), exc_info=True
                )

        log.debug("Input shape %s" % repr(self.input.shape))
        input = self.input.reshape(self.zarr_group['0/0'].shape)
        log.debug("Input reshaped to %s" % repr(input.shape))
        for resolution in resolutions:
            dataset = self.zarr_group['0/%d' % resolution]
            resolution_x_size = dataset.shape[dimension_order.index('X')]
            resolution_y_size = dataset.shape[dimension_order.index('Y')]

            x_tiles = math.ceil(resolution_x_size / self.tile_width)
            y_tiles = math.ceil(resolution_y_size / self.tile_height)

            log.info("# of X (%d) tiles = %d" % (self.tile_width, x_tiles))
            log.info("# of Y (%d) tiles = %d" % (self.tile_height, y_tiles))

            jobs = []
            with MaxQueuePool(ThreadPoolExecutor, self.max_workers) as pool:
                for plane in range(0, plane_count):
                    for y_start in range(
                        0, resolution_y_size, self.tile_height
                    ):
                        y_end = min(
                            resolution_y_size,
                            y_start + self.tile_height
                        )
                        if resolution > 0:
                            y_start *= PYRAMID_SCALE
                            y_end *= PYRAMID_SCALE
                        for x_start in range(
                            0, resolution_x_size, self.tile_width
                        ):
                            x_end = min(
                                resolution_x_size,
                                x_start + self.tile_width
                            )
                            if resolution > 0:
                                x_start *= PYRAMID_SCALE
                                x_end *= PYRAMID_SCALE
                            indexes = self.get_grid_position(resolution, plane)
                            pixels = input[
                                indexes[0], indexes[1], indexes[2],
                                y_start:y_end, x_start:x_end
                            ]
                            log.debug(
                                'Pixels %s retrieved from '
                                '[%d, %d, %d, %d:%d, %d:%d]' % (
                                    repr(pixels.shape),
                                    indexes[0], indexes[1], indexes[2],
                                    y_start, y_end, x_start, x_end
                                )
                            )
                            jobs.append(pool.submit(
                                write_tile, pixels, resolution, plane,
                                x_start, y_start,
                                self.tile_width, self.tile_height
                            ))
                wait(jobs, return_when=ALL_COMPLETED)
                input = self.zarr_group['0/%d' % resolution]
