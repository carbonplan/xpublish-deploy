import copy
import importlib
import json
from typing import List

import cachey
import dask
import datatree as dt
import numpy as np
import xarray as xr
from carbonplan_data.metadata import get_cf_global_attrs
from carbonplan_data.utils import set_zarr_encoding
from fastapi import Depends, FastAPI, HTTPException
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from xpublish.utils.cache import CostTimer
from xpublish.utils.zarr import create_zmetadata, create_zvariables, encode_chunk, get_data_chunk
from zarr.storage import array_meta_key, attrs_key, group_meta_key

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])
DATATREE_ID_ATTR_KEY = "_xpublish_id"


cache = cachey.Cache(available_bytes=1e9)


def _get_version():
    try:
        return importlib.import_module("ndpyramid").__version__
    except ModuleNotFoundError:
        return "-9999"


def _multiscales_template(datasets=[], type="", method="", version="", args=[], kwargs={}):
    # https://forum.image.sc/t/multiscale-arrays-v0-1/37930
    d = [
        {
            "datasets": datasets,
            "type": type,
            "metadata": {
                "method": method,
                "version": version,
                "args": args,
                "kwargs": kwargs,
            },
        }
    ]
    return d


def pyramid_coarsen(ds, factors: List[int], dims: List[str], **kwargs) -> dt.DataTree:
    # multiscales spec
    save_kwargs = locals()
    del save_kwargs["ds"]

    attrs = {
        "multiscales": _multiscales_template(
            datasets=[{"path": str(i) for i in range(len(factors))}],
            type="reduce",
            method="pyramid_coarsen",
            version=_get_version(),
            kwargs=save_kwargs,
        )
    }

    # set up pyramid
    root = xr.Dataset(attrs=attrs)
    pyramid = dt.DataTree(data_objects={"root": root})

    # pyramid data
    for key, factor in enumerate(factors):
        skey = str(key)
        kwargs.update({d: factor for d in dims})
        pyramid[skey] = ds.coarsen(**kwargs).mean()

    return pyramid


def make_grid_ds(ds, dim, transform):
    grid_shape = (dim, dim)
    bounds_shape = (dim + 1, dim + 1)

    xs = np.empty(grid_shape)
    ys = np.empty(grid_shape)
    for i in range(bounds_shape[0]):
        for j in range(bounds_shape[1]):
            if i < grid_shape[0] and j < grid_shape[1]:
                x, y = transform * [j + 0.5, i + 0.5]
                xs[i, j] = x
                ys[i, j] = y

    ds = ds.assign_coords({"x": xr.DataArray(xs[0, :], dims=["x"])})
    ds = ds.assign_coords({"y": xr.DataArray(ys[:, 0], dims=["y"])})
    # ds.attrs['title'] = "Web Mercator Grid"
    # ds.attrs['Convensions'] = "CF-1.8"

    return ds


def pyramid_reproject(
    ds, levels: int = None, pixels_per_tile=128, resampling="average", extra_dim=None
) -> dt.DataTree:
    from rasterio.transform import Affine
    from rasterio.warp import Resampling

    # multiscales spec
    save_kwargs = {"levels": levels, "pixels_per_tile": pixels_per_tile}
    attrs = {
        "multiscales": _multiscales_template(
            datasets=[{"path": str(i) for i in range(levels)}],
            type="reduce",
            method="pyramid_reproject",
            version=_get_version(),
            kwargs=save_kwargs,
        )
    }

    # set up pyramid
    root = xr.Dataset(attrs=attrs)
    pyramid = dt.DataTree(data_objects={"root": root})

    def make_template(da, dim, dst_transform, shape=None):
        template = xr.DataArray(
            data=dask.array.empty(shape, chunks=shape), dims=("y", "x"), attrs=da.attrs
        )
        template = make_grid_ds(template, dim, dst_transform)
        template.coords["spatial_ref"] = xr.DataArray(np.array(1.0))
        return template

    # pyramid data
    def reproject(da, shape=None, dst_transform=None, resampling="average"):
        return da.rio.reproject(
            "EPSG:3857",
            resampling=Resampling[resampling],
            shape=shape,
            transform=dst_transform,
        )

    # this should look the same as template

    for level in range(levels):
        lkey = str(level)
        dim = 2**level * pixels_per_tile

        dst_transform = Affine.translation(-20026376.39, 20048966.10) * Affine.scale(
            (20026376.39 * 2) / dim, -(20048966.10 * 2) / dim
        )

        pyramid[lkey] = xr.Dataset(attrs=ds.attrs)
        for k, da in ds.items():
            template = make_template(ds[k], dim, dst_transform, (dim, dim))
            pyramid[lkey].ds[k] = xr.map_blocks(
                reproject,
                da,
                kwargs=dict(shape=(dim, dim), dst_transform=dst_transform),
                template=template,
            )

    return pyramid


def get_cache():
    print("getting cache")
    return cachey.Cache(available_bytes=1e9)


def get_pyramid():
    with CostTimer() as ct:
        print("fetching pyramid")
        cache_key = "pyramid"
        pyramid = cache.get(cache_key)
        if pyramid is None:
            print("generating pyramid, since cache pyr doesn't exist")
            # with CostTimer() as ct:
            # print('caching pyramid')

            # input dataset
            path = "https://storage.googleapis.com/carbonplan-share/maps-demo/raw/wc2.1_2.5m_tavg_10.tif"

            # open and extract the input dataset
            ds = (
                xr.open_rasterio(path)
                .to_dataset(name="tavg")
                .squeeze()
                .reset_coords(["band"], drop=True)
                .chunk(-1)
            )

            # create the pyramid
            pyramid = pyramid_reproject(ds, levels=6)

            # modify the data in the pyramid
            for child in pyramid.children:
                child.ds = set_zarr_encoding(
                    child.ds,
                    codec_config={"id": "zlib", "level": 1},
                    float_dtype="float32",
                )
                child.ds = child.ds.chunk({"x": 128, "y": 128})
                child.ds["tavg"].attrs.clear()
            pyramid.attrs = get_cf_global_attrs()

            cache.put(cache_key, pyramid, 99999)
            print("adding pyramid to cache")
    print(ct.time)

    return pyramid


# utilities
# TODO: this should replace the current version in xpublish
def jsonify_zmetadata(zmetadata: dict) -> dict:
    """Helper function to convert zmetadata dictionary to a json
    compatible dictionary.
    """
    zjson = copy.deepcopy(zmetadata)

    for key, meta in zmetadata["metadata"].items():
        if array_meta_key not in key:
            continue
        # convert compressor to dict
        compressor = meta.get("compressor", None)
        if compressor is not None:
            compressor_config = compressor.get_config()
            zjson["metadata"][key]["compressor"] = compressor_config

    return zjson


def make_pyramid_zmetadata(pyramid):
    zmeta = {"metadata": {}, "zarr_consolidated_format": 1}

    for node in pyramid.subtree:
        # print(node.name)
        prefix = node.pathstr.replace(node.root.pathstr, "")[1:]  # strips leading /, feels bad :(
        group_meta = create_zmetadata(node.ds)["metadata"]
        for suffix, meta in group_meta.items():
            if prefix:
                key = prefix + "/" + suffix
            else:
                key = suffix
            zmeta["metadata"][key] = meta

    return zmeta


def get_zvariables(pyramid=Depends(get_pyramid)):
    print("get_zvariables")

    cache_key = pyramid.ds.attrs.get(DATATREE_ID_ATTR_KEY, "") + "/" + "zvariables"
    zvariables = cache.get(cache_key)

    if zvariables is None:
        print("caching zvars")
        zvariables = {}

        for node in pyramid.subtree:
            ds_vars = create_zvariables(node.ds)
            prefix = node.pathstr.replace(node.root.pathstr, "")[
                1:
            ]  # strips leading /, feels bad :(

            for suffix, var in ds_vars.items():
                if prefix:
                    key = prefix + "/" + suffix
                else:
                    key = suffix
                zvariables[key] = var

        # we want to permanently cache this: set high cost value
        cache.put(cache_key, zvariables, 99999)

    print("finished get_zvariables")

    return zvariables


def _get_zmetadata(pyramid=Depends(get_pyramid)):
    print("_get_zmetadata")
    zmetadata = make_pyramid_zmetadata(pyramid)
    return zmetadata


@app.get("/{key}")
def get_zmetadata(key: str, zmetadata=Depends(_get_zmetadata)):
    zjson = jsonify_zmetadata(zmetadata)

    if key == ".zmetadata":
        return Response(json.dumps(zjson).encode("ascii"), media_type="application/json")
    elif key == ".zgroup":
        return Response(
            json.dumps(zjson["metadata"][".zgroup"]).encode("ascii"),
            media_type="application/json",
        )
    elif key == ".zattrs":
        return Response(
            json.dumps(zjson["metadata"][".zattrs"]).encode("ascii"),
            media_type="application/json",
        )
    elif key == ".zarray":
        raise HTTPException(status_code=404, detail="not an array")
    else:
        raise ValueError("key not supported")


@app.get("/{group}/{var}/{chunk}")
def get_variable_chunk(
    group: str,
    var: str,
    chunk: str,
    pyramid=Depends(get_pyramid),
    zvariables: dict = Depends(get_zvariables),
    zmetadata: dict = Depends(_get_zmetadata),
):
    print("get request for group/var/chunk")
    print(cache.data.keys())
    # if cache.data == None:
    #     print('cache is empty on call')

    # First check that this request wasn't for variable metadata
    if array_meta_key in chunk:
        return zmetadata["metadata"][f"{group}/{var}/{array_meta_key}"]
    elif attrs_key in chunk:
        return zmetadata["metadata"][f"{group}/{var}/{attrs_key}"]
    elif group_meta_key in chunk:
        return zmetadata["metadata"][f"{group}/{var}/{group_meta_key}"]
    else:
        logger.debug("group is %s", group)
        logger.debug("var is %s", var)
        logger.debug("chunk is %s", chunk)

        cache_key = pyramid.ds.attrs.get(DATATREE_ID_ATTR_KEY, "") + "/" + f"{group}/{var}/{chunk}"
        response = cache.get(cache_key)

        if response is None:
            with CostTimer() as ct:
                arr_meta = zmetadata["metadata"][f"{group}/{var}/{array_meta_key}"]
                da = zvariables[f"{group}/{var}"].data

                data_chunk = get_data_chunk(da, chunk, out_shape=arr_meta["chunks"])

                echunk = encode_chunk(
                    data_chunk.tobytes(),
                    filters=arr_meta["filters"],
                    compressor=arr_meta["compressor"],
                )

                response = Response(echunk, media_type="application/octet-stream")

            cache.put(cache_key, response, ct.time, len(echunk))

        return response
