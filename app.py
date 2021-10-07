import copy
import json

import cachey
import rioxarray  # noqa: F401
import xarray as xr
from carbonplan_data.metadata import get_cf_global_attrs
from carbonplan_data.utils import set_zarr_encoding
from fastapi import Depends, FastAPI, HTTPException
from fastapi.logger import logger
from ndpyramid import pyramid_reproject
from starlette.responses import Response
from xpublish.utils.cache import CostTimer
from xpublish.utils.zarr import (  # jsonify_zmetadata,  defined below to handle subgroups
    create_zmetadata,
    create_zvariables,
    encode_chunk,
    get_data_chunk,
)
from zarr.storage import array_meta_key, attrs_key, group_meta_key

app = FastAPI()
DATATREE_ID_ATTR_KEY = '_xpublish_id'


def get_pyramid():

    # input dataset
    path = "https://storage.googleapis.com/carbonplan-scratch/map-tests/raw/wc2.1_2.5m_tavg_10.tif"

    # open and extract the input dataset
    ds = xr.open_rasterio(path).to_dataset(name="tavg").squeeze().reset_coords(["band"], drop=True)

    # create the pyramid
    pyramid = pyramid_reproject(ds, levels=6)

    # modify the data in the pyramid
    for child in pyramid.children:
        child.ds = set_zarr_encoding(
            child.ds, codec_config={"id": "zlib", "level": 1}, float_dtype="float32"
        )
        child.ds = child.ds.chunk({"x": 128, "y": 128})
        child.ds["tavg"].attrs.clear()
    pyramid.attrs = get_cf_global_attrs()

    print(str(pyramid))

    return pyramid


def get_cache():
    print('get_cache')

    return cachey.Cache(available_bytes=1e6)


# utilities
# TODO: this should replace the current version in xpublish
def jsonify_zmetadata(zmetadata: dict) -> dict:
    """Helper function to convert zmetadata dictionary to a json
    compatible dictionary.
    """
    zjson = copy.deepcopy(zmetadata)

    for key, meta in zmetadata['metadata'].items():
        if array_meta_key not in key:
            continue
        # convert compressor to dict
        compressor = meta.get('compressor', None)
        if compressor is not None:
            compressor_config = compressor.get_config()
            zjson['metadata'][key]['compressor'] = compressor_config

    return zjson


def make_pyramid_zmetadata(pyramid):

    zmeta = {'metadata': {}, "zarr_consolidated_format": 1}

    for node in pyramid.subtree:
        prefix = node.pathstr.replace(node.root.pathstr, "")[1:]  # strips leading /, feels bad :(
        group_meta = create_zmetadata(node.ds)['metadata']

        for suffix, meta in group_meta.items():
            if prefix:
                key = prefix + '/' + suffix
            else:
                key = suffix
            zmeta['metadata'][key] = meta

    return zmeta


def get_zvariables(pyramid=Depends(get_pyramid), cache: cachey.Cache = Depends(get_cache)):
    print('get_zvariables')

    cache_key = pyramid.ds.attrs.get(DATATREE_ID_ATTR_KEY, '') + '/' + 'zvariables'
    zvariables = cache.get(cache_key)

    if zvariables is None:
        zvariables = {}

        for node in pyramid.subtree:
            ds_vars = create_zvariables(node.ds)
            prefix = node.pathstr.replace(node.root.pathstr, "")[
                1:
            ]  # strips leading /, feels bad :(

            for suffix, var in ds_vars.items():
                if prefix:
                    key = prefix + '/' + suffix
                else:
                    key = suffix
                zvariables[key] = var

        # we want to permanently cache this: set high cost value
        cache.put(cache_key, zvariables, 99999)

    print('finished get_zvariables')

    return zvariables


def _get_zmetadata(pyramid=Depends(get_pyramid)):
    print('_get_zmetadata')
    zmetadata = make_pyramid_zmetadata(pyramid)
    print('here')
    return zmetadata


@app.get('/{key}')
def get_zmetadata(key: str, zmetadata=Depends(_get_zmetadata)):
    zjson = jsonify_zmetadata(zmetadata)

    if key == '.zmetadata':
        return Response(json.dumps(zjson).encode('ascii'), media_type='application/json')
    elif key == '.zgroup':
        return Response(
            json.dumps(zjson['metadata']['.zgroup']).encode('ascii'), media_type='application/json'
        )
    elif key == '.zattrs':
        return Response(
            json.dumps(zjson['metadata']['.zattrs']).encode('ascii'), media_type='application/json'
        )
    elif key == '.zarray':
        raise HTTPException(status_code=404, detail="not an array")
    else:
        raise ValueError('key not supported')


@app.get('/{group}/{var}/{chunk}')
def get_variable_chunk(
    group: str,
    var: str,
    chunk: str,
    pyramid=Depends(get_pyramid),
    cache: cachey.Cache = Depends(get_cache),
    zvariables: dict = Depends(get_zvariables),
    zmetadata: dict = Depends(_get_zmetadata),
):

    # First check that this request wasn't for variable metadata
    if array_meta_key in chunk:
        return zmetadata['metadata'][f'{group}/{var}/{array_meta_key}']
    elif attrs_key in chunk:
        return zmetadata['metadata'][f'{group}/{var}/{attrs_key}']
    elif group_meta_key in chunk:
        return zmetadata['metadata'][f'{group}/{var}/{group_meta_key}']
    else:
        logger.debug('group is %s', group)
        logger.debug('var is %s', var)
        logger.debug('chunk is %s', chunk)

        cache_key = pyramid.ds.attrs.get(DATATREE_ID_ATTR_KEY, '') + '/' + f'{group}/{var}/{chunk}'
        response = cache.get(cache_key)

        if response is None:
            with CostTimer() as ct:
                arr_meta = zmetadata['metadata'][f'{group}/{var}/{array_meta_key}']
                da = zvariables[f'{group}/{var}'].data

                data_chunk = get_data_chunk(da, chunk, out_shape=arr_meta['chunks'])

                echunk = encode_chunk(
                    data_chunk.tobytes(),
                    filters=arr_meta['filters'],
                    compressor=arr_meta['compressor'],
                )

                response = Response(echunk, media_type='application/octet-stream')

            cache.put(cache_key, response, ct.time, len(echunk))

        return response
