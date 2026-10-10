# ============================================================
# NEXRAD LEVEL-II RADAR BACKEND
# FAST / CACHED VERSION
#
# Products:
#   REF - Reflectivity
#   VEL - Radial Velocity
#   SW  - Spectrum Width
#   ZDR - Differential Reflectivity
#   CC  - Correlation Coefficient
#   KDP - Specific Differential Phase (derived)
#
# Design:
#   - Native Level-II data for interrogation
#   - Downsampled data for display
#   - Geometry cache
#   - Decoded-volume cache
#   - Rendered PNG cache
#   - Latest 10 frames
#   - KDMX coordinate fallback/fix
# ============================================================

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

import boto3
from botocore import UNSIGNED
from botocore.config import Config

import xradar as xd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import OrderedDict
from urllib.parse import urlencode

import io
import math
import os
import re
import time
import traceback


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# GENERAL CONFIGURATION
# ============================================================

NEXRAD_BUCKET = "unidata-nexrad-level2"

DEFAULT_RADAR = "KDMX"

FRAME_COUNT = 10

FRAME_LIST_TTL_SECONDS = 45

DEFAULT_MAX_RANGE_KM = 230.0
MAX_ALLOWED_RANGE_KM = 460.0


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# Keep all 10 animation volumes decoded if RAM permits.
MAX_DECODED_VOLUMES = 10

# Keep more rendered combinations available.
MAX_RENDERED_FRAMES = 120

# Cache geometry separately from radar values.
MAX_GEOMETRY_CACHE = 30

# Display raster width.
# Native Level-II data is NOT reduced for interrogation.
DISPLAY_WIDTH_PX = 800

DISPLAY_MIN_HEIGHT_PX = 500
DISPLAY_MAX_HEIGHT_PX = 1000

# Display downsampling.
#
# 1 = native resolution
# 2 = approximately 1/4 the cells
#
# Interrogation remains native resolution.
DISPLAY_RAY_STEP = 2
DISPLAY_GATE_STEP = 2

# KDP calculation parameters.
KDP_WINDOW_KM = 5.0
KDP_MIN_POINTS = 5


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CACHE_DIR = Path(
    os.environ.get(
        "RADAR_CACHE_DIR",
        str(BASE_DIR / "radar_cache")
    )
).resolve()

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EARTH CONSTANTS
# ============================================================

EARTH_RADIUS_M = 6371008.8
WEB_MERCATOR_RADIUS = 6378137.0


# ============================================================
# AWS PUBLIC NEXRAD ARCHIVE
# ============================================================

s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(
        signature_version=UNSIGNED
    )
)


# ============================================================
# KNOWN RADAR LOCATIONS
#
# IMPORTANT:
# KDMX is resolved here BEFORE xradar metadata is examined.
#
# This prevents the previous:
#
# RuntimeError:
# Unable to determine radar-site coordinates.
# ============================================================

RADAR_SITE_COORDS = {

    "KDMX": (
        41.7312,
        -93.7229
    )

}


# ============================================================
# PRODUCTS
# ============================================================

PRODUCTS = {

    "REF": {

        "name":
            "Base Reflectivity",

        "fields":
            ["DBZH"],

        "units":
            "dBZ",

        "display_units":
            "dBZ",

        "vmin":
            -10,

        "vmax":
            80,

        "colors": [
            "#646464",
            "#04e9e7",
            "#019ff4",
            "#0300f4",
            "#02fd02",
            "#01c501",
            "#008e00",
            "#fdf802",
            "#e5bc00",
            "#fd9500",
            "#fd0000",
            "#d40000",
            "#bc0000",
            "#f800fd",
            "#9854c6"
        ]
    },


    "VEL": {

        "name":
            "Base Velocity",

        "fields": [
            "VRADH",
            "VRAD"
        ],

        "units":
            "m/s",

        "display_units":
            "mph",

        "vmin":
            -100,

        "vmax":
            100,

        "colors": [
            "#00ffff",
            "#0077ff",
            "#00aa66",
            "#006600",
            "#202020",
            "#777777",
            "#401010",
            "#770000",
            "#ff0000",
            "#ff66aa",
            "#ffff00"
        ]
    },


    "SW": {

        "name":
            "Spectrum Width",

        "fields": [
            "WRADH",
            "WRAD"
        ],

        "units":
            "m/s",

        "display_units":
            "mph",

        "vmin":
            0,

        "vmax":
            50,

        "colors": [
            "#101010",
            "#0055ff",
            "#00ffff",
            "#00ff55",
            "#ffff00",
            "#ff7700",
            "#ff0000",
            "#ff00ff"
        ]
    },


    "ZDR": {

        "name":
            "Differential Reflectivity",

        "fields":
            ["ZDR"],

        "units":
            "dB",

        "display_units":
            "dB",

        "vmin":
            -4,

        "vmax":
            8,

        "colors": [
            "#003cff",
            "#00ffff",
            "#00aa00",
            "#ffff00",
            "#ff8800",
            "#ff0000",
            "#ff00ff"
        ]
    },


    "CC": {

        "name":
            "Correlation Coefficient",

        "fields": [
            "RHOHV",
            "CCORH"
        ],

        "units":
            "",

        "display_units":
            "",

        "vmin":
            0.70,

        "vmax":
            1.00,

        "colors": [
            "#0000ff",
            "#00ffff",
            "#00ff00",
            "#ffff00",
            "#ff8800",
            "#ff0000",
            "#ff00ff"
        ]
    },


    "KDP": {

        "name":
            "Specific Differential Phase",

        "fields":
            ["PHIDP"],

        "units":
            "degrees/km",

        "display_units":
            "°/km",

        "vmin":
            -2,

        "vmax":
            8,

        "colors": [
            "#663399",
            "#004cff",
            "#00bfff",
            "#00ffff",
            "#00cc66",
            "#00ff00",
            "#ffff00",
            "#ff9900",
            "#ff0000",
            "#ff00ff",
            "#ffffff"
        ]
    }

}


# ============================================================
# MEMORY CACHES
# ============================================================

DECODED_VOLUME_CACHE = OrderedDict()

RASTER_CACHE = OrderedDict()

GEOMETRY_CACHE = OrderedDict()

FRAME_LIST_CACHE = {}


# ============================================================
# BASIC HELPERS
# ============================================================

def validate_site(site):

    site = str(site).strip().upper()

    if (
        len(site) != 4
        or
        not site.isalnum()
    ):

        raise ValueError(
            "Radar site must be a four-character site ID."
        )

    return site


def validate_product(product):

    product = str(product).strip().upper()

    if product not in PRODUCTS:

        raise ValueError(
            f"Unsupported radar product: {product}"
        )

    return product


def scalar_value(value):

    try:

        array = np.asarray(value)

        if array.size == 0:
            return None

        candidate = float(
            array.reshape(-1)[0]
        )

        if np.isfinite(candidate):
            return candidate

    except Exception:
        pass

    return None


def sweep_number(name):

    try:

        return int(
            str(name).split("_")[-1]
        )

    except Exception:

        return 9999


# ============================================================
# SCAN TIME
# ============================================================

def scan_time_from_key(
    key,
    fallback=None
):

    filename = str(key).rsplit(
        "/",
        1
    )[-1]

    match = re.search(
        r"[A-Z0-9]{4}(\d{8})_(\d{6})",
        filename
    )

    if match:

        try:

            timestamp = (
                match.group(1)
                +
                match.group(2)
            )

            return datetime.strptime(
                timestamp,
                "%Y%m%d%H%M%S"
            ).replace(
                tzinfo=timezone.utc
            )

        except Exception:
            pass

    return fallback


# ============================================================
# AWS FILE LISTING
# ============================================================

def list_files_for_date(
    site,
    date_value
):

    site = validate_site(site)

    prefix = (
        f"{date_value:%Y/%m/%d}/"
        f"{site}/"
    )

    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    results = []

    for page in paginator.paginate(
        Bucket=NEXRAD_BUCKET,
        Prefix=prefix
    ):

        for obj in page.get(
            "Contents",
            []
        ):

            key = obj.get(
                "Key",
                ""
            )

            filename = key.rsplit(
                "/",
                1
            )[-1]

            upper_name = filename.upper()

            if "_MDM" in upper_name:
                continue

            if upper_name.endswith(
                (
                    ".MD5",
                    ".JSON",
                    ".TXT"
                )
            ):
                continue

            if not filename.startswith(site):
                continue

            size = obj.get(
                "Size",
                0
            )

            if size <= 0:
                continue

            modified = obj.get(
                "LastModified"
            )

            scan_time = scan_time_from_key(
                key,
                modified
            )

            results.append(
                {

                    "key":
                        key,

                    "modified":
                        modified,

                    "scan_time":
                        scan_time,

                    "size":
                        size

                }
            )

    return results


# ============================================================
# FIND LATEST VOLUMES
# ============================================================

def find_latest_volumes(
    site,
    count=FRAME_COUNT,
    force=False
):

    site = validate_site(site)

    now_timestamp = time.time()

    cached = FRAME_LIST_CACHE.get(
        site
    )

    if (
        cached
        and
        not force
        and
        (
            now_timestamp
            -
            cached["timestamp"]
        )
        <
        FRAME_LIST_TTL_SECONDS
    ):

        return cached[
            "frames"
        ][-count:]

    now = datetime.now(
        timezone.utc
    )

    files = []

    # Usually today is enough.
    # Yesterday handles UTC rollover / quiet periods.
    # Third day is a safety fallback.
    for days_back in range(3):

        search_date = (
            now
            -
            timedelta(
                days=days_back
            )
        )

        day_files = (
            list_files_for_date(
                site,
                search_date
            )
        )

        files.extend(
            day_files
        )

        # Performance optimization:
        # Once we already have enough files, no need to
        # keep listing older dates.
        if len(files) >= count:

            break

    if not files:

        raise RuntimeError(
            f"No Level-II files found for {site}."
        )

    unique = {}

    for item in files:

        unique[
            item["key"]
        ] = item

    files = list(
        unique.values()
    )

    files.sort(
        key=lambda item: (
            item.get("scan_time")
            or
            item.get("modified")
            or
            datetime.min.replace(
                tzinfo=timezone.utc
            )
        )
    )

    latest = files[
        -FRAME_COUNT:
    ]

    FRAME_LIST_CACHE[
        site
    ] = {

        "timestamp":
            now_timestamp,

        "frames":
            latest

    }

    return latest[
        -count:
    ]


def find_latest_file(
    site,
    force=False
):

    frames = find_latest_volumes(
        site,
        count=1,
        force=force
    )

    if not frames:

        raise RuntimeError(
            f"No Level-II data found for {site}."
        )

    return frames[-1]


# ============================================================
# FRAME METADATA
# ============================================================

def get_frame_metadata(
    site,
    archive_key
):

    site = validate_site(site)

    archive_key = str(
        archive_key
    ).strip()

    if not archive_key:

        raise ValueError(
            "Missing radar archive key."
        )

    try:

        frames = find_latest_volumes(
            site,
            FRAME_COUNT
        )

        for frame in frames:

            if (
                frame["key"]
                ==
                archive_key
            ):

                return frame

    except Exception:
        pass

    if not re.match(
        r"^\d{4}/\d{2}/\d{2}/",
        archive_key
    ):

        raise RuntimeError(
            "Invalid NEXRAD archive key."
        )

    if (
        f"/{site}/"
        not in archive_key
    ):

        raise RuntimeError(
            "Radar archive key does not match "
            "the requested site."
        )

    filename = archive_key.rsplit(
        "/",
        1
    )[-1]

    if not filename.startswith(site):

        raise RuntimeError(
            "Radar filename does not match "
            "the requested site."
        )

    scan_time = scan_time_from_key(
        archive_key,
        None
    )

    return {

        "key":
            archive_key,

        "modified":
            scan_time,

        "scan_time":
            scan_time,

        "size":
            0

    }


# ============================================================
# DOWNLOAD RADAR FILE
# ============================================================

def download_archive_frame(
    metadata
):

    key = metadata[
        "key"
    ]

    filename = key.rsplit(
        "/",
        1
    )[-1]

    local_file = (
        CACHE_DIR
        /
        filename
    )

    if (
        local_file.exists()
        and
        local_file.stat().st_size > 0
    ):

        return local_file

    print(
        f"Downloading: {filename}"
    )

    try:

        s3.download_file(
            NEXRAD_BUCKET,
            key,
            str(local_file)
        )

    except Exception:

        try:

            if local_file.exists():
                local_file.unlink()

        except Exception:
            pass

        raise

    if (
        not local_file.exists()
        or
        local_file.stat().st_size <= 0
    ):

        raise RuntimeError(
            "Downloaded radar file is empty."
        )

    return local_file


# ============================================================
# DECODE RADAR FILE
# ============================================================

def get_volume_for_frame(
    metadata
):

    key = metadata[
        "key"
    ]

    if key in DECODED_VOLUME_CACHE:

        radar = (
            DECODED_VOLUME_CACHE.pop(
                key
            )
        )

        DECODED_VOLUME_CACHE[
            key
        ] = radar

        return radar

    local_file = (
        download_archive_frame(
            metadata
        )
    )

    print(
        f"Decoding: {local_file.name}"
    )

    try:

        radar = (
            xd.io.open_nexradlevel2_datatree(
                str(local_file)
            )
        )

    except Exception:

        try:

            if local_file.exists():
                local_file.unlink()

        except Exception:
            pass

        raise

    DECODED_VOLUME_CACHE[
        key
    ] = radar

    while (
        len(
            DECODED_VOLUME_CACHE
        )
        >
        MAX_DECODED_VOLUMES
    ):

        old_key, old_radar = (
            DECODED_VOLUME_CACHE.popitem(
                last=False
            )
        )

        print(
            f"RAM cache release: {old_key}"
        )

        try:
            old_radar.close()

        except Exception:
            pass

    return radar


def get_radar_volume(
    site,
    force=False
):

    metadata = find_latest_file(
        site,
        force=force
    )

    radar = get_volume_for_frame(
        metadata
    )

    return (
        radar,
        metadata
    )


# ============================================================
# SWEEP HELPERS
# ============================================================

def get_sweep_names(
    radar
):

    names = []

    try:

        children = list(
            radar.children
        )

    except Exception:

        try:

            children = list(
                radar.keys()
            )

        except Exception:

            children = []

    for name in children:

        name = str(name)

        if name.startswith(
            "sweep_"
        ):

            names.append(
                name
            )

    names.sort(
        key=sweep_number
    )

    return names


def get_sweep_dataset(
    radar,
    sweep_name
):

    node = radar[
        sweep_name
    ]

    try:

        return node.ds

    except Exception:

        return node


def get_sweep_elevation(
    ds
):

    if "fixed_angle" in ds:

        value = scalar_value(
            ds[
                "fixed_angle"
            ].values
        )

        if value is not None:

            return float(
                value
            )

    if "elevation" in ds:

        values = np.asarray(
            ds[
                "elevation"
            ].values,
            dtype=float
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if values.size:

            return float(
                np.nanmedian(
                    values
                )
            )

    return None


def find_field(
    ds,
    product
):

    product = validate_product(
        product
    )

    for field in PRODUCTS[
        product
    ]["fields"]:

        if field in ds.data_vars:

            return field

    return None


def find_best_sweep(
    radar,
    product,
    requested_elevation
):

    product = validate_product(
        product
    )

    requested_elevation = float(
        requested_elevation
    )

    candidates = []

    for sweep_name in get_sweep_names(
        radar
    ):

        ds = get_sweep_dataset(
            radar,
            sweep_name
        )

        field = find_field(
            ds,
            product
        )

        if field is None:
            continue

        elevation = (
            get_sweep_elevation(
                ds
            )
        )

        if elevation is None:
            continue

        candidates.append(
            (

                abs(
                    elevation
                    -
                    requested_elevation
                ),

                float(
                    elevation
                ),

                sweep_name,

                field,

                ds

            )
        )

    if not candidates:

        raise RuntimeError(
            f"No sweep containing {product} "
            "was found."
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    (
        _,
        elevation,
        sweep_name,
        field,
        ds
    ) = candidates[0]

    return (
        sweep_name,
        ds,
        field,
        elevation
    )


def get_product_elevations(
    radar,
    product
):

    product = validate_product(
        product
    )

    elevations = []

    for sweep_name in get_sweep_names(
        radar
    ):

        ds = get_sweep_dataset(
            radar,
            sweep_name
        )

        field = find_field(
            ds,
            product
        )

        if field is None:
            continue

        elevation = (
            get_sweep_elevation(
                ds
            )
        )

        if elevation is None:
            continue

        duplicate = any(
            abs(
                elevation
                -
                existing
            )
            <
            0.15

            for existing
            in elevations
        )

        if not duplicate:

            elevations.append(
                float(
                    elevation
                )
            )

    elevations.sort()

    return elevations


# ============================================================
# RADAR SITE COORDINATES
#
# KDMX FIX IS RETAINED.
#
# KDMX returns immediately from the known-site table.
# ============================================================

def get_site_coordinates(
    ds,
    site=DEFAULT_RADAR
):

    site = validate_site(
        site
    )

    # --------------------------------------------------------
    # KNOWN LOCATION FIRST
    # --------------------------------------------------------

    if site in RADAR_SITE_COORDS:

        latitude, longitude = (
            RADAR_SITE_COORDS[
                site
            ]
        )

        return (
            float(latitude),
            float(longitude)
        )

    latitude = None
    longitude = None

    latitude_names = [
        "latitude",
        "site_latitude",
        "radar_latitude",
        "lat"
    ]

    longitude_names = [
        "longitude",
        "site_longitude",
        "radar_longitude",
        "lon"
    ]

    # --------------------------------------------------------
    # COORDINATES
    # --------------------------------------------------------

    for name in latitude_names:

        try:

            if name in ds.coords:

                candidate = scalar_value(
                    ds.coords[
                        name
                    ].values
                )

                if (
                    candidate is not None
                    and
                    -90.0
                    <=
                    candidate
                    <=
                    90.0
                ):

                    latitude = candidate
                    break

        except Exception:
            pass


    for name in longitude_names:

        try:

            if name in ds.coords:

                candidate = scalar_value(
                    ds.coords[
                        name
                    ].values
                )

                if (
                    candidate is not None
                    and
                    -180.0
                    <=
                    candidate
                    <=
                    180.0
                ):

                    longitude = candidate
                    break

        except Exception:
            pass

    # --------------------------------------------------------
    # DATA VARIABLES
    # --------------------------------------------------------

    if latitude is None:

        for name in latitude_names:

            try:

                if name in ds.data_vars:

                    candidate = scalar_value(
                        ds[
                            name
                        ].values
                    )

                    if (
                        candidate is not None
                        and
                        -90.0
                        <=
                        candidate
                        <=
                        90.0
                    ):

                        latitude = candidate
                        break

            except Exception:
                pass


    if longitude is None:

        for name in longitude_names:

            try:

                if name in ds.data_vars:

                    candidate = scalar_value(
                        ds[
                            name
                        ].values
                    )

                    if (
                        candidate is not None
                        and
                        -180.0
                        <=
                        candidate
                        <=
                        180.0
                    ):

                        longitude = candidate
                        break

            except Exception:
                pass

    # --------------------------------------------------------
    # ATTRIBUTES
    # --------------------------------------------------------

    if latitude is None:

        for name in latitude_names:

            try:

                if name in ds.attrs:

                    candidate = float(
                        ds.attrs[
                            name
                        ]
                    )

                    if (
                        np.isfinite(
                            candidate
                        )
                        and
                        -90.0
                        <=
                        candidate
                        <=
                        90.0
                    ):

                        latitude = candidate
                        break

            except Exception:
                pass


    if longitude is None:

        for name in longitude_names:

            try:

                if name in ds.attrs:

                    candidate = float(
                        ds.attrs[
                            name
                        ]
                    )

                    if (
                        np.isfinite(
                            candidate
                        )
                        and
                        -180.0
                        <=
                        candidate
                        <=
                        180.0
                    ):

                        longitude = candidate
                        break

            except Exception:
                pass

    if (
        latitude is None
        or
        longitude is None
    ):

        raise RuntimeError(
            f"Unable to determine radar-site "
            f"coordinates for {site}."
        )

    return (
        float(latitude),
        float(longitude)
    )


# ============================================================
# RANGE
# ============================================================

def get_ranges_m(
    ds
):

    if "range" not in ds.coords:

        raise RuntimeError(
            "Radar sweep does not contain range."
        )

    ranges = np.asarray(
        ds[
            "range"
        ].values,
        dtype=float
    )

    units = str(
        ds[
            "range"
        ].attrs.get(
            "units",
            "meters"
        )
    ).lower()

    if (
        units.startswith(
            "km"
        )
        or
        "kilometer"
        in units
    ):

        ranges = (
            ranges
            *
            1000.0
        )

    return ranges


# ============================================================
# ORIENT MOMENT ARRAY
# ============================================================

def orient_values(
    values,
    azimuths,
    ranges
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = np.squeeze(
        values
    )

    if values.ndim != 2:

        raise RuntimeError(
            "Radar moment is not 2-D."
        )

    expected = (
        len(
            azimuths
        ),
        len(
            ranges
        )
    )

    if values.shape == expected:

        return values

    if values.T.shape == expected:

        return values.T

    raise RuntimeError(
        "Radar moment dimensions do not match "
        f"coordinates: {values.shape} vs {expected}"
    )


# ============================================================
# UNIT CONVERSION
# ============================================================

def convert_values(
    product,
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    if product in (
        "VEL",
        "SW"
    ):

        return (
            values
            *
            2.2369362920544
        )

    return values


# ============================================================
# FAST KDP
#
# KDP = 0.5 * d(PHIDP)/dr
#
# Processes contiguous valid segments.
# Uses a centered least-squares convolution kernel instead of
# performing a Python regression for every individual gate.
# ============================================================

def calculate_kdp(
    phidp,
    ranges_m,
    window_km=KDP_WINDOW_KM,
    min_points=KDP_MIN_POINTS
):

    phase = np.asarray(
        phidp,
        dtype=float
    )

    ranges_km = (
        np.asarray(
            ranges_m,
            dtype=float
        )
        /
        1000.0
    )

    output = np.full_like(
        phase,
        np.nan,
        dtype=float
    )

    if (
        phase.ndim != 2
        or
        ranges_km.size < min_points
    ):

        return output

    differences = np.diff(
        ranges_km
    )

    differences = differences[
        np.isfinite(
            differences
        )
        &
        (
            differences > 0
        )
    ]

    if differences.size == 0:

        return output

    gate_spacing_km = float(
        np.nanmedian(
            differences
        )
    )

    if gate_spacing_km <= 0:

        return output

    window_gates = max(
        min_points,
        int(
            round(
                window_km
                /
                gate_spacing_km
            )
        )
    )

    if (
        window_gates
        %
        2
        ==
        0
    ):

        window_gates += 1

    half = (
        window_gates
        //
        2
    )

    x = (
        np.arange(
            window_gates,
            dtype=float
        )
        -
        half
    ) * gate_spacing_km

    denominator = np.sum(
        x
        *
        x
    )

    if denominator <= 0:

        return output

    # Linear-regression slope kernel.
    kernel = (
        x
        /
        denominator
    )

    for ray_index in range(
        phase.shape[0]
    ):

        ray = phase[
            ray_index
        ]

        valid_indices = np.flatnonzero(
            np.isfinite(
                ray
            )
        )

        if valid_indices.size < window_gates:

            continue

        split_locations = (
            np.where(
                np.diff(
                    valid_indices
                )
                >
                1
            )[0]
            +
            1
        )

        segments = np.split(
            valid_indices,
            split_locations
        )

        for segment in segments:

            if segment.size < window_gates:

                continue

            segment_phase = ray[
                segment
            ]

            # Unwrap only within contiguous data.
            segment_phase = np.rad2deg(
                np.unwrap(
                    np.deg2rad(
                        segment_phase
                    )
                )
            )

            slopes = np.convolve(
                segment_phase,
                kernel[::-1],
                mode="valid"
            )

            centers = segment[
                half:
                segment.size
                -
                half
            ]

            if centers.size != slopes.size:

                count = min(
                    centers.size,
                    slopes.size
                )

                centers = centers[
                    :count
                ]

                slopes = slopes[
                    :count
                ]

            kdp = (
                0.5
                *
                slopes
            )

            valid_kdp = (
                np.isfinite(
                    kdp
                )
                &
                (
                    np.abs(
                        kdp
                    )
                    <=
                    20.0
                )
            )

            output[
                ray_index,
                centers[
                    valid_kdp
                ]
            ] = kdp[
                valid_kdp
            ]

    return output


# ============================================================
# PREPARE PRODUCT
# ============================================================

def prepare_product_values(
    ds,
    field,
    product,
    azimuths,
    ranges_m
):

    raw = orient_values(
        ds[
            field
        ].values,
        azimuths,
        ranges_m
    )

    if product == "KDP":

        values = calculate_kdp(
            raw,
            ranges_m
        )

    else:

        values = convert_values(
            product,
            raw
        )

    values = np.asarray(
        values,
        dtype=float
    )

    values[
        ~np.isfinite(
            values
        )
    ] = np.nan

    if product == "REF":

        values[
            values < -10.0
        ] = np.nan

    return values


# ============================================================
# COORDINATE EDGES
# ============================================================

def coordinate_edges(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    if values.size == 0:

        return np.array(
            [],
            dtype=float
        )

    if values.size == 1:

        return np.array(
            [
                values[0] - 0.5,
                values[0] + 0.5
            ],
            dtype=float
        )

    middle = (
        values[:-1]
        +
        values[1:]
    ) / 2.0

    first = (
        values[0]
        -
        (
            middle[0]
            -
            values[0]
        )
    )

    last = (
        values[-1]
        +
        (
            values[-1]
            -
            middle[-1]
        )
    )

    return np.concatenate(
        (
            [first],
            middle,
            [last]
        )
    )


# ============================================================
# SLANT -> GROUND DISTANCE
# ============================================================

def slant_to_ground_distance(
    slant_m,
    elevation_deg
):

    slant_m = np.asarray(
        slant_m,
        dtype=float
    )

    effective_radius = (
        4.0
        /
        3.0
        *
        EARTH_RADIUS_M
    )

    elevation = np.deg2rad(
        float(
            elevation_deg
        )
    )

    return (
        effective_radius
        *
        np.arctan2(

            slant_m
            *
            np.cos(
                elevation
            ),

            effective_radius
            +
            slant_m
            *
            np.sin(
                elevation
            )

        )
    )


# ============================================================
# POLAR -> LAT/LON
# ============================================================

def destination_grid(
    radar_lat,
    radar_lon,
    azimuth_edges_deg,
    distance_edges_m
):

    bearings = np.deg2rad(
        np.asarray(
            azimuth_edges_deg,
            dtype=float
        )
    )[:, None]

    angular_distance = (
        np.asarray(
            distance_edges_m,
            dtype=float
        )[None, :]
        /
        EARTH_RADIUS_M
    )

    lat1 = math.radians(
        radar_lat
    )

    lon1 = math.radians(
        radar_lon
    )

    sin_lat1 = math.sin(
        lat1
    )

    cos_lat1 = math.cos(
        lat1
    )

    sin_distance = np.sin(
        angular_distance
    )

    cos_distance = np.cos(
        angular_distance
    )

    latitude = np.arcsin(

        sin_lat1
        *
        cos_distance

        +

        cos_lat1
        *
        sin_distance
        *
        np.cos(
            bearings
        )

    )

    longitude = (

        lon1

        +

        np.arctan2(

            np.sin(
                bearings
            )
            *
            sin_distance
            *
            cos_lat1,

            cos_distance
            -
            sin_lat1
            *
            np.sin(
                latitude
            )

        )

    )

    longitude = (
        longitude
        +
        np.pi
    ) % (
        2.0
        *
        np.pi
    ) - np.pi

    return (
        np.rad2deg(
            longitude
        ),
        np.rad2deg(
            latitude
        )
    )


# ============================================================
# WEB MERCATOR
# ============================================================

def lonlat_to_mercator(
    lon,
    lat
):

    lon = np.asarray(
        lon,
        dtype=float
    )

    lat = np.asarray(
        lat,
        dtype=float
    )

    lat = np.clip(
        lat,
        -85.05112878,
        85.05112878
    )

    x = (
        WEB_MERCATOR_RADIUS
        *
        np.deg2rad(
            lon
        )
    )

    y = (
        WEB_MERCATOR_RADIUS
        *
        np.log(
            np.tan(
                np.pi
                /
                4.0
                +
                np.deg2rad(
                    lat
                )
                /
                2.0
            )
        )
    )

    return (
        x,
        y
    )


def mercator_to_lonlat(
    x,
    y
):

    lon = np.rad2deg(
        np.asarray(
            x,
            dtype=float
        )
        /
        WEB_MERCATOR_RADIUS
    )

    lat = np.rad2deg(

        2.0
        *
        np.arctan(
            np.exp(
                np.asarray(
                    y,
                    dtype=float
                )
                /
                WEB_MERCATOR_RADIUS
            )
        )

        -

        np.pi
        /
        2.0

    )

    return (
        lon,
        lat
    )


# ============================================================
# COLORMAP
# ============================================================

def product_colormap(
    product
):

    config = PRODUCTS[
        product
    ]

    cmap = (
        LinearSegmentedColormap.from_list(
            f"{product}_radar",
            config[
                "colors"
            ],
            N=256
        )
    )

    cmap.set_bad(
        (
            0,
            0,
            0,
            0
        )
    )

    return cmap


# ============================================================
# DISPLAY DOWNSAMPLING
#
# Full arrays remain available to query_gate().
# This affects only the displayed raster.
# ============================================================

def downsample_for_display(
    azimuths,
    ranges_m,
    values
):

    ray_step = max(
        1,
        int(
            DISPLAY_RAY_STEP
        )
    )

    gate_step = max(
        1,
        int(
            DISPLAY_GATE_STEP
        )
    )

    return (

        azimuths[
            ::ray_step
        ],

        ranges_m[
            ::gate_step
        ],

        values[
            ::ray_step,
            ::gate_step
        ]

    )


# ============================================================
# GEOMETRY CACHE
#
# Radar geometry is expensive but is very similar from frame
# to frame.
#
# Cache key includes enough structural information to avoid
# reusing incompatible grids.
# ============================================================

def get_cached_geometry(
    site,
    sweep_name,
    actual_elevation,
    radar_lat,
    radar_lon,
    azimuths,
    ranges_m,
    max_range_km
):

    geometry_key = (

        site,

        sweep_name,

        round(
            float(
                actual_elevation
            ),
            3
        ),

        len(
            azimuths
        ),

        len(
            ranges_m
        ),

        round(
            float(
                max_range_km
            ),
            1
        ),

        int(
            DISPLAY_RAY_STEP
        ),

        int(
            DISPLAY_GATE_STEP
        )

    )

    if geometry_key in GEOMETRY_CACHE:

        geometry = (
            GEOMETRY_CACHE.pop(
                geometry_key
            )
        )

        GEOMETRY_CACHE[
            geometry_key
        ] = geometry

        return geometry

    # --------------------------------------------------------
    # AZIMUTH EDGES
    # --------------------------------------------------------

    azimuth_unwrapped = np.unwrap(
        np.deg2rad(
            azimuths
        )
    )

    azimuth_edges = np.rad2deg(
        coordinate_edges(
            azimuth_unwrapped
        )
    )

    # --------------------------------------------------------
    # RANGE EDGES
    # --------------------------------------------------------

    range_edges = coordinate_edges(
        ranges_m
    )

    range_edges = np.maximum(
        range_edges,
        0.0
    )

    ground_edges = (
        slant_to_ground_distance(
            range_edges,
            actual_elevation
        )
    )

    # --------------------------------------------------------
    # LAT/LON
    # --------------------------------------------------------

    lon_edges, lat_edges = (
        destination_grid(
            radar_lat,
            radar_lon,
            azimuth_edges,
            ground_edges
        )
    )

    # --------------------------------------------------------
    # WEB MERCATOR
    # --------------------------------------------------------

    x_edges, y_edges = (
        lonlat_to_mercator(
            lon_edges,
            lat_edges
        )
    )

    finite_xy = (
        np.isfinite(
            x_edges
        )
        &
        np.isfinite(
            y_edges
        )
    )

    if not np.any(
        finite_xy
    ):

        raise RuntimeError(
            "Unable to calculate radar geometry."
        )

    xmin = float(
        np.nanmin(
            x_edges[
                finite_xy
            ]
        )
    )

    xmax = float(
        np.nanmax(
            x_edges[
                finite_xy
            ]
        )
    )

    ymin = float(
        np.nanmin(
            y_edges[
                finite_xy
            ]
        )
    )

    ymax = float(
        np.nanmax(
            y_edges[
                finite_xy
            ]
        )
    )

    if (
        not np.isfinite(
            xmin
        )
        or
        not np.isfinite(
            xmax
        )
        or
        not np.isfinite(
            ymin
        )
        or
        not np.isfinite(
            ymax
        )
        or
        xmax <= xmin
        or
        ymax <= ymin
    ):

        raise RuntimeError(
            "Invalid radar geometry bounds."
        )

    # --------------------------------------------------------
    # MAPBOX IMAGE CORNERS
    # --------------------------------------------------------

    west, north = (
        mercator_to_lonlat(
            xmin,
            ymax
        )
    )

    east, north2 = (
        mercator_to_lonlat(
            xmax,
            ymax
        )
    )

    east2, south = (
        mercator_to_lonlat(
            xmax,
            ymin
        )
    )

    west2, south2 = (
        mercator_to_lonlat(
            xmin,
            ymin
        )
    )

    coordinates = [

        [
            float(
                west
            ),
            float(
                north
            )
        ],

        [
            float(
                east
            ),
            float(
                north2
            )
        ],

        [
            float(
                east2
            ),
            float(
                south
            )
        ],

        [
            float(
                west2
            ),
            float(
                south2
            )
        ]

    ]

    geometry = {

        "x_edges":
            x_edges,

        "y_edges":
            y_edges,

        "xmin":
            xmin,

        "xmax":
            xmax,

        "ymin":
            ymin,

        "ymax":
            ymax,

        "coordinates":
            coordinates

    }

    GEOMETRY_CACHE[
        geometry_key
    ] = geometry

    while (
        len(
            GEOMETRY_CACHE
        )
        >
        MAX_GEOMETRY_CACHE
    ):

        GEOMETRY_CACHE.popitem(
            last=False
        )

    return geometry


# ============================================================
# RENDER RADAR
# ============================================================

def render_radar_raster(
    radar,
    metadata,
    site,
    product,
    requested_elevation,
    max_range_km=DEFAULT_MAX_RANGE_KM
):

    start_time = time.perf_counter()

    site = validate_site(
        site
    )

    product = validate_product(
        product
    )

    requested_elevation = float(
        requested_elevation
    )

    max_range_km = float(
        max_range_km
    )

    max_range_km = max(
        10.0,
        min(
            max_range_km,
            MAX_ALLOWED_RANGE_KM
        )
    )

    (
        sweep_name,
        ds,
        field,
        actual_elevation
    ) = find_best_sweep(
        radar,
        product,
        requested_elevation
    )

    # --------------------------------------------------------
    # PNG CACHE
    # --------------------------------------------------------

    cache_key = (

        metadata[
            "key"
        ],

        product,

        sweep_name,

        round(
            max_range_km,
            1
        ),

        DISPLAY_WIDTH_PX,

        DISPLAY_RAY_STEP,

        DISPLAY_GATE_STEP

    )

    if cache_key in RASTER_CACHE:

        cached = (
            RASTER_CACHE.pop(
                cache_key
            )
        )

        RASTER_CACHE[
            cache_key
        ] = cached

        print(
            "Raster cache hit:",
            product,
            sweep_name
        )

        return cached

    filename = metadata[
        "key"
    ].rsplit(
        "/",
        1
    )[-1]

    print(
        "Rendering:",
        site,
        product,
        f"{actual_elevation:.3f}°",
        filename
    )

    # --------------------------------------------------------
    # AZIMUTH + RANGE
    # --------------------------------------------------------

    if "azimuth" not in ds:

        raise RuntimeError(
            "Radar sweep does not contain azimuth."
        )

    azimuths = np.asarray(
        ds[
            "azimuth"
        ].values,
        dtype=float
    )

    ranges_m = get_ranges_m(
        ds
    )

    # --------------------------------------------------------
    # PRODUCT VALUES
    # --------------------------------------------------------

    values = prepare_product_values(
        ds,
        field,
        product,
        azimuths,
        ranges_m
    )

    # --------------------------------------------------------
    # KDMX COORDINATE FIX
    # --------------------------------------------------------

    radar_lat, radar_lon = (
        get_site_coordinates(
            ds,
            site
        )
    )

    print(
        f"Radar location {site}: "
        f"{radar_lat:.4f}, "
        f"{radar_lon:.4f}"
    )

    # --------------------------------------------------------
    # VALID RAYS
    # --------------------------------------------------------

    valid_rays = np.isfinite(
        azimuths
    )

    azimuths = azimuths[
        valid_rays
    ]

    values = values[
        valid_rays,
        :
    ]

    if azimuths.size < 2:

        raise RuntimeError(
            "Not enough valid radar rays."
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    order = np.argsort(
        azimuths
    )

    azimuths = azimuths[
        order
    ]

    values = values[
        order,
        :
    ]

    # --------------------------------------------------------
    # REMOVE DUPLICATE AZIMUTHS
    # --------------------------------------------------------

    keep = np.ones(
        azimuths.size,
        dtype=bool
    )

    if azimuths.size > 1:

        keep[1:] = (
            np.diff(
                azimuths
            )
            >
            0.001
        )

    azimuths = azimuths[
        keep
    ]

    values = values[
        keep,
        :
    ]

    if azimuths.size < 2:

        raise RuntimeError(
            "Not enough unique radar rays."
        )

    # --------------------------------------------------------
    # RANGE LIMIT
    # --------------------------------------------------------

    range_mask = (
        ranges_m
        <=
        max_range_km
        *
        1000.0
    )

    if not np.any(
        range_mask
    ):

        raise RuntimeError(
            "Requested range contains no radar gates."
        )

    ranges_m = ranges_m[
        range_mask
    ]

    values = values[
        :,
        range_mask
    ]

    if ranges_m.size < 2:

        raise RuntimeError(
            "Not enough radar range gates."
        )

    native_shape = values.shape

    # --------------------------------------------------------
    # DISPLAY DOWNSAMPLE
    # --------------------------------------------------------

    (
        display_azimuths,
        display_ranges,
        display_values
    ) = downsample_for_display(
        azimuths,
        ranges_m,
        values
    )

    print(
        "Display grid:",
        f"{native_shape[0]}x{native_shape[1]}",
        "->",
        f"{display_values.shape[0]}x"
        f"{display_values.shape[1]}"
    )

    # --------------------------------------------------------
    # GEOMETRY CACHE
    # --------------------------------------------------------

    geometry = get_cached_geometry(

        site,
        sweep_name,
        actual_elevation,

        radar_lat,
        radar_lon,

        display_azimuths,
        display_ranges,

        max_range_km

    )

    x_edges = geometry[
        "x_edges"
    ]

    y_edges = geometry[
        "y_edges"
    ]

    xmin = geometry[
        "xmin"
    ]

    xmax = geometry[
        "xmax"
    ]

    ymin = geometry[
        "ymin"
    ]

    ymax = geometry[
        "ymax"
    ]

    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    width_px = DISPLAY_WIDTH_PX

    aspect = (
        ymax
        -
        ymin
    ) / (
        xmax
        -
        xmin
    )

    height_px = int(
        round(
            width_px
            *
            aspect
        )
    )

    height_px = max(
        DISPLAY_MIN_HEIGHT_PX,
        min(
            height_px,
            DISPLAY_MAX_HEIGHT_PX
        )
    )

    dpi = 100.0

    # --------------------------------------------------------
    # MATPLOTLIB RENDER
    # --------------------------------------------------------

    fig = plt.figure(

        figsize=(

            width_px
            /
            dpi,

            height_px
            /
            dpi

        ),

        dpi=dpi

    )

    fig.patch.set_alpha(
        0.0
    )

    ax = fig.add_axes(
        [
            0,
            0,
            1,
            1
        ]
    )

    ax.set_facecolor(
        (
            0,
            0,
            0,
            0
        )
    )

    config = PRODUCTS[
        product
    ]

    cmap = product_colormap(
        product
    )

    norm = Normalize(

        vmin=config[
            "vmin"
        ],

        vmax=config[
            "vmax"
        ],

        clip=True

    )

    masked_values = (
        np.ma.masked_invalid(
            display_values
        )
    )

    ax.pcolormesh(

        x_edges,
        y_edges,
        masked_values,

        cmap=cmap,
        norm=norm,

        shading="flat",

        antialiased=False,

        linewidth=0,

        rasterized=True

    )

    ax.set_xlim(
        xmin,
        xmax
    )

    ax.set_ylim(
        ymin,
        ymax
    )

    ax.set_axis_off()

    output = io.BytesIO()

    fig.savefig(

        output,

        format="png",

        dpi=dpi,

        transparent=True,

        bbox_inches=None,

        pad_inches=0

    )

    plt.close(
        fig
    )

    png_bytes = (
        output.getvalue()
    )

    output.close()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    scan_time = metadata.get(
        "scan_time"
    )

    if scan_time is None:

        scan_time = metadata.get(
            "modified"
        )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    result = {

        "png":
            png_bytes,

        "coordinates":
            geometry[
                "coordinates"
            ],

        "field":
            field,

        "sweep_name":
            sweep_name,

        "actual_elevation":
            float(
                actual_elevation
            ),

        "radar_latitude":
            float(
                radar_lat
            ),

        "radar_longitude":
            float(
                radar_lon
            ),

        "archive_key":
            metadata[
                "key"
            ],

        "time": (
            scan_time.isoformat()
            if scan_time
            else None
        ),

        "render_seconds":
            round(
                elapsed,
                3
            ),

        "native_shape":
            [
                int(
                    native_shape[0]
                ),
                int(
                    native_shape[1]
                )
            ],

        "display_shape":
            [
                int(
                    display_values.shape[0]
                ),
                int(
                    display_values.shape[1]
                )
            ]

    }

    RASTER_CACHE[
        cache_key
    ] = result

    while (
        len(
            RASTER_CACHE
        )
        >
        MAX_RENDERED_FRAMES
    ):

        RASTER_CACHE.popitem(
            last=False
        )

    print(
        f"Rendered {product} in "
        f"{elapsed:.2f} seconds"
    )

    return result


# ============================================================
# DISTANCE / BEARING
# ============================================================

def distance_and_bearing(
    radar_lat,
    radar_lon,
    target_lat,
    target_lon
):

    lat1 = math.radians(
        radar_lat
    )

    lon1 = math.radians(
        radar_lon
    )

    lat2 = math.radians(
        target_lat
    )

    lon2 = math.radians(
        target_lon
    )

    dlat = (
        lat2
        -
        lat1
    )

    dlon = (
        lon2
        -
        lon1
    )

    a = (

        math.sin(
            dlat
            /
            2.0
        ) ** 2

        +

        math.cos(
            lat1
        )
        *
        math.cos(
            lat2
        )
        *
        math.sin(
            dlon
            /
            2.0
        ) ** 2

    )

    a = min(
        1.0,
        max(
            0.0,
            a
        )
    )

    central_angle = (

        2.0
        *
        math.atan2(

            math.sqrt(
                a
            ),

            math.sqrt(
                1.0
                -
                a
            )

        )

    )

    distance_m = (
        EARTH_RADIUS_M
        *
        central_angle
    )

    y = (
        math.sin(
            dlon
        )
        *
        math.cos(
            lat2
        )
    )

    x = (

        math.cos(
            lat1
        )
        *
        math.sin(
            lat2
        )

        -

        math.sin(
            lat1
        )
        *
        math.cos(
            lat2
        )
        *
        math.cos(
            dlon
        )

    )

    bearing = (
        math.degrees(
            math.atan2(
                y,
                x
            )
        )
        +
        360.0
    ) % 360.0

    return (
        distance_m,
        bearing
    )


# ============================================================
# FULL-RESOLUTION INTERROGATION
#
# IMPORTANT:
# No DISPLAY_RAY_STEP or DISPLAY_GATE_STEP is used here.
# ============================================================

def query_gate(
    radar,
    site,
    product,
    requested_elevation,
    latitude,
    longitude
):

    site = validate_site(
        site
    )

    product = validate_product(
        product
    )

    (
        sweep_name,
        ds,
        field,
        actual_elevation
    ) = find_best_sweep(
        radar,
        product,
        requested_elevation
    )

    radar_lat, radar_lon = (
        get_site_coordinates(
            ds,
            site
        )
    )

    azimuths = np.asarray(
        ds[
            "azimuth"
        ].values,
        dtype=float
    )

    ranges_m = get_ranges_m(
        ds
    )

    values = prepare_product_values(
        ds,
        field,
        product,
        azimuths,
        ranges_m
    )

    (
        surface_distance_m,
        bearing
    ) = distance_and_bearing(

        radar_lat,
        radar_lon,

        latitude,
        longitude

    )

    azimuth_difference = np.abs(

        (
            (
                azimuths
                -
                bearing
                +
                180.0
            )
            %
            360.0
        )

        -

        180.0

    )

    safe_difference = np.where(

        np.isfinite(
            azimuth_difference
        ),

        azimuth_difference,

        np.inf

    )

    if not np.any(
        np.isfinite(
            safe_difference
        )
    ):

        raise RuntimeError(
            "No valid radar azimuths."
        )

    ray_index = int(
        np.argmin(
            safe_difference
        )
    )

    range_index = int(
        np.argmin(
            np.abs(
                ranges_m
                -
                surface_distance_m
            )
        )
    )

    value = values[
        ray_index,
        range_index
    ]

    output_value = (

        float(
            value
        )

        if np.isfinite(
            value
        )

        else None

    )

    return {

        "success":
            True,

        "site":
            site,

        "product":
            product,

        "product_name":
            PRODUCTS[
                product
            ][
                "name"
            ],

        "field":
            field,

        "value":
            output_value,

        "units":
            PRODUCTS[
                product
            ][
                "display_units"
            ],

        "elevation":
            float(
                actual_elevation
            ),

        "range_mi":
            float(
                ranges_m[
                    range_index
                ]
                /
                1609.344
            ),

        "azimuth":
            float(
                azimuths[
                    ray_index
                ]
            ),

        "requested_bearing":
            float(
                bearing
            ),

        "requested_range_mi":
            float(
                surface_distance_m
                /
                1609.344
            ),

        "sweep":
            sweep_name,

        "radar_latitude":
            float(
                radar_lat
            ),

        "radar_longitude":
            float(
                radar_lon
            )

    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "service": "nexrad-radar"
    }), 200


# ============================================================
# STATUS
# ============================================================

@app.route(
    "/api/status"
)
def status():

    return jsonify(
        {

            "success":
                True,

            "backend":
                "Fast NEXRAD Level-II Raster Backend",

            "default_radar":
                DEFAULT_RADAR,

            "frame_count":
                FRAME_COUNT,

            "display_width":
                DISPLAY_WIDTH_PX,

            "display_ray_step":
                DISPLAY_RAY_STEP,

            "display_gate_step":
                DISPLAY_GATE_STEP,

            "decoded_cache":
                len(
                    DECODED_VOLUME_CACHE
                ),

            "raster_cache":
                len(
                    RASTER_CACHE
                ),

            "geometry_cache":
                len(
                    GEOMETRY_CACHE
                ),

            "products":
                list(
                    PRODUCTS.keys()
                ),

            "known_radar_coordinates":
                list(
                    RADAR_SITE_COORDS.keys()
                )

        }
    )


# ============================================================
# PRODUCTS
# ============================================================

@app.route(
    "/api/products/<site>"
)
def products(
    site
):

    try:

        site = validate_site(
            site
        )

        radar, metadata = (
            get_radar_volume(
                site
            )
        )

        output = {}

        for product in PRODUCTS:

            elevations = (
                get_product_elevations(
                    radar,
                    product
                )
            )

            output[
                product
            ] = {

                "name":
                    PRODUCTS[
                        product
                    ][
                        "name"
                    ],

                "units":
                    PRODUCTS[
                        product
                    ][
                        "display_units"
                    ],

                "elevations": [
                    round(
                        value,
                        3
                    )
                    for value
                    in elevations
                ]

            }

        return jsonify(
            {

                "success":
                    True,

                "site":
                    site,

                "archive_key":
                    metadata[
                        "key"
                    ],

                "products":
                    output

            }
        )

    except Exception as error:

        traceback.print_exc()

        return jsonify(
            {
                "success":
                    False,

                "error":
                    str(
                        error
                    )
            }
        ), 500


# ============================================================
# ELEVATIONS
# ============================================================

@app.route(
    "/api/elevations/<site>/<product>"
)
def elevations(
    site,
    product
):

    try:

        site = validate_site(
            site
        )

        product = validate_product(
            product
        )

        radar, metadata = (
            get_radar_volume(
                site
            )
        )

        values = (
            get_product_elevations(
                radar,
                product
            )
        )

        return jsonify(
            {

                "success":
                    True,

                "site":
                    site,

                "product":
                    product,

                "archive_key":
                    metadata[
                        "key"
                    ],

                "elevations": [
                    round(
                        value,
                        3
                    )
                    for value
                    in values
                ]

            }
        )

    except Exception as error:

        traceback.print_exc()

        return jsonify(
            {
                "success":
                    False,

                "error":
                    str(
                        error
                    )
            }
        ), 500


# ============================================================
# FRAME LIST
# ============================================================

@app.route(
    "/api/frames/<site>"
)
def frames(
    site
):

    try:

        site = validate_site(
            site
        )

        force = (
            request.args.get(
                "refresh",
                "0"
            )
            ==
            "1"
        )

        requested_count = request.args.get(
            "count",
            1,
            type=int
        )

        requested_count = max(
            1,
            min(
                requested_count,
                FRAME_COUNT
            )
        )

        volumes = (
            find_latest_volumes(
                site,
                requested_count,
                force=force
            )
        )

        result = []

        for index, volume in enumerate(
            volumes
        ):

            scan_time = volume.get(
                "scan_time"
            )

            if scan_time is None:

                scan_time = volume.get(
                    "modified"
                )

            result.append(
                {

                    "index":
                        index,

                    "key":
                        volume[
                            "key"
                        ],

                    "filename":
                        volume[
                            "key"
                        ].rsplit(
                            "/",
                            1
                        )[-1],

                    "time": (
                        scan_time.isoformat()
                        if scan_time
                        else None
                    )

                }
            )

        return jsonify(
            {

                "success":
                    True,

                "site":
                    site,

                "count":
                    len(
                        result
                    ),

                "frames":
                    result

            }
        )

    except Exception as error:

        traceback.print_exc()

        return jsonify(
            {

                "success":
                    False,

                "error":
                    str(
                        error
                    )

            }
        ), 500


# ============================================================
# LATEST RASTER METADATA
# ============================================================

@app.route(
    "/api/radar-raster/"
    "<site>/<product>/<elevation>"
)
def radar_raster(
    site,
    product,
    elevation
):

    try:

        site = validate_site(
            site
        )

        product = validate_product(
            product
        )

        requested_elevation = float(
            elevation
        )

        max_range_km = request.args.get(
            "max_range_km",
            DEFAULT_MAX_RANGE_KM,
            type=float
        )

        radar, metadata = (
            get_radar_volume(
                site
            )
        )

        raster = (
            render_radar_raster(

                radar,
                metadata,

                site,
                product,

                requested_elevation,

                max_range_km

            )
        )

        query_string = urlencode(
            {

                "max_range_km":
                    max_range_km,

                "scan":
                    metadata[
                        "key"
                    ].rsplit(
                        "/",
                        1
                    )[-1]

            }
        )

        image_url = (

            f"/api/radar-image/"
            f"{site}/"
            f"{product}/"
            f"{requested_elevation}.png"
            f"?{query_string}"

        )

        return jsonify(
            {

                "success":
                    True,

                "site":
                    site,

                "product":
                    product,

                "field":
                    raster[
                        "field"
                    ],

                "actual_elevation":
                    raster[
                        "actual_elevation"
                    ],

                "coordinates":
                    raster[
                        "coordinates"
                    ],

                "image_url":
                    image_url,

                "archive_key":
                    metadata[
                        "key"
                    ],

                "time":
                    raster[
                        "time"
                    ],

                "render_seconds":
                    raster[
                        "render_seconds"
                    ],

                "native_shape":
                    raster[
                        "native_shape"
                    ],

                "display_shape":
                    raster[
                        "display_shape"
                    ],

                "radar_latitude":
                    raster[
                        "radar_latitude"
                    ],

                "radar_longitude":
                    raster[
                        "radar_longitude"
                    ]

            }
        )

    except Exception as error:

        traceback.print_exc()

        return jsonify(
            {

                "success":
                    False,

                "error":
                    str(
                        error
                    )

            }
        ), 500


# ============================================================
# LATEST PNG
# ============================================================

@app.route(
    "/api/radar-image/"
    "<site>/<product>/<elevation>.png"
)
def radar_image(
    site,
    product,
    elevation
):

    try:

        site = validate_site(
            site
        )

        product = validate_product(
            product
        )

        requested_elevation = float(
            elevation
        )

        max_range_km = request.args.get(
            "max_range_km",
            DEFAULT_MAX_RANGE_KM,
            type=float
        )

        radar, metadata = (
            get_radar_volume(
                site
            )
        )

        raster = (
            render_radar_raster(

                radar,
                metadata,

                site,
                product,

                requested_elevation,

                max_range_km

            )
        )

        return send_file(

            io.BytesIO(
                raster[
                    "png"
                ]
            ),

            mimetype="image/png",

            max_age=0

        )

    except Exception as error:

        traceback.print_exc()

        return jsonify(
            {

                "success":
                    False,

                "error":
                    str(
                        error
                    )

            }
        ), 500


# ============================================================
# HISTORICAL FRAME METADATA
# ============================================================

@app.route(
    "/api/frame-raster/"
    "<site>/<product>/<elevation>"
)
def frame_raster(
    site,
    product,
    elevation
):

    try:

        site = validate_site(
            site
        )

        product = validate_product(
            product
        )

        requested_elevation = float(
            elevation
        )

        archive_key = request.args.get(
            "key",
            type=str
        )

        if not archive_key:

            return jsonify(
                {

                    "success":
                        False,

                    "error":
                        "Missing radar frame archive key."

                }
            ), 400

        max_range_km = request.args.get(
            "max_range_km",
            DEFAULT_MAX_RANGE_KM,
            type=float
        )

        metadata = (
            get_frame_metadata(
                site,
                archive_key
            )
        )

        radar = (
            get_volume_for_frame(
                metadata
            )
        )

        raster = (
            render_radar_raster(

                radar,
                metadata,

                site,
                product,

                requested_elevation,

                max_range_km

            )
        )

        query_string = urlencode(
            {

                "key":
                    archive_key,

                "max_range_km":
                    max_range_km

            }
        )

        image_url = (

            f"/api/frame-image/"
            f"{site}/"
            f"{product}/"
            f"{requested_elevation}.png"
            f"?{query_string}"

        )

        return jsonify(
            {

                "success":
                    True,

                "site":
                    site,

                "product":
                    product,

                "field":
                    raster[
                        "field"
                    ],

                "actual_elevation":
                    raster[
                        "actual_elevation"
                    ],

                "coordinates":
                    raster[
                        "coordinates"
                    ],

                "image_url":
                    image_url,

                "archive_key":
                    archive_key,

                "time":
                    raster[
                        "time"
                    ],

                "render_seconds":
                    raster[
                        "render_seconds"
                    ],

                "native_shape":
                    raster[
                        "native_shape"
                    ],

                "display_shape":
                    raster[
                        "display_shape"
                    ],

                "radar_latitude":
                    raster[
                        "radar_latitude"
                    ],

                "radar_longitude":
                    raster[
                        "radar_longitude"
                    ]

            }
        )

    except Exception as error:

        traceback.print_exc()

        return jsonify(
            {

                "success":
                    False,

                "error":
                    str(
                        error
                    )

            }
        ), 500


# ============================================================
# HISTORICAL FRAME PNG
# ============================================================

@app.route(
    "/api/frame-image/"
    "<site>/<product>/<elevation>.png"
)
def frame_image(
    site,
    product,
    elevation
):

    try:

        site = validate_site(
            site
        )

        product = validate_product(
            product
        )

        requested_elevation = float(
            elevation
        )

        archive_key = request.args.get(
            "key",
            type=str
        )

        if not archive_key:

            return jsonify(
                {

                    "success":
                        False,

                    "error":
                        "Missing radar frame archive key."

                }
            ), 400

        max_range_km = request.args.get(
            "max_range_km",
            DEFAULT_MAX_RANGE_KM,
            type=float
        )

        metadata = (
            get_frame_metadata(
                site,
                archive_key
            )
        )

        radar = (
            get_volume_for_frame(
                metadata
            )
        )

        raster = (
            render_radar_raster(

                radar,
                metadata,

                site,
                product,

                requested_elevation,

                max_range_km

            )
        )

        return send_file(

            io.BytesIO(
                raster[
                    "png"
                ]
            ),

            mimetype="image/png",

            max_age=300

        )

    except Exception as error:

        traceback.print_exc()

        return jsonify(
            {

                "success":
                    False,

                "error":
                    str(
                        error
                    )

            }
        ), 500


# ============================================================
# INTERROGATION
# ============================================================

@app.route(
    "/api/query/"
    "<site>/<product>/<elevation>"
)
def query(
    site,
    product,
    elevation
):

    try:

        site = validate_site(
            site
        )

        product = validate_product(
            product
        )

        requested_elevation = float(
            elevation
        )

        latitude = request.args.get(
            "lat",
            type=float
        )

        longitude = request.args.get(
            "lon",
            type=float
        )

        if (
            latitude is None
            or
            longitude is None
        ):

            return jsonify(
                {

                    "success":
                        False,

                    "error":
                        "Both lat and lon are required."

                }
            ), 400

        archive_key = request.args.get(
            "key",
            type=str
        )

        if archive_key:

            metadata = (
                get_frame_metadata(
                    site,
                    archive_key
                )
            )

            radar = (
                get_volume_for_frame(
                    metadata
                )
            )

        else:

            radar, metadata = (
                get_radar_volume(
                    site
                )
            )

        result = query_gate(

            radar,

            site,

            product,

            requested_elevation,

            latitude,

            longitude

        )

        result[
            "archive_key"
        ] = metadata[
            "key"
        ]

        scan_time = metadata.get(
            "scan_time"
        )

        if scan_time is None:

            scan_time = metadata.get(
                "modified"
            )

        result[
            "time"
        ] = (
            scan_time.isoformat()
            if scan_time
            else None
        )

        return jsonify(
            result
        )

    except Exception as error:

        traceback.print_exc()

        return jsonify(
            {

                "success":
                    False,

                "error":
                    str(
                        error
                    )

            }
        ), 500


# ============================================================
# FORCE LIVE REFRESH
# ============================================================

@app.route(
    "/api/refresh/<site>"
)
def refresh(
    site
):

    try:

        site = validate_site(
            site
        )

        frames = (
            find_latest_volumes(
                site,
                FRAME_COUNT,
                force=True
            )
        )

        if not frames:

            raise RuntimeError(
                "No radar frames found."
            )

        latest = frames[-1]

        scan_time = latest.get(
            "scan_time"
        )

        if scan_time is None:

            scan_time = latest.get(
                "modified"
            )

        return jsonify(
            {

                "success":
                    True,

                "site":
                    site,

                "count":
                    len(
                        frames
                    ),

                "latest_key":
                    latest[
                        "key"
                    ],

                "latest_time": (
                    scan_time.isoformat()
                    if scan_time
                    else None
                )

            }
        )

    except Exception as error:

        traceback.print_exc()

        return jsonify(
            {

                "success":
                    False,

                "error":
                    str(
                        error
                    )

            }
        ), 500


# ============================================================
# CACHE INFORMATION
# ============================================================

@app.route(
    "/api/cache-status"
)
def cache_status():

    return jsonify(
        {

            "success":
                True,

            "decoded_volumes":
                len(
                    DECODED_VOLUME_CACHE
                ),

            "decoded_limit":
                MAX_DECODED_VOLUMES,

            "rendered_rasters":
                len(
                    RASTER_CACHE
                ),

            "raster_limit":
                MAX_RENDERED_FRAMES,

            "geometry_entries":
                len(
                    GEOMETRY_CACHE
                ),

            "geometry_limit":
                MAX_GEOMETRY_CACHE

        }
    )


# ============================================================
# 404
# ============================================================

@app.errorhandler(404)
def not_found(
    error
):

    return jsonify(
        {

            "success":
                False,

            "error":
                "Endpoint not found."

        }
    ), 404


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=================================================="
    )
    print(
        " FAST NEXRAD LEVEL-II RADAR SERVER"
    )
    print(
        "=================================================="
    )
    print(
        f" Radar:              {DEFAULT_RADAR}"
    )
    print(
        f" Animation frames:   {FRAME_COUNT}"
    )
    print(
        f" Decoded cache:      {MAX_DECODED_VOLUMES}"
    )
    print(
        f" Raster cache:       {MAX_RENDERED_FRAMES}"
    )
    print(
        f" Geometry cache:     {MAX_GEOMETRY_CACHE}"
    )
    print(
        f" Raster width:       {DISPLAY_WIDTH_PX}px"
    )
    print(
        f" Ray downsample:     {DISPLAY_RAY_STEP}x"
    )
    print(
        f" Gate downsample:    {DISPLAY_GATE_STEP}x"
    )
    print(
        " Interrogation:      NATIVE RESOLUTION"
    )
    print(
        " Products:           REF VEL SW ZDR CC KDP"
    )
    print(
        f" KDMX coordinates:   "
        f"{RADAR_SITE_COORDS['KDMX'][0]}, "
        f"{RADAR_SITE_COORDS['KDMX'][1]}"
    )
    print(
        "=================================================="
    )
    print(
        " http://127.0.0.1:5000"
    )
    print(
        "=================================================="
    )
    print()

    port = int(
        os.environ.get(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )