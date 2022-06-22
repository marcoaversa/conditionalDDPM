from pathlib import Path

import tifffile

_read_only_tags = [
    "ImageWidth",
    "ImageLength",
    "BitsPerSample",
    "Compression",
    "PhotometricInterpretation",
    "StripOffsets",
    "SamplesPerPixel",
    "StripByteCounts",
    "NewSubfileType",
    "ExifTag",  # Actually a SubIFD without image data
    "GPSTag",  # Same
    "SubIFDs",
    "RowsPerStrip",
    "Software",
    "TileOffsets",
    "TileByteCounts",
    "DefaultCropOrigin",  # This and the two below must be compatible with
    "DefaultCropSize",  # the actual image dimensions.
    "ActiveArea",
]


def is_cfa_ifd(ifd: tifffile.TiffPage):
    return (
        ifd.tags["NewSubfileType"].value == tifffile.TIFF.FILETYPE.UNDEFINED
        and ifd.tags["PhotometricInterpretation"].value
        == tifffile.TIFF.PHOTOMETRIC.CFA
    )


def get_read_write_tags(ifd: tifffile.TiffPage):
    tags = list([t for t in ifd.tags if t.name not in _read_only_tags])
    return tags


def dng_from_template(preview_data, data, dng_in_path, dng_out_path):
    with tifffile.TiffReader(dng_in_path) as fh:
        main_ifd = fh.pages[0]
        main_tags = get_read_write_tags(main_ifd)
        has_cfa_ifd = is_cfa_ifd(main_ifd)

        if not has_cfa_ifd and main_ifd.subifds is not None:
            for subifd_num in range(len(main_ifd.subifds)):
                ifd = main_ifd.pages.get(subifd_num)
                if is_cfa_ifd(ifd):
                    cfa_tags = get_read_write_tags(ifd)
                    has_cfa_ifd = True
                    break

    if not has_cfa_ifd:
        raise RuntimeError("No CFA data found in IFD0 or its SubIFDs.")

    main_tag_tuples = list(
        [(t.code, t.dtype, t.count, t.value, True) for t in main_tags]
    )
    cfa_tag_tuples = list(
        [(t.code, t.dtype, t.count, t.value, True) for t in cfa_tags]
    )

    cfa_tag_tuples += [
        (254, 4, 1, 0, True),  # NewSubfileType
    ]

    with tifffile.TiffWriter(dng_out_path) as fh:
        fh.write(
            data=preview_data,
            photometric=tifffile.TIFF.PHOTOMETRIC.MINISBLACK,
            subfiletype=tifffile.TIFF.FILETYPE.REDUCEDIMAGE,
            extratags=main_tag_tuples,
            subifds=1,
        )
        fh.write(
            data=data,
            photometric=tifffile.TIFF.PHOTOMETRIC.CFA,
            extratags=cfa_tag_tuples,
        )


def parse_file(dngpath: Path):
    print(f"Parsing {dngpath.name}...")

    with tifffile.TiffReader(dngpath) as fh:
        for i, ifd in enumerate(fh.pages):
            print(f"{_prefix(0)}IFD {i}")
            print_ifd(0, ifd)


def _prefix(level):
    return "  " * level


def print_ifd(level, ifd):
    print(f"{_prefix(level + 1)}Image data: {ifd.shape}, {ifd.dtype}")
    for t in ifd.tags:
        print(f"{_prefix(level + 1)}{t}")

    if ifd.subifds is None:
        return
    for j in range(len(ifd.subifds)):
        print(f"{_prefix(level + 1)}SUBIFD {j}")
        print_ifd(level + 1, ifd.pages.get(j))


def main():
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description="Parse structure of DNG file")
    parser.add_argument("dng_file", type=str)

    args = parser.parse_args()
    dngpath = Path(args.dng_file)
    if not dngpath.is_file():
        raise RuntimeError(f"File {dngpath.name} does not exist.")

    im = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    dng_from_template(
        im[::2, ::2], im, dngpath, dngpath.with_suffix(".out.dng")
    )


if __name__ == "__main__":
    main()