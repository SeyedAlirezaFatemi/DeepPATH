"""
File name: 0b_tileLoop_deepzoom.py
Date created: March/2017

Source:
    Tiling code inspired from
    https://github.com/openslide/openslide-python/blob/master/examples/deepzoom/deepzoom_tile.py
    which is Copyright (c) 2010-2015 Carnegie Mellon University
    The code has been extensively modified

Objective:
    Tile svs, jpg or dcm images with the possibility of rejecting some tiles based based on xml or jpg masks

Be careful:
    Overload of the node - may have memory issue if node is shared with other jobs.

The overall process is:
    1. Tile the svs images and convert into jpg
    2. Sort the jpg images into train/valid/test at a given magnification and put them in appropriate classes
    3. Convert each of the sets into TFRecord format
"""

from __future__ import print_function

import argparse
import json
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
import re
import shutil
from unicodedata import normalize
import numpy as np
import scipy.misc
from glob import glob
from multiprocessing import Process, JoinableQueue
import os
import sys
import pydicom
import imageio
from imageio import imread
from imageio import imwrite
from scipy.misc import imresize

from xml.dom import minidom
from PIL import Image, ImageDraw

VIEWER_SLIDE_NAME = 'slide'


class TileWorker(Process):
    """
    A child process that generates and writes tiles.
    """

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds, quality, bkg_threshold, roi_pc):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self.queue = queue
        self.slidepath = slidepath
        self.tile_size = tile_size
        self.overlap = overlap
        self.limit_bounds = limit_bounds
        self.quality = quality
        self.slide = None
        self.bkg_threshold = bkg_threshold
        self.roi_pc = roi_pc

    def run(self):
        self.slide = open_slide(self.slidepath)
        last_associated = None
        deep_zoom = self.get_dz()
        while True:
            data = self.queue.get()
            if data is None:
                self.queue.task_done()
                break
            associated, level, address, outfile, format, outfile_bw, percent_masked, save_masks, tile_mask = data
            if last_associated != associated:
                deep_zoom = self.get_dz(associated)
                last_associated = associated
            try:
                # Generate tile for the specified level and address.
                tile = deep_zoom.get_tile(level, address)
                # Check the percentage of the image with "information". Should be above 50%.
                # Convert tile to greyscale.
                gray = tile.convert('L')
                # Black and white with threshold = 220.
                bw = gray.point(lambda x: 0 if x < 220 else 1, 'F')
                # Convert black and white image to numpy array.
                arr = np.array(np.asarray(bw))
                avg_bkg = np.average(bw)
                # Check if the image is mostly background.
                if avg_bkg <= (self.bkg_threshold / 100.0):
                    print(f"Percent_masked: {percent_masked:.6f}, {self.roi_pc / 100.0:.6f}")
                    # If an Aperio selection was made, check if it is within the selected region.
                    # If no xml is provided, percent_masked is 1
                    if percent_masked >= (self.roi_pc / 100.0):
                        tile.save(outfile, quality=self.quality)
                        if bool(save_masks):
                            height = tile_mask.shape[0]
                            width = tile_mask.shape[1]
                            tile_mask_o = np.zeros((height, width, 3), 'uint8')
                            max_val = float(tile_mask.max())
                            tile_mask_o[..., 0] = (tile_mask[:, :].astype(float) / max_val * 255.0).astype(int)
                            tile_mask_o[..., 1] = (tile_mask[:, :].astype(float) / max_val * 255.0).astype(int)
                            tile_mask_o[..., 2] = (tile_mask[:, :].astype(float) / max_val * 255.0).astype(int)
                            tile_mask_o = imresize(tile_mask_o, (arr.shape[0], arr.shape[1], 3))
                            tile_mask_o[tile_mask_o < 10] = 0
                            tile_mask_o[tile_mask_o >= 10] = 255
                            imwrite(outfile_bw, tile_mask_o)
                else:
                    print(f"Ignored tile with level : {level} , address : {address} , average background : {avg_bkg}.")
            except:
                print(level, address)
                print(f"Image {self.slidepath} failed at dz.get_tile for level {level:f}")
            self.queue.task_done()

    def get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self.slide.associated_images[associated])
        else:
            image = self.slide
        return DeepZoomGenerator(image, self.tile_size, self.overlap, limit_bounds=self.limit_bounds)


class DeepZoomImageTiler:
    """
    Handles generation of tiles and metadata for a single image.
    """

    def __init__(self, dz, basename, format, associated, queue, slide, basename_jpg, xml_file, mask_type, xml_label,
                 roi_pc, img_extension, save_masks, mag):
        self.dz = dz
        self.basename = basename
        self.basename_jpg = basename_jpg
        self.format = format
        self.associated = associated
        self.queue = queue
        self.processed = 0
        self.slide = slide
        self.xml_file = xml_file
        self.mask_type = mask_type
        self.xml_label = xml_label
        self.roi_pc = roi_pc
        self.img_extension = img_extension
        self.save_masks = save_masks
        self.magnification = mag

    def run(self):
        self.write_tiles()
        self.write_dzi()

    def write_tiles(self):
        # Get slide dimensions, zoom levels, and objective information
        factors = self.slide.level_downsamples
        try:
            objective = float(self.slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            print(f"{self.basename} - Obj information found - {objective}")
        except:
            print(f"{self.basename} - No Obj information found")
            if ("jpg" in self.img_extension) | ("dcm" in self.img_extension):
                objective = 1.
                print(f"Input is jpg - will be tiled as such with {objective}")
            else:
                return
        # Calculate magnifications
        available = tuple(objective / x for x in factors)
        if len(available) < 1:
            print(self.basename + " - Objective field empty!")
            return

        xml_valid = False
        # A dir was provided for xml files
        print(self.xml_file, self.img_extension)
        img_id = os.path.basename(self.basename)
        xml_dir = os.path.join(self.xml_file, img_id + '.xml')
        print(f"xml: {xml_dir}")
        if (self.xml_file != '') & (self.img_extension != 'jpg') & (self.img_extension != 'dcm'):
            print("Read xml file...")
            mask, xml_valid, img_fact = self.xml_read(xml_dir, self.xml_label)
            if xml_valid == False:
                print("Error: xml {} file cannot be read properly - please check format".format(xml_dir))
                return
        elif (self.xml_file != '') & (self.img_extension == 'dcm'):
            print("Check mask for dcm")
            mask, xml_valid, img_fact = self.jpg_mask_read(xml_dir)
            if not xml_valid:
                print(f"Error: xml {xml_dir} file cannot be read properly - please check format")
                return

        print(f'Current directory: {self.basename}')

        for level in range(self.dz.level_count - 1, -1, -1):
            current_mag = available[0] / pow(2, self.dz.level_count - (level + 1))
            # If self.magnification is -1, tile at all magnification levels.
            if 0 < self.magnification != current_mag:
                continue
            tile_dir = os.path.join(f"{self.basename}_files", str(current_mag))
            if not os.path.exists(tile_dir):
                os.makedirs(tile_dir)
            cols, rows = self.dz.level_tiles[level]
            if xml_valid:
                print("xml valid")
            for row in range(rows):
                for col in range(cols):
                    tile_name = os.path.join(tile_dir, f'{col}_{row}.{self.format}')
                    tile_name_bw = os.path.join(tile_dir, f'{col}_{row}_mask.{self.format}')
                    if xml_valid:
                        dlocation, Dlevel, Dsize = self.dz.get_tile_coordinates(level, (col, row))
                        Ddimension = tuple([pow(2, (self.dz.level_count - 1 - level)) * x for x in
                                            self.dz.get_tile_dimensions(level, (col, row))])
                        startIndY_current_level_conv = (int((dlocation[1]) / img_fact))
                        endIndY_current_level_conv = (int((dlocation[1] + Ddimension[1]) / img_fact))
                        startIndX_current_level_conv = (int((dlocation[0]) / img_fact))
                        endIndX_current_level_conv = (int((dlocation[0] + Ddimension[0]) / img_fact))
                        tile_mask = mask[startIndY_current_level_conv:endIndY_current_level_conv,
                                    startIndX_current_level_conv:endIndX_current_level_conv]
                        percent_masked = mask[startIndY_current_level_conv:endIndY_current_level_conv,
                                         startIndX_current_level_conv:endIndX_current_level_conv].mean()

                        print(Ddimension, startIndY_current_level_conv, endIndY_current_level_conv,
                              startIndX_current_level_conv, endIndX_current_level_conv)

                        if self.mask_type == 0:
                            # Keep ROI outside of the mask
                            percent_masked = 1.0 - percent_masked

                        if percent_masked > 0:
                            print(f"PercentMasked_p {percent_masked:.3f}")
                        else:
                            print(f"PercentMasked_0 {percent_masked:.3f}")
                    else:
                        percent_masked = 1.0
                        tile_mask = []

                    # Check if image has been tiled before.
                    if not os.path.exists(tile_name):
                        # Workers will do the tiling.
                        self.queue.put((self.associated, level, (col, row), tile_name, self.format, tile_name_bw,
                                        percent_masked, self.save_masks, tile_mask))
                    self.tile_done()

    def tile_done(self):
        self.processed += 1
        count, total = self.processed, self.dz.tile_count
        if count % 100 == 0 or count == total:
            print(f"Tiling {self.associated or 'slide'}: wrote {count:d}/{total:d} tiles",
                  end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

    def write_dzi(self):
        with open(f'{self.basename}.dzi', 'w') as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        return self.dz.get_dzi(self.format)

    def jpg_mask_read(self, xml_dir):
        img_fact = 1
        try:
            # xml_dir: change extension from xml to *jpg
            xml_dir = xml_dir[:-4] + "mask.jpg"
            # xml_content = read xml_dir image
            xml_content = imread(xml_dir)
            xml_content = xml_content - np.min(xml_content)
            mask = xml_content / np.max(xml_content)
            # we want image between 0 and 1
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xml_dir)")
            return [], xml_valid, 1.0

        return mask, xml_valid, img_fact

    def xml_read(self, xml_dir, attribute_name):
        # Original size of the image
        ImgMaxSizeX_orig = float(self.dz.level_dimensions[-1][0])
        ImgMaxSizeY_orig = float(self.dz.level_dimensions[-1][1])
        # Number of centers at the highest resolution
        cols, rows = self.dz.level_tiles[-1]

        new_fact = max(ImgMaxSizeX_orig, ImgMaxSizeY_orig) / min(max(ImgMaxSizeX_orig, ImgMaxSizeY_orig), 15000.0)

        print("Image info:")
        print(ImgMaxSizeX_orig, ImgMaxSizeY_orig, cols, rows)
        try:
            xmlcontent = minidom.parse(xml_dir)
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid, 1.0

        xy = {}
        xy_neg = {}
        labelIDs = xmlcontent.getElementsByTagName('Annotation')
        for labelID in labelIDs:
            if (attribute_name == []) | (attribute_name == ''):
                isLabelOK = True
            else:
                try:
                    labeltag = labelID.getElementsByTagName('Attribute')[0]
                    if (attribute_name == labeltag.attributes['Value'].value):
                        # if (Attribute_Name==labeltag.attributes['Name'].value):
                        isLabelOK = True
                    else:
                        isLabelOK = False
                except:
                    isLabelOK = False
            if attribute_name == "non_selected_regions":
                isLabelOK = True

            if isLabelOK:
                regionlist = labelID.getElementsByTagName('Region')
                for region in regionlist:
                    vertices = region.getElementsByTagName('Vertex')
                    regionID = region.attributes['Id'].value
                    NegativeROA = region.attributes['NegativeROA'].value
                    if len(vertices) > 0:
                        # print( len(vertices) )
                        if NegativeROA == "0":
                            xy[regionID] = []
                            for vertex in vertices:
                                x = int(round(float(vertex.attributes['X'].value) / new_fact))
                                y = int(round(float(vertex.attributes['Y'].value) / new_fact))
                                xy[regionID].append((x, y))

                        elif NegativeROA == "1":
                            xy_neg[regionID] = []
                            for vertex in vertices:
                                x = int(round(float(vertex.attributes['X'].value) / new_fact))
                                y = int(round(float(vertex.attributes['Y'].value) / new_fact))
                                xy_neg[regionID].append((x, y))

        print(f"img_fact: {new_fact}")

        img = Image.new('L', (int(ImgMaxSizeX_orig / new_fact), int(ImgMaxSizeY_orig / new_fact)), 0)
        for regionID in xy.keys():
            xy_a = xy[regionID]
            ImageDraw.Draw(img, 'L').polygon(xy_a, outline=255, fill=255)
        for regionID in xy_neg.keys():
            xy_a = xy_neg[regionID]
            ImageDraw.Draw(img, 'L').polygon(xy_a, outline=255, fill=0)
        mask = np.array(img)
        scipy.misc.toimage(mask).save(os.path.join(os.path.split(self.basename[:-1])[0], "mask_" + os.path.basename(
            self.basename) + "_" + attribute_name + ".jpeg"))
        return mask / 255.0, xml_valid, new_fact


class DeepZoomStaticTiler:
    """
    Handles generation of tiles and metadata for all images in a slide.
    Runs a DeepZoomImageTiler for each image in a slide.
    """

    def __init__(self, slidepath, basename, format, tile_size, overlap,
                 limit_bounds, quality, workers, with_viewer, bkg_threshold, basename_jpg, xml_file, mask_type,
                 roi_percentage, o_label, img_extension, save_masks, magnification):
        if with_viewer:
            # Check extra dependency before doing a bunch of work
            import jinja2

        self.slide = open_slide(slidepath)
        self.basename = basename
        self.basename_jpg = basename_jpg
        self.xml_file = xml_file
        self.mask_type = mask_type
        self.format = format
        self.tile_size = tile_size
        self.overlap = overlap
        self.limit_bounds = limit_bounds
        self.queue = JoinableQueue(2 * workers)
        self.workers = workers
        self.with_viewer = with_viewer
        self.bkg_threshold = bkg_threshold
        self.roi_percentage = roi_percentage
        self.xml_label = o_label
        self.img_extension = img_extension
        self.save_masks = save_masks
        self.magnification = magnification
        self.dzi_data = {}

        for _ in range(workers):
            TileWorker(self.queue, slidepath, tile_size, overlap, limit_bounds, quality, self.bkg_threshold,
                       self.roi_percentage).start()

    def run(self):
        self.run_image()
        if self.with_viewer:
            for name in self.slide.associated_images:
                self.run_image(name)
            self.write_html()
            self.write_static()
        self.shutdown()

    def run_image(self, associated=None):
        """
        Run a single image from self.slide.
        """
        if associated is None:
            image = self.slide
            if self.with_viewer:
                basename = os.path.join(self.basename, VIEWER_SLIDE_NAME)
            else:
                basename = self.basename
        else:
            image = ImageSlide(self.slide.associated_images[associated])
            basename = os.path.join(self.basename, self.slugify(associated))
        dz = DeepZoomGenerator(image, self.tile_size, self.overlap, limit_bounds=self.limit_bounds)
        tiler = DeepZoomImageTiler(dz, basename, self.format, associated, self.queue, self.slide,
                                   self.basename_jpg, self.xml_file, self.mask_type, self.xml_label,
                                   self.roi_percentage, self.img_extension, self.save_masks, self.magnification)
        tiler.run()
        self.dzi_data[self.url_for(associated)] = tiler.get_dzi()

    def url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self.slugify(associated)
        return '%s.dzi' % base

    def write_html(self):
        import jinja2
        env = jinja2.Environment(loader=jinja2.PackageLoader(__name__), autoescape=True)
        template = env.get_template('slide-multipane.html')
        associated_urls = dict((n, self.url_for(n))
                               for n in self.slide.associated_images)
        try:
            mpp_x = self.slide.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = self.slide.properties[openslide.PROPERTY_NAME_MPP_Y]
            mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            mpp = 0
        # Embed the dzi metadata in the HTML to work around Chrome's
        # refusal to allow XmlHttpRequest from file:///, even when
        # the originating page is also a file:///
        data = template.render(slide_url=self.url_for(None), slide_mpp=mpp, associated=associated_urls,
                               properties=self.slide.properties, dzi_data=json.dumps(self.dzi_data))
        with open(os.path.join(self.basename, 'index.html'), 'w') as fh:
            fh.write(data)

    def write_static(self):
        basesrc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'static')
        basedst = os.path.join(self.basename, 'static')
        self.copy_dir(basesrc, basedst)
        self.copy_dir(os.path.join(basesrc, 'images'),
                      os.path.join(basedst, 'images'))

    def copy_dir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def shutdown(self):
        # Shutdown workers
        for _ in range(self.workers):
            self.queue.put(None)
        self.queue.join()


def xml_read_labels(xml_path):
    try:
        xml_content = minidom.parse(xml_path)
        xml_valid = True
    except:
        xml_valid = False
        print("Error with minidom.parse(xml_path)")
        return [], xml_valid
    label_tag = xml_content.getElementsByTagName('Attribute')
    xml_labels = []
    for xml_label in label_tag:
        xml_labels.append(xml_label.attributes['Value'].value)
    if not xml_labels:
        xml_labels = ['']
    print(xml_labels)
    return xml_labels, xml_valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='%(prog)s [options] [slidepath]')

    parser.add_argument('slides_dir', help='path to slides directory')

    # limit_bounds: True to render only the non-empty slide region
    parser.add_argument('-L', '--ignore_bounds', dest='limit_bounds',
                        default=True, action='store_false',
                        help='display entire scan area')

    # overlap
    parser.add_argument('-e', '--overlap', metavar='PIXELS', dest='overlap',
                        type=int, default=1,
                        help='overlap of adjacent tiles [1]')

    parser.add_argument('-f', '--format', metavar='{jpeg|png}', dest='format',
                        default='jpeg',
                        help='image format for tiles [jpeg]')

    # workers
    parser.add_argument('-j', '--jobs', metavar='COUNT', dest='workers', type=int, default=4,
                        help='number of worker processes to start [4]')

    # basename = The path were the output images must be saved
    parser.add_argument('-o', '--output', metavar='NAME', dest='basename',
                        help='base name of output file')

    parser.add_argument('-q', '--quality', metavar='QUALITY', dest='quality',
                        type=int, default=90,
                        help='JPEG compression quality [90]')

    # with_viewer
    parser.add_argument('-r', '--viewer', dest='with_viewer',
                        action='store_true',
                        help='generate directory tree with HTML viewer')

    # tile_size
    parser.add_argument('-s', '--tile_size', metavar='PIXELS', dest='tile_size',
                        type=int, default=254,
                        help='tile size [254]')

    # bkg_threshold
    parser.add_argument('-b', '--bkg_threshold', metavar='PIXELS', dest='bkg_threshold',
                        type=float, default=50,
                        help='Max background threshold [50]; percentage of background allowed')

    # xml_file
    parser.add_argument('-x', '--xml_file', metavar='NAME', dest='xml_file',
                        help='xml file if needed')

    # mask_type
    parser.add_argument('-m', '--mask_type', metavar='COUNT', dest='mask_type',
                        type=int, default=1,
                        help='If xml file is used, keep tile within the ROI (1) or outside of it (0)')

    # roi_pc
    parser.add_argument('-R', '--roi_pc', metavar='PIXELS', dest='roi_pc',
                        type=float, default=50,
                        help='To be used with xml file - minimum percentage of tile covered by ROI (white)')

    # o_ref_label
    parser.add_argument('-l', '--o_label_ref', metavar='NAME', dest='o_label_ref',
                        help='To be used with xml file - Only tile for label which contains the characters in oLabel')

    # save_masks
    parser.add_argument('-S', '--save_masks', metavar='NAME', dest='save_masks',
                        default=False,
                        help='set to True if you want to save ALL masks for ALL tiles (will be saved in same directory with <mask> suffix)')

    # tmp_dcm
    parser.add_argument('-t', '--tmp_dcm', metavar='NAME', dest='tmp_dcm',
                        help='base name of output folder to save intermediate dcm images converted to jpg (we assume the patient ID is the folder name in which the dcm images are originally saved)')

    # magnification
    parser.add_argument('-M', '--mag', metavar='PIXELS', dest='mag',
                        type=float, default=-1,
                        help='Magnification at which tiling should be done (-1 of all)')

    args = parser.parse_args()

    slides_dir = args.slides_dir
    if args.basename is None:
        args.basename = os.path.splitext(os.path.basename(slides_dir))[0]
    if args.xml_file is None:
        args.xml_file = ''

    # Get  images from the data/ file.
    files = glob(slides_dir)
    # If slidepath is not a pattern and points to only one file => split('.')[-1]
    # The input is a pattern. That's why we split('*').
    img_extension = slides_dir.split('*')[-1]
    print(slides_dir)
    print(files)
    print("***********************")

    files = sorted(files)
    # Iterate over all images
    # A DeepZoomStaticTiler is run for each eligible
    for img_number in range(len(files)):
        filename = files[img_number]
        # basename_jpg = Image name without extension
        args.basename_jpg = os.path.splitext(os.path.basename(filename))[0]
        print(f"Processing: {args.basename_jpg} with extension: {img_extension}")
        # We assume the patient ID is the folder name in which the dcm images are originally saved.

        if "dcm" in img_extension:
            print(f"Convert {filename} dcm to jpg")
            if args.tmp_dcm is None or not os.path.isdir(args.tmp_dcm):
                parser.error('Missing output folder for dcm>jpg intermediate files')
            # Ignore .jpg ones
            if filename[-3:] == 'jpg':
                continue
            dcm_image_file = pydicom.read_file(filename)
            dcm_raw_image = dcm_image_file.pixel_array
            maxVal = float(dcm_raw_image.max())
            minVal = float(dcm_raw_image.min())
            height = dcm_raw_image.shape[0]
            width = dcm_raw_image.shape[1]
            image = np.zeros((height, width, 3), 'uint8')
            image[..., 0] = ((dcm_raw_image[:, :].astype(float) - minVal) / (maxVal - minVal) * 255.0).astype(int)
            image[..., 1] = ((dcm_raw_image[:, :].astype(float) - minVal) / (maxVal - minVal) * 255.0).astype(int)
            image[..., 2] = ((dcm_raw_image[:, :].astype(float) - minVal) / (maxVal - minVal) * 255.0).astype(int)
            # Name of the folder the file is in
            dcm_id = os.path.basename(os.path.dirname(filename))
            args.basename_jpg = f"{dcm_id}_{args.basename_jpg}"
            filename = os.path.join(args.tmp_dcm, f"{args.basename_jpg}.jpg")
            print(filename)
            imageio.imwrite(filename, image)

            output = os.path.join(args.basename, args.basename_jpg)

            # After converting dcm to jpg, it's time to tile.
            try:
                DeepZoomStaticTiler(filename, output, args.format, args.tile_size, args.overlap, args.limit_bounds,
                                    args.quality, args.workers, args.with_viewer, args.bkg_threshold, args.basename_jpg,
                                    args.xml_file, args.mask_type, args.roi_pc, '', img_extension, args.save_masks,
                                    args.mag).run()
            except:
                print(f"Failed to process file {filename}, error: {sys.exc_info()[0]}")

        elif args.xml_file != '':
            xml_path = os.path.join(args.xml_file, f'{args.basename_jpg}.xml')
            print(f"xml: {xml_path}")
            if os.path.isfile(xml_path):
                if args.mask_type == 1:
                    # Keep tiles within ROI
                    xml_labels, xml_valid = xml_read_labels(xml_path)
                    for o_label in xml_labels:
                        print(f"Label is {o_label} and ref is {args.o_label_ref}")
                        if (args.o_label_ref in o_label) or (args.o_label_ref == ''):
                            output = os.path.join(args.basename, o_label, args.basename_jpg)
                            if not os.path.exists(os.path.join(args.basename, o_label)):
                                os.makedirs(os.path.join(args.basename, o_label))
                            try:
                                DeepZoomStaticTiler(filename, output, args.format, args.tile_size, args.overlap,
                                                    args.limit_bounds, args.quality, args.workers, args.with_viewer,
                                                    args.bkg_threshold, args.basename_jpg, args.xml_file,
                                                    args.mask_type,
                                                    args.roi_pc, o_label, img_extension, args.save_masks,
                                                    args.mag).run()
                            except:
                                print(f"Failed to process file {filename}, error: {sys.exc_info()[0]}")
                else:
                    # Keep tiles outside ROI
                    # Background
                    o_label = "non_selected_regions"
                    output = os.path.join(args.basename, o_label, args.basename_jpg)
                    if not os.path.exists(os.path.join(args.basename, o_label)):
                        os.makedirs(os.path.join(args.basename, o_label))
                    try:
                        DeepZoomStaticTiler(filename, output, args.format, args.tile_size, args.overlap,
                                            args.limit_bounds, args.quality, args.workers, args.with_viewer,
                                            args.bkg_threshold,
                                            args.basename_jpg, args.xml_file, args.mask_type, args.roi_pc, o_label,
                                            img_extension, args.save_masks, args.mag).run()
                    except:
                        print(f"Failed to process file {filename}, error: {sys.exc_info()[0]}")

            else:
                if (img_extension == "jpg") | (img_extension == "dcm"):
                    print("Input image to be tiled is jpg or dcm and not svs - will be treated as such")
                    output = os.path.join(args.basename, args.basename_jpg)
                    try:
                        DeepZoomStaticTiler(filename, output, args.format, args.tile_size, args.overlap,
                                            args.limit_bounds, args.quality, args.workers, args.with_viewer,
                                            args.bkg_threshold,
                                            args.basename_jpg, args.xml_file, args.mask_type, args.roi_pc, '',
                                            img_extension, args.save_masks, args.mag).run()
                    except:
                        print(f"Failed to process file {filename}, error: {sys.exc_info()[0]}")
                else:
                    print(
                        f"No xml file found for slide {args.basename_jpg}.svs (expected: {xml_path}). Directory or "
                        f"xml file does not exist")
                    continue
        # No xml, No dcm
        else:
            output = os.path.join(args.basename, args.basename_jpg)
            if os.path.exists(output + "_files"):
                print(f"Image {args.basename_jpg} already tiled")
                continue
            try:
                DeepZoomStaticTiler(filename, output, args.format, args.tile_size, args.overlap, args.limit_bounds,
                                    args.quality, args.workers, args.with_viewer, args.bkg_threshold, args.basename_jpg,
                                    args.xml_file, args.mask_type, args.roi_pc, '', img_extension, args.save_masks,
                                    args.mag).run()
            except:
                print(f"Failed to process file {filename}, error: {sys.exc_info()[0]}")

    print("End")
