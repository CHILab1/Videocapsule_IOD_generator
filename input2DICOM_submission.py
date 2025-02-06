from pydicom.dataset import Dataset, FileDataset
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
from sys import exit
from pydicom.uid import UID, RLELossless
import pandas as pd
import numpy as np
import datetime
import pydicom
import json
import cv2
import os
from sys import byteorder
import argparse


class Input2DICOM():
    def __init__(self, input_file=None, metadata_json=None, output_file=None, metadata_tags=None):

        self.data = None

        if not metadata_tags:

            self.metadata = {
                #!"Patient": {
                "Patient ID": ["00100020",  "LO",  "1"],
                "Patient's Name": ["00100010",  "PN",  "1"],
                "Patient's Birth Date": ["00100030",  "DA",  "1"],
                "Patient's Sex": ["00100040",  "CS",  "1"],

                #!"General Study": {
                "Study Date": ["00080020",  "DA",  "1"],
                "Study Time": ["00080030",  "TM",  "1"],
                "Referring Physician's Name": ["00080090",  "PN",  "1"],
                "Study Instance UID": ["0020000D",  "UI",  "1"],
                "Study ID": ["00200010",  "SH",  "2"],
                "Accession Number": ["00080050",  "SH",  "2"],
                
                #!"Patient Study": {
                "Admitting Diagnoses Description": ["00081080",  "LO",  "3"],
                "Patient's Age": ["00101010",  "AS",  "3"],
                "Patient's Size": ["00101020",  "DS",  "3"],
                "Patient's Weight": ["00101030",  "DS",  "3"],
                "Medical Alerts": ["00102000",  "LO",  "3"],
                "Allergies": ["00102110",  "LO",  "3"],
                "Smoking Status": ["001021A0",  "CS",  "3"],
                "Pregnacy Status": ["001021C0",  "US",  "3"],
                "Admission ID": ["00380060",  "LO",  "3"],
                "Patient State": ["00380500",  "LO",  "3"],
                # !    "General Series": {
                "Series Date": ["00080021",  "DA",  "3"],
                "Series Time": ["00080031",  "TM",  "3"],
                "Modality": ["00080060",  "CS",  "1"],
                "Anatomical Orientation Type": ["00102210",  "CS",  "1C"],
                "Body Part Examined": ["00180015",  "CS",  "3"],
                "Series Instance UID": ["0020000E",  "UI",  "1"],
                "SeriesNumber": ["00200011",  "IS",  "2"],
                "Laterality": ["00200062",  "CS",  "3"],
                # !    "General Equipment": {
                "Manufacturer": ["00080070",  "LO",  "2"],
                "InstitutionName": ["00080080",  "LO",  "3"],
                "Manufacturer's Model Name": ["00081090",  "LO",  "3"],
                "Device Serial Number": ["00181000",  "LO",  "3"],
                "Device UID": ["00181002",  "UI",  "3"],
                "Software Versions": ["00181020",  "LO",  "3"],
                "Date of Last Calibration": ["00181200",  "DA",  "3"],
                "Date of Installation": ["00181205",  "DA",  "3"],
                # !    "Cine":{
                "Cine Rate": ["00180040",  "IS",  "3"],
                "Frame Time": ["00181063",  "DS",  "1C"],
                "Frame Time Vector": ["00181065",  "DS",  "1C"],
                # !    "Multi-frame": {
                "Number of Frames": ["00280008",  "IS",  "1"],
                "Frame Increment Pointer": ["00280009",  "AT",  "1"],
                # !    "Image Pixel": {
                "Rows": ["00280010",  "US",  "1"],
                "Columns": ["00280011",  "US",  "1"],
                "Bits Allocated": ["00280100",  "US",  "1"],
                # !    "Acquisition Context": {
                "PixelSpacing": ["00280030",  "DS",  "3"],
                'Acquisition Context Sequence': ['00400555', 'SQ', '2',
                                                 {
                                                     "DateTime": ["0040A120", "DT", "1"],
                                                     "Date": ["0040A121", "DA", "1"],
                                                     "Time": ["0040A122", "TM", "1"],
                                                     "PersonName": ["0040A123", "PN", "1"],
                                                     "TextValue": ["0040A160", "UT", "1"],
                                                     "Concept Name Code Sequence": ["0040A043", "SQ", "1",
                                                                                    {
                                                                                        "Code Value": ["00080100", "SH", "1"],
                                                                                        "Coding SchemeDesignator": ["00080102", "SH", "2"],
                                                                                        "Coding SchemeVersion": ["00080103", "SH", "2"],
                                                                                        "Code Meaning": ["00080104", "LO", "2"],
                                                                                        "Mapping Resource": ["00080105", "CS", "2"],
                                                                                        "Context GroupVersion": ["00080106", "DT", "2"],
                                                                                        "Context GroupLocalVersion": ["00080107", "DT", "2"],
                                                                                        "Context GroupExtensionCreatorUID": ["0008010B", "UI", "2"],
                                                                                        "Long Code Value": ["00080119", "UC", "2"],
                                                                                        "URN Code Value": ["00080120", "UR", "2"],
                                                                                    }
                                                                                    ],
                                                 }],
                # !    "Device":{
                "Acquisition Context Description": ["00400556",  "ST",  "1"],

                #!    "VL Image": {
                "Image Type": ["00080008",  "CS",  "1"],
                "Content Time": ["00080033",  "TM",  "1C"],
                "Imager Pixel Spacing": ["00181164",  "DS",  "3"],
                "Samples per Pixel": ["00280002",  "US",  "1"],
                "Photometric Interpretation": ["00280004",  "CS",  "1"],
                "Planar Configuration": ["00280006",  "US",  "1C"],
                "Pixel Spacing": ["00280030",  "DS",  "3"],
                "Bits Allocated": ["00280100",  "US",  "1"],
                "Bits Stored": ["00280101",  "US",  "1"],
                "High Bit": ["00280102",  "US",  "1"],
                "Pixel Representation": ["00280103",  "US",  "1"],
                "Lossy Image Compression": ["00282110",  "CS",  "2"],
                #!    "SOP Common": {
                "SOP Class UID": ["00080016",  "UI",  "1"],
                "SOP Instance UID": ["00080018",  "UI",  "1"],
                
                #! General Image
                "Instance Number": ["00200013",  "IS",  "2"],
                "Patient Orientation": ["00200020",  "CS",  "2"],
            }
        else:
            self.metadata = metadata_tags

        # ? Check if metadata JSON file is provided and what type it is.
        if metadata_json:
            if os.path.exists(metadata_json):
                self.check_filetype(metadata_json)
            else:
                print(f"Metadata JSON file not found: {metadata_json}")
                return
        # ? Create a DICOM file with the metadata provided.

        # ? Check if input file (video or image) exists and process it.
        if not os.path.exists(input_file):
            print(f"Input file does not exist: {input_file}")
            return
        else:
            self.input_file = self.process_file(input_file)

        self.ds = self.create_dicom_dict(output_file)

#! VIDEO/IMAGE PROCESSING
    def process_file(self, input_file_path):
        type_file = self.check_type(input_file_path)
        if type_file == "image":
            self.process_image(input_file_path)
            return True
        elif type_file == "video":
            self.process_video(input_file_path)
            return True
        else:
            return "Unsupported file format."

    def check_type(self, file_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        if any(file_path.lower().endswith(ext) for ext in video_extensions):
            return "video"
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        if any(file_path.lower().endswith(ext) for ext in image_extensions):
            return "image"

    def process_image(self, input_file_path):
        acquisition_datetime = self.extract_datetime_from_image(
            input_file_path) or datetime.datetime.now()
        img = Image.open(input_file_path).convert("RGB")
        print("type(img): ", type(img))
        frame = np.array(img)
        self.metadata['Samples per Pixel'].append(
            self.get_samples_per_pixel(frame))
        self.metadata['Photometric Interpretation'].append(
            "RGB" if self.get_samples_per_pixel(frame) == 3 else "MONOCHROME2")
        rows, cols = frame.shape[:2]
        self.metadata['Rows'].append(rows)
        self.metadata['Columns'].append(cols)
        self.metadata['Number of Frames'].append(1)
        self.metadata['PixelData'] = frame.astype(np.uint8).tobytes()

    def process_video(self, input_file_path):
        acquisition_datetime = self.extract_datetime_from_video(
            input_file_path) or datetime.datetime.now()

        cap = cv2.VideoCapture(input_file_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {input_file_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_time = 1 / fps if fps > 0 else 1 / 25
        frames = []
        frame_count = 0
        while True and frame_count < 999:
            success, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
        cap.release()
        if not frames:
            raise ValueError("No frames were extracted from the video.")
        elif frame_count == 999:
            print("Warning: Maximum frame limit (999) reached.")
        rows, cols = frames[0].shape[:2]
        samples_per_pixel = frames[0].shape[2] if len(
            frames[0].shape) == 3 else 1
        photometric_interpretation = "RGB" if samples_per_pixel == 3 else "MONOCHROME2"
        planar_configuration = "0" if samples_per_pixel == 3 else 1
        self.metadata['Planar Configuration'].append(planar_configuration)
        self.metadata['Study Date'].append(
            acquisition_datetime.strftime("%Y%m%d"))
        self.metadata['Study Time'].append(
            acquisition_datetime.strftime("%H%M%S"))
        self.metadata['Samples per Pixel'].append(str(samples_per_pixel))
        self.metadata['Photometric Interpretation'].append(
            photometric_interpretation)
        self.metadata['Rows'].append(str(rows))
        self.metadata['Columns'].append(str(cols))
        self.metadata['Number of Frames'].append(str(int(frame_count)))
        self.metadata['Frame Time'].append(str(int(frame_time)))
        self.pixel_data = np.stack(frames, axis=0).astype(np.uint8)
        #! decomenntare salvo 04-02-25
        self.data = self.pixel_data.tobytes()
        #! decomenntare salvo 04-02-25
        self.metadata['Bits Allocated'].append('8')
        self.metadata['Bits Stored'].append('8')
        self.metadata['High Bit'].append('7')
        self.metadata['Pixel Representation'].append('0')
        self.metadata['Frame Increment Pointer'].append(
            self.metadata['Frame Time'][0])

    def get_samples_per_pixel(self, frame):
        if len(frame.shape) == 2:
            return 1  # Grayscale image.
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            return 3  # Color image (RGB).
        else:
            raise ValueError("Unknown image format.")

    def extract_datetime_from_image(self, file_path):
        try:
            img = Image.open(file_path)
            exif_data = img._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    decoded_tag = ExifTags.TAGS.get(tag, tag)
                    if decoded_tag == "DateTimeOriginal":
                        return datetime.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
        except Exception as e:
            print(f"Error extracting EXIF metadata: {e}")
        return None

    def extract_datetime_from_video(self, file_path):
        try:
            timestamp = os.path.getmtime(file_path)
            return datetime.datetime.fromtimestamp(timestamp)
        except Exception as e:
            print(f"Error reading video metadata: {e}")
        return None

#! METADATA PROCESSING
    def check_filetype(self, json_file):
        if json_file.endswith("csv"):
            csv = pd.read_csv(json_file)
            for i in range(len(csv)):
                tag = csv.loc[i, 'name_tag']
                value = csv.loc[i, 'value']
                self.metadata[tag].append(value)
        elif json_file.endswith("json"):
            with open(json_file, "r") as json_file:
                self.metadata = self.assign_values(
                    json.load(json_file), self.metadata)
        print("Metadata: ", self.metadata)

    def create_dicom_dict(self, output_dicom_path):
        a = "1.2.840.10008.5.1.4.1.1.77.1.4.1"
        self.metadata["SOP Instance UID"].append(a)
        self.metadata['SOP Class UID'].append(a)
        self.metadata["Study Instance UID"].append(pydicom.uid.generate_uid())
        self.metadata["Series Instance UID"].append(pydicom.uid.generate_uid())

        dcm_meta = pydicom.dataset.Dataset()
        dcm_meta.file_meta = pydicom.dataset.FileMetaDataset()
        dcm_meta.file_meta.MediaStorageSOPClassUID = UID(
            '1.2.840.10008.5.1.4.1.1.77.1.4.1')
        dcm_meta.file_meta.MediaStorageSOPInstanceUID = UID(
            '1.2.840.10008.5.1.4.1.1.77.1.4.1')
        
        dcm_meta.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dcm_meta.file_meta.FileMetaInformationVersion = b"\x00\x01"
        dcm_meta.is_little_endian = True
        dcm_meta.is_implicit_VR = False
        
        dcm = self.check_values(self.metadata)
        dcm = Dataset.from_json(dcm)
        dcm = FileDataset(None, dataset=dcm, file_meta=dcm_meta, preamble=b"\0" * 128)
        dcm.PixelData = self.dataA
        
        pydicom.filewriter.dcmwrite(output_dicom_path, dcm)
        return f"DICOM saved to: {output_dicom_path}"

    def remove_spaces(self, vr, value):
        if vr == 'PN':
            return value.replace(" ", "^")
        return value

    def check_values(self, dictio):
        dcm = {}
        for idx, item in enumerate(dictio.items()):
            if len(item[1]) > 3:
                if isinstance(item[1][-1], dict):
                    dcm[f'{item[1][0]}'] = {'vr': item[1][1], 'Value': [
                        self.check_values(self.remove_spaces(item[1][1], item[1][-1]))]}
                else:
                    dcm[f'{item[1][0]}'] = {'vr': item[1][1], 'Value': [
                        self.remove_spaces(item[1][1], item[1][-1])]}
            else:
                if item[1][-1] == '1' and "UID" not in item[0]:
                    print(f"Missing value for {item[0]} that is required.")
                    exit()
                elif item[1][-1] == '2':
                    dcm[f'{item[1][0]}'] = {'vr': item[1][1], 'Value': [None]}
                elif item[1][-1] == '3':
                    pass
        return dcm

    def assign_values(self, data, schema):
        """Assegna i valori ai tag DICOM rispettando la struttura dello schema."""
        result = {}

        for key, value in schema.items():
            if isinstance(value, list): 
                if len(value) > 3:
                    tag_code, vr, vm, *sub_schema = value 
                else:
                    tag_code, vr, vm = value
                    sub_schema = None
                if key in data:
                    if sub_schema and isinstance(sub_schema[0], dict):
                        result[key] = [tag_code, vr, vm,
                                       self.assign_values(data[key], sub_schema[0])]
                    else:
                        result[key] = [tag_code, vr, vm, data[key]]
                else:
                    result[key] = [tag_code, vr, vm]
            else:
                raise ValueError(
                    f"Formato schema non valido per chiave: {key}")

        return result

def main():

    parser = argparse.ArgumentParser(description="Convert input file to DICOM format.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input file (image or video).")
    parser.add_argument("-m", "--metadata", required=False, help="Path to the metadata JSON file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output DICOM file.")
    args = parser.parse_args()

    converter = Input2DICOM(input_file=args.input, metadata_json=args.metadata, output_file=args.output)
    print(converter.ds)

if __name__ == "__main__":
    main()

