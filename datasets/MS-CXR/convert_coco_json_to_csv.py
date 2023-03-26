import pandas as pd
from pathlib import Path
from pycocotools.coco import COCO

DICOM_ID_COL_NAME = "dicom_id"
IMAGE_PATH_COL_NAME = "path"
CATEGORY_COL_NAME = "category_name"
LABEL_TEXT_COL_NAME = "label_text"
BBOX_X_COORD_COL_NAME = "x"
BBOX_Y_COORD_COL_NAME = "y"
BBOX_WIDTH_COL_NAME = "w"
BBOX_HEIGHT_COL_NAME = "h"
IMAGE_WIDTH_COL_NAME = "image_width"
IMAGE_HEIGHT_COL_NAME = "image_height"

def convert_json_to_csv(path_to_json_file: Path, path_to_output_csv_file: Path) -> None:
    """
    This script converts annotations from the MS-CXR coco json file into a csv format.
    The csv file has the following columns:
    `dicom_id`, `path`, `category_name`, `label_text`, `image_width`, `image_height`
    and bounding boxes described by `x`, `y`, `w` and `h`.

    A combination of image, finding (`category_name`) and phrase (`label_text`) can have
    multiple bounding boxes associated with it, and in this case the bounding boxes will appear
    as separate entries in the csv, one per row.

    :param path_to_json_file: Path to the MS-CXR COCO json file
    :param path_to_output_csv_file: The output csv will be written to this location.
    """
    coco = COCO(annotation_file=path_to_json_file)

    image_phrase_pairs = []

    for img_id, anns in coco.imgToAnns.items():
        img = coco.loadImgs(img_id)[0]
        dicom_id = Path(img["file_name"]).stem
        for ann in anns:
            category_name = coco.loadCats(ann["category_id"])[0]["name"]
            bbox = ann["bbox"]
            image_phrase_pairs.append({DICOM_ID_COL_NAME: dicom_id,
                                       CATEGORY_COL_NAME: category_name,
                                       LABEL_TEXT_COL_NAME: ann["label_text"],
                                       IMAGE_PATH_COL_NAME: img["path"],
                                       BBOX_X_COORD_COL_NAME: int(bbox[0]),
                                       BBOX_Y_COORD_COL_NAME: int(bbox[1]),
                                       BBOX_WIDTH_COL_NAME: int(bbox[2]),
                                       BBOX_HEIGHT_COL_NAME: int(bbox[3]),
                                       IMAGE_WIDTH_COL_NAME: img["width"],
                                       IMAGE_HEIGHT_COL_NAME: img["height"]})

    df = pd.DataFrame.from_records(image_phrase_pairs)
    df.to_csv(path_to_output_csv_file, index=False)

if __name__ == "__main__":
    convert_json_to_csv("MS_CXR_Local_Alignment_v1.0.0.json", "MS_CXR_Local_Alignment_v1.0.0.csv")