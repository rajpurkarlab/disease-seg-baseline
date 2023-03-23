import argparse
import sys

from models.run_biovil import plot_phrase_grounding as ppgb
from models.run_chexzero import plot_phrase_grounding as ppgc

def parse_clip_use_case_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        choices=["chexzero", "biovil"],
        help="name of the model to use",
    )
    parser.add_argument(
        "img_path",
        type=str,
        help="path to image",
    )
    parser.add_argument(
        "query",
        type=str,
        help="NL text for segmentation",
    )
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")
    return args

def main():
    args = parse_clip_use_case_args()
    if args.model_name == "chexzero":
        ppgc(args.img_path, args.query.replace("_", " "))
    elif args.model_name == "biovil":
        ppgb(args.img_path, args.query.replace("_", " "))
    else:
        raise NotImplementedError(
                f"{args.model_name} target_name is not implemented!"
            )

if __name__ == "__main__":
    main()