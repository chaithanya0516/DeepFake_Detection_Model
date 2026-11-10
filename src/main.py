"""DeepFakeShield command line entry point: train, predict, video."""

import argparse

import predict
import train
import video_processor

COMMANDS = (
    ("train", train, "train the CNN and save it to models/"),
    ("predict", predict, "classify one or more images"),
    ("video", video_processor, "classify a video by averaging frame scores"),
)


def build_parser():
    parser = argparse.ArgumentParser(prog="deepfakeshield", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, module, help_text in COMMANDS:
        command = subparsers.add_parser(name, help=help_text, description=help_text)
        module.add_arguments(command)
        command.set_defaults(handler=module.run)
    return parser


def main():
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
