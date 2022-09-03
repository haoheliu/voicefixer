#!/usr/bin/python3
from genericpath import exists
import os.path
import argparse
from voicefixer import VoiceFixer
import torch
import os


def writefile(infile, outfile, mode, append_mode, cuda, verbose=False):
    if append_mode is True:
        outbasename, outext = os.path.splitext(os.path.basename(outfile))
        outfile = os.path.join(
            os.path.dirname(outfile), "{}-mode{}{}".format(outbasename, mode, outext)
        )
    if verbose:
        print("Processing {}, mode={}".format(infile, mode))
    voicefixer.restore(input=infile, output=outfile, cuda=cuda, mode=int(mode))

def check_arguments(args):
    process_file, process_folder = len(args.infile) != 0, len(args.infolder) != 0
    # assert len(args.infile) == 0 and len(args.outfile) == 0 or process_file, \
    #         "Error: You should give the input and output file path at the same time. The input and output file path we receive is %s and %s" % (args.infile, args.outfile)
    # assert len(args.infolder) == 0 and len(args.outfolder) == 0 or process_folder, \
    #         "Error: You should give the input and output folder path at the same time. The input and output folder path we receive is %s and %s" % (args.infolder, args.outfolder)
    assert (
        process_file or process_folder
    ), "Error: You need to specify a input file path (--infile) or a input folder path (--infolder) to proceed. For more information please run: voicefixer -h"

    # if(args.cuda and not torch.cuda.is_available()):
    #     print("Warning: You set --cuda while no cuda device found on your machine. We will use CPU instead.")
    if process_file:
        assert os.path.exists(args.infile), (
            "Error: The input file %s is not found." % args.infile
        )
        output_dirname = os.path.dirname(args.outfile)
        if len(output_dirname) > 1:
            os.makedirs(output_dirname, exist_ok=True)
    if process_folder:
        assert os.path.exists(args.infolder), (
            "Error: The input folder %s is not found." % args.infile
        )
        output_dirname = args.outfolder
        if len(output_dirname) > 1:
            os.makedirs(args.outfolder, exist_ok=True)

    return process_file, process_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VoiceFixer - restores degraded speech"
    )
    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        default="",
        help="An input file to be processed by VoiceFixer.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        default="outfile.wav",
        help="An output file to store the result.",
    )

    parser.add_argument(
        "-ifdr",
        "--infolder",
        type=str,
        default="",
        help="Input folder. Place all your wav file that need process in this folder.",
    )
    parser.add_argument(
        "-ofdr",
        "--outfolder",
        type=str,
        default="outfolder",
        help="Output folder. The processed files will be stored in this folder.",
    )

    parser.add_argument(
        "--mode", help="mode", choices=["0", "1", "2", "all"], default="0"
    )
    parser.add_argument('--disable-cuda', help='Set this flag if you do not want to use your gpu.', default=False, action="store_true")
    parser.add_argument(
        "--silent",
        help="Set this flag if you do not want to see any message.",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.disable_cuda:
        cuda = True
    else:
        cuda = False

    process_file, process_folder = check_arguments(args)

    if not args.silent:
        print("Initializing VoiceFixer")
    voicefixer = VoiceFixer()

    if not args.silent:
        print("Start processing the input file %s." % args.infile)

    if process_file:
        audioext = os.path.splitext(os.path.basename(args.infile))[-1]
        if audioext != ".wav":
            raise ValueError(
                "Error: Error processing the input file. We only support the .wav format currently. Please convert your %s format to .wav. Thanks."
                % audioext
            )
        if args.mode == "all":
            for file_mode in range(3):
                writefile(
                    args.infile,
                    args.outfile,
                    file_mode,
                    True,
                    cuda,
                    verbose=not args.silent,
                )
        else:
            writefile(
                args.infile,
                args.outfile,
                args.mode,
                False,
                cuda,
                verbose=not args.silent,
            )

    if process_folder:
        if not args.silent:
            files = [
                file
                for file in os.listdir(args.infolder)
                if (os.path.splitext(os.path.basename(file))[-1] == ".wav")
            ]
            print(
                "Found %s .wav files in the input folder %s. Start processing."
                % (len(files), args.infolder)
            )
        for file in os.listdir(args.infolder):
            outbasename, outext = os.path.splitext(os.path.basename(file))
            in_file = os.path.join(args.infolder, file)
            out_file = os.path.join(args.outfolder, file)

            if args.mode == "all":
                for file_mode in range(3):
                    writefile(
                        in_file,
                        out_file,
                        file_mode,
                        True,
                        cuda,
                        verbose=not args.silent,
                    )
            else:
                writefile(
                    in_file, out_file, args.mode, False, cuda, verbose=not args.silent
                )

    if not args.silent:
        print("Done")