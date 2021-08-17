import io
import shlex
import subprocess

import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
import argparse
import os
import sys
import shutil
#import Global.test
#import Face_Detection.detect_all_dlib
#import Face_Enhancement.test_face
from subprocess import call
from types import SimpleNamespace
from PIL import Image

sg.theme("darkAmber")


def modify(image_filename=None, cv2_frame=None, win: sg.Window = None):
    def print_pb(value,index):
        print(value)
        if win:
            win.write_event_value("$pb", (index,value))

    def run_cmd(command):
        try:
            call(command, shell=True)
        except KeyboardInterrupt:
            print("Process interrupted")
            sys.exit(1)
        except subprocess.SubprocessError:
            print("running:", repr(shlex.split(command)))
            raise

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str,
                        default=image_filename, help="Test images")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="Restored images, please use the absolute path",
    )
    parser.add_argument("--GPU", type=str, default="-1", help="0,1,2")
    parser.add_argument(
        "--checkpoint_name", type=str, default="Setting_9_epoch_100", help="choose which checkpoint"
    )
    parser.add_argument("--with_scratch", default="--with_scratch", action="store_true")
    opts = parser.parse_args()

    gpu1 = opts.GPU

    # resolve relative paths before changing directory
    print(opts.input_folder, "======>")
    opts.input_folder = os.path.abspath(opts.input_folder)
    print(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    main_environment = os.getcwd()

    # Stage 1: Overall Quality Improve
    print_pb("Running Stage 1: Overall restoration",1)
    os.chdir("./Global")
    stage_1_input_dir = opts.input_folder
    stage_1_output_dir = os.path.join(
        opts.output_folder, "stage_1_restore_output")
    if not os.path.exists(stage_1_output_dir):
        os.makedirs(stage_1_output_dir)

    if not opts.with_scratch:
        namespace = SimpleNamespace()
        Global.test.main()
        stage_1_command = (
            "python test.py", "--test_mode Full", "--Quality_restore", "--test_input"
            , stage_1_input_dir
            , "--outputs_dir"
            , stage_1_output_dir
            , "--gpu_ids"
            , gpu1
        )
        run_cmd(stage_1_command)
    else:

        mask_dir = os.path.join(stage_1_output_dir, "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")
        stage_1_command_1 = (
            "python", "detection.py",
            "--test_path", stage_1_input_dir
            , "--output_dir"
            , mask_dir
            , "--input_size", "full_size"
            , "--GPU"
            , gpu1
        )
        stage_1_command_2 = (
            "python", "test.py", "--Scratch_and_Quality_restore", "--test_input"
            , new_input
            , "--test_mask"
            , new_mask
            , "--outputs_dir"
            , stage_1_output_dir
            , "--gpu_ids"
            , gpu1
        )
        run_cmd(stage_1_command_1)
        run_cmd(stage_1_command_2)

    # Solve the case when there is no face in the old photo
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, stage_4_output_dir)

    print("Finish Stage 1 ...")
    print("\n")

    # Stage 2: Face Detection

    print_pb("Running Stage 2: Face Detection",2)

    os.chdir(".././Face_Detection")
    stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_2_output_dir = os.path.join(
        opts.output_folder, "stage_2_detection_output")
    if not os.path.exists(stage_2_output_dir):
        os.makedirs(stage_2_output_dir)
    stage_2_command = (
        "python", "detect_all_dlib.py", "--url", stage_2_input_dir,
        "--save_url", stage_2_output_dir
    )
    run_cmd(stage_2_command)
    print("Finish Stage 2 ...")
    print("\n")

    # Stage 3: Face Restore
    print_pb("Running Stage 3: Face Enhancement",3)
    os.chdir(".././Face_Enhancement")
    stage_3_input_mask = "./"
    stage_3_input_face = stage_2_output_dir
    stage_3_output_dir = os.path.join(
        opts.output_folder, "stage_3_face_output")
    if not os.path.exists(stage_3_output_dir):
        os.makedirs(stage_3_output_dir)
    stage_3_command = (
        "python"
        , "test_face.py"
        , "--old_face_folder"
        , stage_3_input_face
        , "--old_face_label_folder"
        , stage_3_input_mask
        , "--tensorboard_log"
        , "--name"
        , opts.checkpoint_name
        , "--gpu_ids"
        , gpu1
        , "--load_size"
        , "256"
        , "--label_nc"
        , "18"
        , "--no_instance"
        , "--preprocess_mode"
        , "resize"
        , "--batchSize", "4"
        , "--results_dir"
        , stage_3_output_dir
        , "--no_parsing_map"
    )
    run_cmd(stage_3_command)
    print("Finish Stage 3 ...")
    print("\n")

    # Stage 4: Warp back
    print_pb("Running Stage 4: Blending",4)
    os.chdir(".././Face_Detection")
    stage_4_input_image_dir = os.path.join(
        stage_1_output_dir, "restored_image")
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    stage_4_command = (
        "python"
        , "align_warp_back_multiple_dlib.py"
        , "--origin_url"
        , stage_4_input_image_dir
        , "--replace_url"
        , stage_4_input_face_dir
        , "--save_url"
        , stage_4_output_dir
    )
    run_cmd(stage_4_command)
    print("Finish Stage 4 ...")
    print("\n")

    print_pb("All the processing is done. Please check the results.",5)
    if win:
        win.write_event_value('$done-image', stage_4_output_dir)


# --------------------------------- The GUI ---------------------------------

def gui():
    # First the window layout...
    images_col = [[sg.Text('Input file:'), sg.In(enable_events=True, key='-IN FILE-'), sg.FileBrowse()],
                  [sg.Button('Modify Photo', key='-MPHOTO-'), sg.Button('Exit')],
                  [sg.Image(filename='', size=(20, 20), key='-IN-'), sg.Image(filename='', key='-OUT-')],
                  [sg.ProgressBar(5, orientation='h', size=(20, 20), key="-pb-"), sg.Text("0", key="-pb-text-")]]
    # ----- Full layout -----
    layout = [[sg.VSeperator(), sg.Column(images_col)]]

    # ----- Make the window -----
    window = sg.Window('Bringing-old-photos-back-to-life', layout, grab_anywhere=True).finalize()
    # ----- Run the Event Loop -----
    prev_filename = colorized = cap = None
    filename = ''

    while True:
        event, values = window.read()
        if event in (None, 'Exit'):
            break

        elif event == "$pb":
            window["-pb-"].update(values["$pb"][0])
            window["-pb-text-"].update(values["$pb"][1])

        elif event == '-MPHOTO-':
            # n1 = filename.split(os.path.sep)[-2]
            # n2 = filename.split(os.path.sep)[-3]
            # n3 = filename.split(os.path.sep)[-1]
            # filename = str(f".\\{n2}\\{n1}")
            import threading
            threading.Thread(target=modify, args=(os.path.dirname(filename),), kwargs=dict(win=window)).start()

        elif event == "$done-image":

            # global f_image
            f_image = f'./output/final_output/{os.path.basename(filename)}'

            image = Image.open(filename)
            image.thumbnail((400, 400))
            bio = io.BytesIO()
            image.save(bio, format="PNG")

            window["-OUT-"].update(data=bio.getvalue())

        elif event == '-IN FILE-':  # A single filename was chosen
            filename = values['-IN FILE-']
            if not filename:
                continue

            if filename != prev_filename:
                prev_filename = filename
                try:
                    image = Image.open(filename)
                    image.thumbnail((400, 400))
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    window["-IN-"].update(data=bio.getvalue())

                except Exception:
                    import traceback

                    traceback.print_exc()
                    continue

    # ----- Exit program -----
    window.close()


if __name__ == '__main__':
    gui()
