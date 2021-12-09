import pandas as pd
import torchvision
import numpy as np

from torch import Tensor
from src.models.catalog.frame_info import FrameInfo
from src.models.catalog.properties import ColorSpace
# from src.models.siammask import *
import torch
import SiamMask
# # import SiamMask.utils as smutils
import SiamMask.utils.config_helper as cfhelper
import SiamMask.experiments.siammask_sharp.custom as csm

import SiamMask.utils.load_helper as ldhelper
import SiamMask.tools.test as smtest

import cv2
import argparse

class SiamMaskTracker:
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    @property
    def name(self) -> str:
        return "siammask"

    def __init__(self):

        print("SIAMMASK INIT")
        # self.model = siamMask()

        
        parser = argparse.ArgumentParser()
        parser.add_argument("--resume")
        parser.add_argument("--config")
        parser.add_argument("--base_path")
        # // parser.add_argument("--cpu")
        # // let args = parser.parse_args(["--resume", "../SiamMask/experiments/siammask_sharp/SiamMask_VOT.pth", "--config", "../SiamMask/experiments/siammask_sharp/config_vot.json", "--base_path", "./OIST_Data/downsampled"])
        # // let args = parser.parse_args(["--resume", "../SiamMask/checkpoint_e20.pth", "--config", "../SiamMask/experiments/siammask_sharp/config_vot.json", "--base_path", "./OIST_Data/downsampled"])
        print("here")
        # args = parser.parse_args(["--resume", "./SiamMask/model_sharp/checkpoint_e20.pth", "--config", "./SiamMask/experiments/siammask_sharp/config_vot.json", "--base_path", "./OIST_Data/downsampled"])
        args = parser.parse_args(["--resume", "./SiamMask/model_sharp/checkpoint_e20.pth", "--config", "./SiamMask/experiments/siammask_sharp/config_vot.json","--base_path", "./data/ua_detrac"])

        print("ARGUMENTS", args)
        self.device = torch.device('cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        print("load config")
        self.cfg = cfhelper.load_config(args)
        print("sssss2")
        self.siammask = csm.Custom(anchors=self.cfg['anchors'])
        print("sssss3")
        if args.resume:
            # assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            self.siammask = ldhelper.load_pretrain(self.siammask, args.resume)
            print("sddddd")
        print("sfffffs")

        self.siammask.eval().to(self.device)






    @property
    def input_format(self) -> FrameInfo:
        return FrameInfo(-1, -1, 3, ColorSpace.RGB)


    def track(self, ims: Tensor) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed
        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_boxes (List[List[BoundingBox]]),
            predicted_scores (List[List[float]])
        """
        # Select ROI
        cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
        
        print("GET PREDICTIONS")
        print("type ims", type(ims))
        # print("ims", ims, ims.shape)
        # print("ims[0]", ims[0], ims[0].shape)
        # img_np = ims.to_numpy()
        print("ims[0] type", type(ims[0]))
        outcome = pd.DataFrame()
        # print("ims[0] numpy", type(img_np), img_np.shape)
        try:
            init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
            x, y, w, h = init_rect
        except:
            print("aaaaahhhh")
            exit()

        toc = 0
        print("here shape of ims", ims.shape)
        for f, im in enumerate(ims):
            tic = cv2.getTickCount()
            print(f)
            if f == 0:  # init
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                state = smtest.siamese_init(im, target_pos, target_sz, self.siammask, self.cfg['hp'], device=self.device)  # init tracker
            elif f > 0:  # tracking
                state = smtest.siamese_track(state, im, mask_enable=True, refine_enable=True, device=self.device)  # track
                # state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device, debug = True)  # track

                location = state['ploygon'].flatten()
                outcome = outcome.append(
                        {
                            "boxes": np.array(location)
                        },
                        ignore_index=True)
                mask = state['mask'] > state['p'].seg_thr

                im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
                cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                # print("image shape", im.shape)
                cv2.imshow('SiamMask', im)
                key = cv2.waitKey(1)
                if key > 0:
                    break

            toc += cv2.getTickCount() - tic


                
        print("end of the function")
        print("output", outcome)
        return outcome


    def __call__(self, *args, **kwargs):
        print("CALL FUNCTION")
        print("kwargs", kwargs)
        # # print("args", args[0].data)
        # # print()
        # self.track(args)


        frames = None
        if len(args):
            frames = args[0]
        if isinstance(frames, pd.DataFrame):
            frames = frames.transpose().values.tolist()[0]
        return self.track(frames)
        
