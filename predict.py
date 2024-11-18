#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from argparse import Namespace
import warnings
import yaml
import torch
import cv2
from rich import print
from imutils import paths
from rich.progress import track
import numpy as np 
# from sklearn.metrics import accuracy_score

from lprnet import LPRNet, numpy2tensor, decode, accuracy

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open("config/kor_config.yaml",encoding="utf8") as f:
        args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    load_model_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    lprnet = LPRNet(args).to(device).eval()
    lprnet.load_state_dict(torch.load(args.pretrained))
    print(f"Successful to build network in {time.time() - load_model_start}sec")

    imgs = [el for el in paths.list_images(args.test_dir)]
    labels = [
        os.path.basename(n).split(".")[0].split("-")[0].split("_")[0]
        for n in track(imgs, description="Making labels... ")
    ]
    # print(labels)

    # Warm Up
    im = numpy2tensor(cv2.imdecode(np.fromfile(imgs[0], dtype=np.uint8), cv2.IMREAD_UNCHANGED), args.img_size).unsqueeze(0).to(device)
    lprnet(im)

    times = []
    preds = []
    acc = []
    for i, img in track(
        enumerate(imgs),
        description="Inferencing... ",
        total=len(imgs),
    ):
        try:
            im = numpy2tensor(cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED), args.img_size).unsqueeze(0).to(device)
        except:
            continue
        t0 = time.time()
        logit = lprnet(im).detach().to("cpu")
        pred, _ = decode(logit, args.chars)
        # print(pred)
        t1 = time.time()

        acc.append(pred[0] == labels[i])
        times.append((t1 - t0) * 1000)
        preds.append(pred)

    print("\n-----Accuracy-----")
    print(
        f"correct: {sum(acc)}/{len(acc)}, "
        + f"incorrect: {len(acc) - sum(acc)}/{len(acc)}"
    )
    print(f"accuracy: {sum(acc) / len(acc) * 100:.2f} %")
    print("\n-----inference time-----")
    print(f"mean: {sum(times) / len(times):.4f} ms")
    print(f"max: {max(times):.4f} ms")
    print(f"min: {min(times):.4f} ms")

class OcrPredictor:
    def __init__(self):
        with open("./LPRNet/config/kor_config.yaml",encoding="utf8") as f:
            self.args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        load_model_start = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)
        self.lprnet = LPRNet(self.args).to(self.device).eval()
        self.lprnet.load_state_dict(torch.load(self.args.pretrained))
        print(f"Successful to build network in {time.time() - load_model_start}sec")

    def predict(self, img):
        im = numpy2tensor(img, self.args.img_size).unsqueeze(0).to(self.device)
        self.lprnet(im)
        logit = self.lprnet(im).detach().to("cpu")
        # print("logit", logit)
        pred, _ = decode(logit, self.args.chars)
        # print("pred", pred)
        return pred