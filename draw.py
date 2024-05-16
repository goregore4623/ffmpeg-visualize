import numpy as np
import os
import seaborn as sns
import random
from PIL import Image, ImageDraw, ImageFile, ImageFont
from shapely.geometry import Polygon
from typing import Any
import ffmpeg
import re
import glob
import cv2
from pathlib import Path
import argparse


# txtファイルの読み込み , 区切りで抽出
def _load_sequence(det_path: Path) -> np.ndarray: 
    return np.loadtxt(det_path, delimiter=",")#入力ファイルの読み込み,を文字間の区切りとする

        
def show_args(args: argparse): # 引数の表示
    print("--------------------------------------------------------------")
    print(" Args                      | values")
    print("--------------------------------------------------------------")
    args_data = vars(args)
    for arg_name, arg_value in args_data.items():
        print(f" {arg_name.ljust(25)} : {arg_value}")
    
    print("--------------------------------------------------------------")
    
def get_extension(image_dir: Path):
    """
    get extension of image files
    input:
    -image_dir (Path)
    output:
    -extension (str)
    """
    extension = list(image_dir.glob('*'))[0].suffix
    return extension

class VideoWriter:
    def __init__(
        self,
        filename,
        framerate=1,
        size=(1920, 1080),
        pix_fmt_in=None,
        pix_fmt_ou="yuv420p",
        quality=28,
    ):
        self.filename = filename
        self.framerate = framerate
        self.out = None
        self.width = size[0]
        self.height = size[1]
        self.maxsize = max(size)
        self.quality = quality
        self.pix_fmt_in = pix_fmt_in
        self.pix_fmt_ou = pix_fmt_ou

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.out is not None:
            self.out.stdin.close()
            self.out.wait()

    def _resize(self, image, max_size, image_size):
        image_height, image_width = image_size
        aspect_ratio = float(image_width) / float(image_height)

        if image_width > image_height:
            new_width = max_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(new_height * aspect_ratio)

        return cv2.resize(
            np.array(image), (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )

    def openpipe(self, size, pix_fmt_in):
        width, height = size
        fps = self.framerate
        quality = self.quality
        output = self.filename

        return (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt=pix_fmt_in,
                s="{}x{}".format(width, height),
                r=fps,
            )
            # .filter("scale", output_size[0], output_size[1])
            .output(
                str(output),
                pix_fmt=self.pix_fmt_ou,
                qmax=quality,
                qmin=quality,
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
            # .run_async(pipe_stdin=True, quiet=True)
        )

    def write(self, image):
        self.out.stdin.write(np.array(image).tobytes())

    def update(self, image):
        if isinstance(image, Image.Image):
            orig_width, orig_height = image.size
            pix_fmt = "rgb24"
        elif isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
            pix_fmt = "bgr24"
            pass
        else:
            raise ValueError(f"image must be Image or ndarray, but {type(image)}!")

        if orig_width != self.width or orig_height != self.height:
            out_image = self._resize(
                image,
                max_size=self.maxsize,
                image_size=(orig_height, orig_width),
            )
            out_height, out_width, _ = out_image.shape
        else:
            out_image = image
            out_height, out_width = orig_height, orig_width
            
        # 奇数サイズの画像の場合に偶数サイズに調整する
        out_height = out_height - (out_height % 2)
        out_width = out_width - (out_width % 2)
        out_image = cv2.resize(out_image, (out_width, out_height))

        if self.out is None:
            self.out = self.openpipe((out_width, out_height), pix_fmt_in=pix_fmt)

        self.write(out_image)
    
class AnnotaionVisualizer:
    def __init__(
        self,
        image_size: tuple = (1920, 1080),
        class_max_mask_bit: int = 0b01111111,
        bbox_line_width: int = 3,
        track_line_width: int = 5,
        bbox_alpha: int = 128,
        line_alpha: int = 200,
        font_size: int = 0,
        font_name="DejaVuSerif-Bold",
    ):

        self.__track_line_width = track_line_width
        self.__line_alpha = line_alpha

        # RGB color palette
        self.__colors = [
            tuple(int(c * 255) for c in color)
            for color in sns.hls_palette(n_colors=class_max_mask_bit + 0b01)
        ]
        _rs = np.random.RandomState(1234567)
        _rs.shuffle(self.__colors)

        self.__bbox_line_width = bbox_line_width
        self.__bbox_alpha = bbox_alpha
        self.__class_max_mask_bit = class_max_mask_bit

        self.__font_name = font_name
        # フォントを作成する
        if font_size == 0:
            font_size = max(12, int(0.025 * min(image_size)))

        self.__font = ImageFont.truetype(font_name, size=font_size)
        self.__font_size = font_size
        
    def draw_text(self, image, text: str, position=(10, 10), font_color=(255, 0, 0), alpha=150):
        """
        画像にテキストを描画する関数

        Parameters:
            image (PIL.Image.Image): 描画対象の画像
            text (str): 描画するテキスト
            position (tuple): テキストの描画位置 (x, y)
            font_color (tuple): フォントの色 (R, G, B)
            alpha (int): フォントの透明度

        Returns:
            PIL.Image.Image: テキストが描画された画像
        """
        draw = ImageDraw.Draw(image)
        # フォントの指定
        font=ImageFont.truetype(self.__font_name, size=int(self.__font_size))
        # テキストの描画
        draw.text(position, text, font=font, fill=(*font_color, alpha))

        return image

    def get_color_by_id(self, id):
        sel = int(int(id) & self.__class_max_mask_bit)
        if sel > self.__class_max_mask_bit:
            print(f"{sel},")
        return self.__colors[sel]
        # return self.__colors[int(id)]

    def draw_label_with_pil(self, draw, bbox, caption, color):
        xmin, ymin = bbox[0:2]
        tbbox = draw.textbbox((xmin, ymin), caption, font=self.__font)
        text_margin = tbbox[1] - ymin
        draw.rectangle(
            ((xmin, ymin), (tbbox[2] + text_margin, tbbox[3] + text_margin)),
            fill=color + (self.__bbox_alpha,),
        )
        draw.text((xmin, ymin), caption, fill="white", font=self.__font)
        draw.rectangle(
            ((xmin, ymin), (tbbox[2] + text_margin, tbbox[3] + text_margin)),
            outline=color + (self.__bbox_alpha,),
            width=self.__bbox_line_width,
        )

    def draw_rec_bbox_with_pil(self, draw, bbox, color):
        draw.rectangle(
            bbox,  # (xmin, ymin, xmax, ymax),
            outline=color + (self.__bbox_alpha,),
            width=self.__bbox_line_width,
        )
        
    def draw_poly_bbox_with_pil(self, draw, bbox, color):
        draw.polygon(
            bbox,  # (x1, y1, x2, y2, x3, y3, x4, y4),
            outline=color + (self.__bbox_alpha,),
            width=self.__bbox_line_width,
        )

    def draw_line(self, draw, track, color):
        draw.line(
            track,
            fill=color + (self.__line_alpha,),
            width=self.__track_line_width,
            joint=None,
            # joint="curve",
        )

    def puttext_top_with_pil(self, draw, text):
        tbbox = draw.textbbox(
            (10, 10),
            text,
            font=ImageFont.truetype(self.__font_name, size=int(self.__font_size * 1.6)),
        )
        text_margin = tbbox[1] - 10
        draw.rectangle(
            ((10, 10), (tbbox[2] + text_margin, tbbox[3] + text_margin)),
            fill=(255, 255, 255) + (60,),
            # fill=(0, 0, 0) + (50,),
        )
        draw.text(
            (10, 10),
            text,
            fill="red",
            font=ImageFont.truetype(self.__font_name, size=int(self.__font_size * 1.6)),
        )

    def puttext_bottom_with_pil(self, draw, text, pos):
        text_bbox = draw.textbbox(
            (0, 0),
            text,
            font=ImageFont.truetype(self.__font_name, size=int(self.__font_size * 1.6)),
        )

        draw.text(
            (text_bbox[0] + 10, pos[1] - text_bbox[3] - 10),
            text,
            fill="red",
            font=ImageFont.truetype(self.__font_name, size=int(self.__font_size * 1.6)),
        )

    def draw_history(
        self,
        image: Any,
        ids: list,
        history: np.ndarray,
        rec_type: str = "regular",
    ):
        draw = ImageDraw.Draw(image, mode="RGBA")
        # pass
        # 各オブジェクトIDに対応するバウンディングボックスの中心座標を計算
        center_points = {}
        for class_id in ids:
            class_data = history[history[:, 1] == class_id]
            if rec_type == "regular":
                centers = [
                    (int((2 * bbox[0] + bbox[2]) / 2), int((2 * bbox[1] + bbox[3]) / 2))
                    for bbox in class_data[:, 2:6]
                ]
                center_points[class_id] = centers
            elif rec_type == "rotate":
                centers = [
                    tuple(calc_ploygon_center(bbox))
                    for bbox in class_data[:, 2:10]
                ]
                center_points[class_id] = centers

        # バウンディングボックスの軌跡を描画
        for class_id, centers in center_points.items():
            # color = self.__colors[class_id]
            color = self.get_color_by_id(class_id)
            self.draw_line(draw, centers, color)

    def draw_annotations(
        self,
        image: Any,
        image_annotations: np.ndarray,
        rec_type: str = "regular",
        text: str = None,
        text_bottom: str = None,
        show_camera_id: bool = True,
        show_conf: bool = False,
        color_def: tuple[int]= None,
    ) -> Image.Image:
        draw: Any = None
        if isinstance(image, Image.Image):
            pass
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(np.flip(image, axis=2))
        else:
            raise ValueError(f"image must be Image or ndarray, but {type(image)}!")

        draw = ImageDraw.Draw(image, mode="RGBA")
        _puttext_top = self.puttext_top_with_pil
        _puttext_bottom = self.puttext_bottom_with_pil
        if rec_type == "regular":
            _draw_bbox_reg = self.draw_rec_bbox_with_pil
        elif rec_type == "rotate":
            _draw_bbox_poly = self.draw_poly_bbox_with_pil
        _draw_label = self.draw_label_with_pil
        w, h = image.size

        if text:
            _puttext_top(draw, text)

        if text_bottom:
            _puttext_bottom(draw, text_bottom, (w, h))

        for annotation in image_annotations:
            # class ID
            class_id = annotation[1].astype(int)

            # color for class ID
            if color_def is None:
                color = self.get_color_by_id(class_id)
            else:
                color = color_def

            # bbox
            # bboxは1~に対して画像の座標は0スタートなので、合わせるために-1する
            if rec_type == "regular":
                bbox_ltwh = annotation[2:6]
                xmin = bbox_ltwh[0] 
                ymin = bbox_ltwh[1] 
                xmax = xmin + bbox_ltwh[2]
                ymax = ymin + bbox_ltwh[3]
                bbox_minmax = [int(xmin), int(ymin), int(xmax), int(ymax)]
            elif rec_type == "rotate":
                bbox_minmax = annotation[2:10].astype(int).tolist()
                # print(f"id: {annotation[10]}")
            else :
                ValueError(f"rec_type must be regular or rotate, but {rec_type}!")

            # draw bbox
            caption = ""
            # 矩形を描画する。
            if rec_type == "regular" and bbox_minmax[0] <= bbox_minmax[2] and bbox_minmax[1] < bbox_minmax[3]:
                _draw_bbox_reg(draw, bbox_minmax, color)
            elif rec_type == "rotate":
                _draw_bbox_poly(draw, bbox_minmax, color)
            else:
                caption += "ERROR"

            # ラベルを設定
            caption += f" {class_id}"

            # ラベルに"conf" 表示
            if show_conf:
                caption += f"  {annotation[6]:.0%}"

            # ラベルに"camera_id" 表示
            if show_camera_id:
                camera_id = ""
                if len(annotation) > 10:
                    camera_id = int(annotation[10])

                    # if len(camera_id) != 0 and camera_id != -1:
                    if camera_id != -1 and camera_id != "":
                        # camera_idが存在する場合
                        caption += f" ({camera_id})"

            _draw_label(draw, bbox_minmax, caption, color)

        return image