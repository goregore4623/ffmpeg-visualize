from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import psutil

from draw import AnnotaionVisualizer, get_extension, show_args, _load_sequence, VideoWriter

class ImageViewer:
    def __init__(self, 
                 frame_num: int,
                 cam_rec_type: str,
                 put_timestamp: bool,
    ):
        self.cam_visualizer = AnnotaionVisualizer()
        
        self.frame_num = frame_num
        self.cam_rec_type = cam_rec_type
        self.put_timestamp = put_timestamp
        
    def img_open(self, img_path: Path):
        img = Image.open(img_path)
        result_img = np.array(img, dtype=np.uint8)
        img.close() 
        return result_img  

    def draw_img(self, frame_id: int ,det_objs, img_path: Path):
        img = Image.open(img_path)
        # 描画
        self.cam_visualizer.draw_annotations(img, det_objs, self.cam_rec_type)
        
        if self.put_timestamp: # タイムスタンプを挿入
            self.cam_visualizer.draw_text(img, str(Path(img_path).stem))
        
        result_img = np.array(img, dtype=np.uint8)
        img.close() 
        return result_img
    
    def create_video(self, imgs: list, output_file: str, fps: int=5): # 画像リストから動画を作成
        print(f'[INFO] creating video: {output_file}')
        W, H = imgs[0].shape[:2]
        with VideoWriter(
            filename=output_file,
            framerate=fps,
            size=(W, H),
        ) as visualizer:
            for im in imgs:
                visualizer.update(Image.fromarray(im))

            print(f"Save as {output_file}")
    
    def create_video_no_draw(self, img_lists: list, output_file: str, fps: int=5): # 画像リストから動画を作成
        print(f'[INFO] creating video: {output_file}')
        img = np.array(Image.open(img_lists[0]), dtype=np.uint8)
        
        W, H = img.shape[:2]
        with VideoWriter(
            filename=output_file,
            framerate=fps,
            size=(W, H),
        ) as visualizer:
            for img_list in img_lists:
                img = Image.open(img_list)
                img_np = np.array(img, dtype=np.uint8)
                visualizer.update(Image.fromarray(img_np))
                img.close()

            print(f"Save as {output_file}")

def main(args):
    # データ準備####################################################################
    img_dir = args.img_dir
    output_dir = args.output_dir
    csv_path = args.csv_path
    
    
    # ディレクトリが存在するか確認
    if not img_dir.is_dir():
        raise ValueError(f"The specified path {img_dir} is not a directory.")
    else:
        extension = get_extension(img_dir)
        img_lists = sorted([str(img) for img in img_dir.glob(f"*{extension}")])
        
    # フレーム数の設定
    if args.t_end:
        frame_num=args.t_end
    else:
        frame_num = len(img_lists)
    frame_ids = np.arange(1, frame_num + 1)
    
    viewer = ImageViewer(
            frame_num = frame_num,
            cam_rec_type = args.rec_type,
            put_timestamp = args.put_timestamp,
        )
    
    if csv_path is not None: # 描画情報がある場合
        det_objs = _load_sequence(csv_path)
        
        # 描画
        print(f'[INFO] drawing information')
        
        with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            draw_img_list = list(
                tqdm(
                    executor.map(
                        viewer.draw_img,
                        frame_ids,
                        [det_objs[det_objs[:, 0] == frame_id] for frame_id in frame_ids],
                        img_lists,
                    ),
                    total=frame_num,
                )
            )
        
        # 動画の保存
        video_path = f"{output_dir / args.prefix}.mp4"
        viewer.create_video(draw_img_list, str(video_path), args.fps)
        print(f'[INFO] result saved to: {video_path}')
    else: # 画像の読み込み
        draw_img_list = []
        print(f'[INFO] drawing information')
        viewer.create_video_no_draw(img_lists, str(output_dir / f"{args.prefix}.mp4"), args.fps)
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # input options
    parser.add_argument(
        "-i", "--img_dir", 
        type=Path, 
        help="path to img directory", 
        required=True
        )
    parser.add_argument(
        "-o", "--output_dir", 
        type=Path, 
        required=True, 
        help="path/to/mp4/output/dir"
        )
    parser.add_argument(
        "-p", "--prefix", 
        default="detect", 
        help="mp4/file/name"
        )
    
    # drawing options
    parser.add_argument(
        "-c", "--csv_path", 
        default=None, 
        help="text of unassociated bboxes", 
        required=False
        )
    parser.add_argument(
        '--no_textbox', 
        action='store_true' , 
        help='not draw id box'
        )
    parser.add_argument(
        "-f", "--fps", 
        type=float, 
        default=30, 
        help="FPS of video"
        )
    parser.add_argument(
        "-r","--rec_type",
        default="regular", 
        help="/choose/rectangle/type/rotate/or/regular/"
        )
    parser.add_argument(
        '--t_end', 
        type=int, 
        help='end frame number'
        )
    parser.add_argument(
        "--put_timestamp", 
        action="store_true", 
        help="Put timestamp on output video"
    )
    
    # parralel options
    parser.add_argument(
        "-j", "--jobs", 
        type=int, 
        default=psutil.cpu_count(), 
        help="number of parallel jobs"
        )

    args = parser.parse_args()
    show_args(args)

    main(args)
