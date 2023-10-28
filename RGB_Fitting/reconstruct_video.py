import os
import time
import argparse
import torch

from model import ours_fit_model_cropface630resize1024

from utils.visual_utils import Logger
from utils.data_utils import setup_seed
from dataset.fit_dataset import FitDataset
import cv2
from utils.data_utils import setup_seed, tensor2np, np2tensor, draw_mask, draw_landmarks, img3channel, read_img
import numpy as np
from tqdm import tqdm


def main(args):

    fit_model = ours_fit_model_cropface630resize1024.FitModel(cpk_dir=args.checkpoints_dir,
                                                              topo_dir=args.topo_dir,
                                                              texgan_model_name=args.texgan_model_name,
                                                              device=args.device)
    
    dataset_op = FitDataset(lm_detector_path=os.path.join(args.checkpoints_dir, 'lm_model/68lm_detector.pb'),
                            mtcnn_detector_path=os.path.join(args.checkpoints_dir, 'mtcnn_model/mtcnn_model.pb'),
                            parsing_model_pth=os.path.join(args.checkpoints_dir, 'parsing_model/79999_iter.pth'),
                            parsing_resnet18_path=os.path.join(args.checkpoints_dir,
                                                               'resnet_model/resnet18-5c106cde.pth'),
                            lm68_3d_path=os.path.join(args.topo_dir, 'similarity_Lm3D_all.mat'),
                            batch_size=1,
                            device=args.device)   

    basename = os.path.basename(args.input) 
    

    logger = Logger(
        vis_dir=os.path.join(args.output_dir, basename),
        flag=f'MAIN',
        is_tb=True)
    logger.write_txt_log(f'reconstruct video: {os.path.join(args.input)} with model {args.model_input_dir}')


    os.makedirs(args.output_dir, exist_ok=True)

    uv_img = np2tensor(cv2.imread(os.path.join(args.model_input_dir, "stage3_uv.png"))[:, :, [2, 1, 0]])

    cap = cv2.VideoCapture(args.input)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    tmp_mp4 = os.path.join(args.output_dir, "reconstruct_tmp.mp4")
    target_mp4 = os.path.join(args.output_dir, "reconstruct.mp4")

    video_writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (224 * 2, 224))

    success,image = cap.read()
    count = 0
    first_input_data = dataset_op.get_input_data(image)
    width, height, scale, x_offset, y_offset = first_input_data['trans_params']
    x_offset = x_offset[0]
    y_offset = y_offset[0]
    search_width = 224 * 2 / scale
    center_x = x_offset + 224  / 2 / scale
    center_y = y_offset + 224  / 2 / scale
    left = max(int(center_x - search_width / 2), 0)
    right = min(int(center_x + search_width / 2), width - 1)
    top = max(int(center_y - search_width / 2), 0)
    bottom = min(int(center_y + search_width / 2), height - 1)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    

    for i in tqdm(range(frame_count), "Video Frame Handle"):
        # BGR --> RGB
        image = image[top:bottom, left:right, [2, 1, 0]]
        input_data = dataset_op.get_input_data(image, False)
        face = fit_model.render_one_frame(input_data, 224, None, uv_img, logger)
        input = np.clip(tensor2np(input_data['img']), 0, 255).astype(np.uint8)
        face = np.clip(tensor2np(face), 0, 255).astype(np.uint8)
        output = np.concatenate([input, face], 1)[:, :, [2, 1, 0]]
        video_writer.write(output)

        success,image = cap.read()
        if not success:
            break
        count += 1
    
    video_writer.release()

    cmd = f'ffmpeg -i "{tmp_mp4}" -i "{args.input}" -y -map 0:v -map 1:a -c:v copy -c:a copy "{target_mp4}"'
    print(cmd)
    os.system(cmd)
    os.remove(tmp_mp4)
    


if __name__ == '__main__':
    '''Usage
    cd ./RGB_Fitting
    python reconstruct_video.py \
        --input ../mp4s/zhangzhang.mp4 \
        --model_input_dir ../mp4s_outputs/zhangzhang \
        --output_dir ../mp4s_output/zhangzhang \
        --checkpoints_dir ../checkpoints \
        --topo_dir ../topo_assets \
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',
                        type=str,
                        default='../mp4s/zhangzhang.mp4',
                        help='mp4 file path to get face from')
    parser.add_argument('--model_input_dir',
                        type=str,
                        default='../mp4s_outputs/zhangzhang',
                        help='directory of prebuilt face 3d data')
    parser.add_argument('--output_dir',
                        type=str,
                        default='../mp4s_outputs/zhangzhang',
                        help='directory of outputs')
    parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='pretrained models.')
    parser.add_argument('--topo_dir', type=str, default='../topo_assets', help='assets of topo.')
    parser.add_argument('--device', type=str, default='cuda', help='cuda/cpu')
    parser.add_argument('--texgan_model_name', type=str, default='texgan_ffhq_uv.pth', help='texgan model name.')
    args = parser.parse_args()

    setup_seed(123)  # fix random seed
    main(args)

