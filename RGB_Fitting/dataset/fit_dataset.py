import torch
import numpy as np
from scipy.io import loadmat
import cv2
from time import strftime
from utils.data_utils import read_img, img3channel, img2mask, np2pillow, pillow2np, nps2tensors, np2tensor
from utils.preprocess_utils import align_img, estimate_norm
from third_party import Landmark68_API, SkinMask_API, FaceParsing_API, FAN


class FitDataset:

    def __init__(self, lm_detector_path, mtcnn_detector_path, parsing_model_pth, parsing_resnet18_path, lm68_3d_path,
                 batch_size, device):
        self.lm68_model = Landmark68_API(lm_detector_path=lm_detector_path, mtcnn_path=mtcnn_detector_path)
        self.fan_model = FAN()
        self.skin_model = SkinMask_API()
        self.parsing_model = FaceParsing_API(parsing_pth=parsing_model_pth,
                                             resnet18_path=parsing_resnet18_path,
                                             device=device)
        self.lm68_3d = loadmat(lm68_3d_path)['lm']
        self.batch_size = batch_size
        self.device = device

    def get_input_data(self, img, need_skin_mask=True):
        with torch.no_grad():
            if isinstance(img, str):
                input_img = read_img(img)
            else:
                input_img = img

            raw_img = np2pillow(input_img)

            # detect 68 landmarks
            # raw_lm = self.lm68_model(input_img)
            raw_lm = self.fan_model(input_img)
            if raw_lm is None:
                return None
                
            raw_lm = raw_lm.astype(np.float32)

            # calculate skin attention mask
            if need_skin_mask:
                raw_skin_mask = self.skin_model(input_img, return_uint8=True)
                raw_skin_mask = img3channel(raw_skin_mask)
                raw_skin_mask = np2pillow(raw_skin_mask)
                # face parsing mask
                require_part = ['face', 'l_eye', 'r_eye', 'mouth']
                seg_mask_dict, _ = self.parsing_model(input_img, require_part=require_part)
                face_mask = seg_mask_dict['face']
                ex_mouth_mask = 1 - seg_mask_dict['mouth']
                ex_eye_mask = 1 - img2mask(seg_mask_dict['l_eye'] + seg_mask_dict['r_eye'], thre=0.5)
                raw_parse_mask = face_mask * ex_mouth_mask * ex_eye_mask
                raw_parse_mask = np2pillow(raw_parse_mask, src_range=1.0)
                # alignment
                trans_params, img, lm, skin_mask, parse_mask, mouth_mask = align_img(raw_img, raw_lm, self.lm68_3d, raw_skin_mask,
                                                                        raw_parse_mask, np2pillow(seg_mask_dict['mouth'], src_range=1.0))
            else:
                raw_skin_mask = None
                # alignment
                trans_params, img, lm, skin_mask, parse_mask, mouth_mask = align_img(raw_img, raw_lm, self.lm68_3d, None,
                                                                        None, None)



            # to tensor
            _, H = img.size
            M = estimate_norm(lm, H)
            img_tensor = np2tensor(pillow2np(img), device=self.device)
            if need_skin_mask:
                skin_mask_tensor = np2tensor(pillow2np(skin_mask), device=self.device)[:, :1, :, :]
                parse_mask_tensor = np2tensor(pillow2np(parse_mask), device=self.device)[:, :1, :, :]
                mouth_mask_tensor = np2tensor(pillow2np(mouth_mask), device=self.device)[:, :1, :, :]
            else:
                skin_mask_tensor = None
                parse_mask_tensor = None
                mouth_mask_tensor = None
            lm_tensor = torch.tensor(np.array(lm).astype(np.float32)).unsqueeze(0).to(self.device)
            M_tensor = torch.tensor(np.array(M).astype(np.float32)).unsqueeze(0).to(self.device)

            return {
                'img': img_tensor,
                'skin_mask': skin_mask_tensor,
                'parse_mask': parse_mask_tensor,
                'lm': lm_tensor,
                'M': M_tensor,
                'trans_params': trans_params,
                'mouth_mask': mouth_mask_tensor
            }

    def get_mp4_data(self, mp4_path, mask=False):
        with torch.no_grad():
            cap = cv2.VideoCapture(mp4_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            mp4_array_result = {
                'img': [],
                'skin_mask': [],
                'parse_mask': [],
                'lm': [],
                'M': [],
                'trans_params': []
            }
            frame_index = 0
            while cap.isOpened():
                print(f"{strftime('%H:%M:%S')} handling frame_index={frame_index}")
                if frame_index >= fps:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frame_index += 1
                # BGR --> RGB
                input_img = frame[:, :, [2, 1, 0]]
                raw_img = np2pillow(input_img)
                # detect 68 landmarks
                raw_lm = self.lm68_model(input_img)
                if raw_lm is None:
                    return None
                    
                raw_lm = raw_lm.astype(np.float32)

                # calculate skin attention mask
                if mask:
                    raw_skin_mask = self.skin_model(input_img, return_uint8=True)
                    raw_skin_mask = img3channel(raw_skin_mask)
                    raw_skin_mask = np2pillow(raw_skin_mask)

                    # face parsing mask
                    require_part = ['face', 'l_eye', 'r_eye', 'mouth']
                    seg_mask_dict, _ = self.parsing_model(input_img, require_part=require_part)
                    face_mask = seg_mask_dict['face']
                    ex_mouth_mask = 1 - seg_mask_dict['mouth']
                    ex_eye_mask = 1 - img2mask(seg_mask_dict['l_eye'] + seg_mask_dict['r_eye'], thre=0.5)
                    raw_parse_mask = face_mask * ex_mouth_mask * ex_eye_mask
                    raw_parse_mask = np2pillow(raw_parse_mask, src_range=1.0)
                else:
                    raw_skin_mask = None
                    raw_parse_mask = None

                # alignment
                #trans_params, img, lm, skin_mask, parse_mask = align_img(raw_img, raw_lm, self.lm68_3d, raw_skin_mask,
                #                                                        raw_parse_mask)
                trans_params, img, lm, skin_mask, parse_mask = align_img(raw_img, raw_lm, self.lm68_3d, raw_skin_mask,
                                         raw_parse_mask)
                M = np.array(estimate_norm(lm, height)).astype(np.float32)
                lm = np.array(lm).astype(np.float32)
                mp4_array_result['img'].append(pillow2np(img))
                if mask:
                    mp4_array_result['skin_mask'].append(pillow2np(skin_mask)[:, :, :1])
                    mp4_array_result['parse_mask'].append(pillow2np(parse_mask)[:, :, :1])
                mp4_array_result['lm'].append(lm)
                mp4_array_result['M'].append(M)
                mp4_array_result['trans_params'].append(trans_params)
            
            mp4_result = {}
            mp4_result['img'] = nps2tensors(np.array(mp4_array_result['img']))
            if mask:
                mp4_result['skin_mask'] = nps2tensors(np.array(mp4_array_result['skin_mask']))
                mp4_result['parse_mask'] = nps2tensors(np.array(mp4_array_result['parse_mask']))
            mp4_result['lm'] = torch.tensor(np.array(mp4_array_result['lm']))
            mp4_result['M'] = torch.tensor(np.array(mp4_array_result['M']))
            mp4_result['trans_params'] = mp4_array_result['trans_params']

        return mp4_result
            

                





