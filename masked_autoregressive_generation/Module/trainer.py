import os
import torch
import numpy as np
from tqdm import trange
from typing import Union

from ma_sh.Model.mash import Mash

from base_trainer.Module.base_trainer import BaseTrainer

from masked_autoregressive_generation.Dataset.mash import MashDataset
from masked_autoregressive_generation.Dataset.single_shape import SingleShapeDataset
from masked_autoregressive_generation.Model.mar import MAR, mar_base


class Trainer(BaseTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        device: str = "auto",
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        use_dataloader_x: bool = False,
        use_amp: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.mash_channel = 400
        self.mask_degree = 3
        self.sh_degree = 2

        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            device,
            warm_step_num,
            finetune_step_num,
            lr,
            lr_batch_size,
            ema_start_step,
            ema_decay_init,
            ema_decay,
            save_result_folder_path,
            save_log_folder_path,
            best_model_metric_name,
            is_metric_lower_better,
            sample_results_freq,
            use_dataloader_x,
            use_amp,
        )
        return

    def createDatasets(self) -> bool:
        if True:
            mash_file_path = os.environ['HOME'] + '/chLi/Dataset/MashV4/ShapeNet/03636649/583a5a163e59e16da523f74182db8f2.npy'
            self.dataloader_dict['single_shape'] =  {
                'dataset': SingleShapeDataset(mash_file_path),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['mash'] =  {
                'dataset': MashDataset(self.dataset_root_folder_path, 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['image'] =  {
                'dataset': EmbeddingDataset(
                    self.dataset_root_folder_path,
                    {
                        'clip': 'Objaverse_82K/render_clip',
                        'dino': 'Objaverse_82K/render_dino',
                    },
                    'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['points'] =  {
                'dataset': EmbeddingDataset(self.dataset_root_folder_path, 'PointsEmbedding', 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['text'] =  {
                'dataset': EmbeddingDataset(self.dataset_root_folder_path, 'TextEmbedding_ShapeGlot', 'train'),
                'repeat_num': 10,
            }

        if True:
            self.dataloader_dict['eval'] =  {
                'dataset': MashDataset(self.dataset_root_folder_path, 'eval'),
            }

            self.dataloader_dict['eval']['dataset'].paths_list = self.dataloader_dict['eval']['dataset'].paths_list[:64]

        return True

    def createModel(self) -> bool:
        self.model = mar_base(
            anchor_num=self.mash_channel,
            device=self.device,
        ).to(self.device)

        return True

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        if 'category_id' in data_dict.keys():
            data_dict['condition'] = data_dict['category_id']
        elif 'embedding' in data_dict.keys():
            data_dict['condition'] = data_dict['embedding']
        else:
            print('[ERROR][Trainer::toCondition]')
            print('\t valid condition type not found!')
            exit()

        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        gt_latents = data_dict['mash_params'].clone().detach()
        z = result_dict['z']
        mask = result_dict['mask']

        # diffloss
        loss = self.model.module.forward_loss(z=z, target=gt_latents, mask=mask)

        loss_dict = {
            'Loss': loss,
        }

        return loss_dict

    @torch.no_grad()
    def sampleModelStep(self, model: MAR, model_name: str) -> bool:
        sample_gt = True
        sample_num = 3
        dataset = self.dataloader_dict['single_shape']['dataset']

        model.eval()

        data = dataset.__getitem__(0)
        gt_mash = data['mash_params']
        condition = data['category_id']

        print('[INFO][Trainer::sampleModelStep]')
        print("\t start diffuse", sample_num, "mashs....")
        if isinstance(condition, int):
            condition_tensor = torch.ones([sample_num]).long().to(self.device) * condition
        elif isinstance(condition, np.ndarray):
            # condition dim: 1x768
            condition_tensor = torch.from_numpy(condition).type(torch.float32).to(self.device).repeat(sample_num, 1)
        elif isinstance(condition, dict):
            condition_tensor = {}
            for key in condition.keys():
                condition_tensor[key] = condition[key].type(torch.float32).to(self.device).unsqueeze(0).repeat(sample_num, *([1] * condition[key].dim()))
        else:
            print('[ERROR][Trainer::sampleModelStep]')
            print('\t condition type not valid!')
            return False

        sampled_array = model.sample_tokens(
            bsz=sample_num,
            num_iter=64,
            labels=condition_tensor,
            progress=True,
        )

        mash_model = Mash(
            self.mash_channel,
            self.mask_degree,
            self.sh_degree,
            20,
            800,
            0.4,
            dtype=torch.float64,
            device=self.device,
        )

        if sample_gt and not self.gt_sample_added_to_logger:
            gt_mash = dataset.normalizeInverse(gt_mash)

            sh2d = 2 * self.mask_degree + 1
            ortho_poses = gt_mash[:, :6]
            positions = gt_mash[:, 6:9]
            mask_params = gt_mash[:, 9 : 9 + sh2d]
            sh_params = gt_mash[:, 9 + sh2d :]

            mash_model.loadParams(
                mask_params=mask_params,
                sh_params=sh_params,
                positions=positions,
                ortho6d_poses=ortho_poses
            )

            pcd = mash_model.toSamplePcd()

            self.logger.addPointCloud('GT_MASH/gt_mash', pcd, self.step)

            self.gt_sample_added_to_logger = True

        for i in trange(sample_num):
            mash_params = sampled_array[i]

            mash_params = dataset.normalizeInverse(mash_params)

            sh2d = 2 * self.mask_degree + 1
            ortho_poses = mash_params[:, :6]
            positions = mash_params[:, 6:9]
            mask_params = mash_params[:, 9 : 9 + sh2d]
            sh_params = mash_params[:, 9 + sh2d :]

            mash_model.loadParams(
                mask_params=mask_params,
                sh_params=sh_params,
                positions=positions,
                ortho6d_poses=ortho_poses
            )

            pcd = mash_model.toSamplePcd()

            self.logger.addPointCloud(model_name + '/pcd_' + str(i), pcd, self.step)

        return True
