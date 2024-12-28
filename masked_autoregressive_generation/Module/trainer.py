import os
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm, trange
from copy import deepcopy
from typing import Union
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from masked_autoregressive_generation.Dataset.mash import MashDataset
from masked_autoregressive_generation.Dataset.single_shape import SingleShapeDataset
from masked_autoregressive_generation.Model.mar import MAR, mar_base
from masked_autoregressive_generation.Method.path import createFileFolder, renameFile, removeFile
from masked_autoregressive_generation.Method.time import getCurrentTime
from masked_autoregressive_generation.Module.logger import Logger

from ma_sh.Model.mash import Mash


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def check_and_replace_nan_in_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN detected in gradient: {name}")
            param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad)
    return True

class Trainer(object):
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
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.local_rank = setup_distributed()

        self.mash_channel = 400
        self.mask_degree = 3
        self.sh_degree = 2
        self.context_dim = 512
        self.n_heads = 8
        self.d_head = 64
        self.depth = 24

        self.accum_iter = accum_iter
        if device == 'auto':
            self.device = torch.device('cuda:' + str(self.local_rank))
        else:
            self.device = device
        self.warm_step_num = warm_step_num / accum_iter
        self.finetune_step_num = finetune_step_num
        self.lr = lr * batch_size / 256 * self.accum_iter * dist.get_world_size()
        self.ema_start_step = ema_start_step
        self.ema_decay_init = ema_decay_init
        self.ema_decay = ema_decay

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path

        self.step = 0
        self.epoch = 0
        self.loss_dict_list = []

        self.logger = None
        if self.local_rank == 0:
            self.logger = Logger()

        self.dataloader_dict = {}

        if False:
            mash_file_path = os.environ['HOME'] + '/Dataset/MashV4/ShapeNet/03636649/583a5a163e59e16da523f74182db8f2.npy'
            self.dataloader_dict['single_shape'] =  {
                'dataset': SingleShapeDataset(mash_file_path),
                'repeat_num': 1,
            }

        if True:
            self.dataloader_dict['mash'] =  {
                'dataset': MashDataset(dataset_root_folder_path, 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['image'] =  {
                'dataset': EmbeddingDataset(
                    dataset_root_folder_path,
                    {
                        'clip': 'Objaverse_82K/render_clip',
                        'dino': 'Objaverse_82K/render_dino',
                    },
                    'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['points'] =  {
                'dataset': EmbeddingDataset(dataset_root_folder_path, 'PointsEmbedding', 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['text'] =  {
                'dataset': EmbeddingDataset(dataset_root_folder_path, 'TextEmbedding_ShapeGlot', 'train'),
                'repeat_num': 10,
            }

        if True:
            self.dataloader_dict['eval'] =  {
                'dataset': MashDataset(dataset_root_folder_path, 'eval'),
            }

        for key, item in self.dataloader_dict.items():
            if key == 'eval':
                self.dataloader_dict[key]['dataloader'] = DataLoader(
                    item['dataset'],
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                continue

            self.dataloader_dict[key]['sampler'] = DistributedSampler(item['dataset'])
            self.dataloader_dict[key]['dataloader'] = DataLoader(
                item['dataset'],
                sampler=self.dataloader_dict[key]['sampler'],
                batch_size=batch_size,
                num_workers=num_workers,
            )

        self.channels = int(
            9 + (2 * self.mask_degree + 1) + ((self.sh_degree + 1) ** 2)
        )
        self.model = mar_base(anchor_num=self.mash_channel).to(self.device)

        if self.local_rank == 0:
            self.ema_model = deepcopy(self.model)
            self.ema_loss = None

        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        if model_file_path is not None:
            self.loadModel(model_file_path)

        self.optim = AdamW(self.model.parameters(), lr=self.lr)
        self.sched = LambdaLR(self.optim, lr_lambda=self.warmup_lr)

        self.initRecords()

        self.gt_sample_added_to_logger = False
        return

    def initRecords(self) -> bool:
        if self.logger is None:
            return True

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Trainer::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_state_dict = torch.load(model_file_path, map_location='cpu')
        if 'model' in model_state_dict.keys():
            self.model.module.load_state_dict(model_state_dict["model"])
        if 'step' in model_state_dict.keys():
            self.step = model_state_dict['step']

        if self.local_rank == 0:
            if 'ema_model' in model_state_dict.keys():
                self.ema_model.load_state_dict(model_state_dict["ema_model"])
            if 'ema_loss' in model_state_dict.keys():
                self.ema_loss = model_state_dict['ema_loss']

        print('[INFO][Trainer::loadModel]')
        print('\t model loaded from:', model_file_path)

        return True

    def getLr(self) -> float:
        return self.optim.state_dict()["param_groups"][0]["lr"]

    def warmup_lr(self, step: int) -> float:
        if self.warm_step_num == 0:
            return 1.0

        return min(step, self.warm_step_num) / self.warm_step_num

    def toEMADecay(self) -> float:
        if self.step <= self.ema_start_step:
            return self.ema_decay_init + self.step / self.ema_start_step * (self.ema_decay - self.ema_decay_init)

        return self.ema_decay

    def ema(self) -> bool:
        ema_decay = self.toEMADecay()

        source_dict = self.model.module.state_dict()
        target_dict = self.ema_model.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * ema_decay + source_dict[key].data * (1 - ema_decay)
            )
        return True

    def trainStep(
        self,
        data: dict,
    ) -> dict:
        self.model.train()

        mash_params = data['mash_params'].to(self.device)
        condition = data['condition']
        if isinstance(condition, torch.Tensor):
            condition = condition.to(self.device)
        else:
            for key in condition.keys():
                condition[key] = condition[key].to(self.device)

        loss = self.model(mash_params, condition)

        loss_item = loss.clone().detach().cpu().numpy()

        accum_loss = loss / self.accum_iter

        accum_loss.backward()

        if not check_and_replace_nan_in_grad(self.model):
            print('[ERROR][Trainer::trainStep]')
            print('\t check_and_replace_nan_in_grad failed!')
            exit()

        if (self.step + 1) % self.accum_iter == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.sched.step()
            if self.local_rank == 0:
                self.ema()
            self.optim.zero_grad()

        loss_dict = {
            "Loss": loss_item,
        }

        return loss_dict

    @torch.no_grad()
    def evalStep(
        self,
        data: dict,
    ) -> dict:
        self.model.eval()

        mash_params = data['mash_params'].to(self.device)
        condition = data['condition']
        if isinstance(condition, torch.Tensor):
            condition = condition.to(self.device)
        else:
            for key in condition.keys():
                condition[key] = condition[key].to(self.device)

        model_loss = self.model.module(mash_params, condition)

        model_loss_item = model_loss.clone().detach().cpu().numpy()

        ema_loss = self.ema_model(mash_params, condition)

        ema_loss_item = ema_loss.clone().detach().cpu().numpy()

        loss_dict = {
            "ModelLoss": model_loss_item,
            "EMALoss": ema_loss_item,
        }

        return loss_dict

    @torch.no_grad()
    def sampleModelStep(self, model: MAR, model_name: str) -> bool:
        sample_gt = False
        sample_num = 3
        dataset = self.dataloader_dict['mash']['dataset']

        model.eval()

        data = dataset.__getitem__(0)
        gt_mash = data['mash_params']
        condition = data['category_id']

        if sample_gt:
            gt_mash = dataset.normalizeInverse(gt_mash)

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
        )

        print(sampled_array.shape)
        exit()

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

    @torch.no_grad()
    def sampleStep(self) -> bool:
        if self.local_rank != 0:
            return True

        self.sampleModelStep(self.model.module, 'Model')
        return True

    @torch.no_grad()
    def sampleEMAStep(self) -> bool:
        if self.local_rank != 0:
            return True

        self.sampleModelStep(self.ema_model, 'EMA')
        return True

    def toCondition(self, data: dict) -> Union[torch.Tensor, None]:
        if 'category_id' in data.keys():
            return data['category_id']

        if 'embedding' in data.keys():
            return data["embedding"]

        print('[ERROR][Trainer::toCondition]')
        print('\t valid condition type not found!')
        return None

    def trainEpoch(self, data_name: str) -> bool:
        if data_name not in self.dataloader_dict.keys():
            print('[ERROR][Trainer::trainEpoch]')
            print('\t data not exist!')
            print('\t data_name:', data_name)
            return False

        dataloader_dict = self.dataloader_dict[data_name]
        dataloader_dict['sampler'].set_epoch(self.epoch)

        dataloader = dataloader_dict['dataloader']

        if self.local_rank == 0:
            pbar = tqdm(total=len(dataloader))
        for data in dataloader:
            condition = self.toCondition(data)
            if condition is None:
                print('[ERROR][Trainer::trainEpoch]')
                print('\t toCondition failed!')
                continue

            conditional_data = {
                'mash_params': data['mash_params'],
                'condition': condition,
            }

            train_loss_dict = self.trainStep(conditional_data)

            self.loss_dict_list.append(train_loss_dict)

            lr = self.getLr()

            if (self.step + 1) % self.accum_iter == 0 and self.local_rank == 0:
                for key in train_loss_dict.keys():
                    value = 0
                    for i in range(len(self.loss_dict_list)):
                        value += self.loss_dict_list[i][key]
                    value /= len(self.loss_dict_list)
                    self.logger.addScalar("Train/" + key, value, self.step)
                self.logger.addScalar("Train/Lr", lr, self.step)

                if self.ema_loss is None:
                    self.ema_loss = train_loss_dict["Loss"]
                else:
                    ema_decay = self.toEMADecay()

                    self.ema_loss = self.ema_loss * ema_decay + train_loss_dict["Loss"] * (1 - ema_decay)
                self.logger.addScalar("Train/EMALoss", self.ema_loss, self.step)

                self.loss_dict_list = []

            if self.local_rank == 0:
                pbar.set_description(
                    "EPOCH %d LOSS %.6f LR %.4f"
                    % (
                        self.epoch,
                        train_loss_dict["Loss"],
                        self.getLr() / self.lr,
                    )
                )

            self.step += 1

            if self.local_rank == 0:
                pbar.update(1)

        if self.local_rank == 0:
            pbar.close()

        self.epoch += 1

        return True

    @torch.no_grad()
    def evalEpoch(self) -> bool:
        if self.local_rank != 0:
            return True

        if 'eval' not in self.dataloader_dict.keys():
            return True

        print('[INFO][Trainer::evalEpoch]')
        print('\t start evaluating ...')

        dataloader = self.dataloader_dict['eval']['dataloader']

        for data in dataloader:
            condition = self.toCondition(data)
            if condition is None:
                print('[ERROR][Trainer::evalEpoch]')
                print('\t toCondition failed!')
                continue

            conditional_data = {
                'mash_params': data['mash_params'],
                'condition': condition,
            }

            eval_loss_dict = self.evalStep(conditional_data)

            for key, item in eval_loss_dict.items():
                self.logger.addScalar("Eval/" + key, item, self.step)

            break

        return True

    def train(self) -> bool:
        final_step = self.step + self.finetune_step_num

        if self.local_rank == 0:
            print("[INFO][Trainer::train]")
            print("\t start training ...")

        while self.step < final_step or self.finetune_step_num < 0:

            for data_name in self.dataloader_dict.keys():
                if data_name == 'eval':
                    continue

                repeat_num = self.dataloader_dict[data_name]['repeat_num']

                for i in range(repeat_num):
                    if self.local_rank == 0:
                        print('[INFO][Trainer::train]')
                        print('\t start training on dataset [', data_name, '] ,', i + 1, '/', repeat_num, '...')

                    if not self.trainEpoch(data_name):
                        print('[ERROR][Trainer::train]')
                        print('\t trainEpoch failed!')
                        return False

                    self.autoSaveModel("total")

                    if not self.evalEpoch():
                        print('[ERROR][Trainer::train]')
                        print('\t evalEpoch failed!')
                        return False

                    if self.epoch % 50 == 0:
                        self.sampleStep()
                        self.sampleEMAStep()

        return True

    def saveModel(self, save_model_file_path: str) -> bool:
        createFileFolder(save_model_file_path)

        model_state_dict = {
            "model": self.model.module.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "ema_loss": self.ema_loss,
            "step": self.step,
        }

        torch.save(model_state_dict, save_model_file_path)

        return True

    def autoSaveModel(self, name: str) -> bool:
        if self.local_rank != 0:
            return True

        if self.save_result_folder_path is None:
            return False

        save_last_model_file_path = self.save_result_folder_path + name + "_model_last.pth"

        tmp_save_last_model_file_path = save_last_model_file_path[:-4] + "_tmp.pth"

        self.saveModel(tmp_save_last_model_file_path)

        removeFile(save_last_model_file_path)
        renameFile(tmp_save_last_model_file_path, save_last_model_file_path)

        return True
