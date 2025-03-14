# Copyright (c) OpenRobotLab. All rights reserved.
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from terminaltables import AsciiTable
from mmdet.evaluation import eval_map
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process, get_rank)
from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu
from mmengine.logging import MMLogger, print_log

from bip3d.registry import METRICS
from bip3d.structures import get_box_type

from ..indoor_eval import indoor_eval


@METRICS.register_module()
class IndoorDetMetric(BaseMetric):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 iou_thr: List[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 batchwise_anns: bool = False,
                 save_result_path=None,
                 eval_part=None,
                 **kwargs):
        super(IndoorDetMetric, self).__init__(prefix=prefix,
                                              collect_device=collect_device,
                                              **kwargs)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        self.batchwise_anns = batchwise_anns
        self.save_result_path = save_result_path
        if eval_part is None:
            eval_part = ["scannet", "3rscan", "matterport3d", "arkit"]
        self.eval_part = eval_part

    def process(self, data_batch: dict, data_samples: Sequence[dict]):
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            eval_ann_info = data_sample['eval_ann_info']
            eval_ann_info["scan_id"] = data_sample["scan_id"]
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def compute_metrics(self, results: list, part="Overall", split=True):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        ann_infos = []
        pred_results = []

        for eval_ann, sinlge_pred_results in results:
            ann_infos.append(eval_ann)
            pred_results.append(sinlge_pred_results)

        # some checkpoints may not record the key "box_type_3d"
        box_type_3d, box_mode_3d = get_box_type(
            self.dataset_meta.get('box_type_3d', 'depth'))

        print_log(f"eval : {len(ann_infos)}, {len(pred_results)}, rank: {get_rank()}")
        ret_dict = indoor_eval(ann_infos,
                               pred_results,
                               self.iou_thr,
                               self.dataset_meta['classes'],
                               logger=logger,
                               box_mode_3d=box_mode_3d,
                               classes_split=self.dataset_meta.get(
                                   'classes_split', None) if split else None,
                               part=part)

        return ret_dict

    def evaluate(self, size: int):
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)
        print_log(
            f"number of results: "
            f"{len(self.results)}, size: {size}, "
            f"batchwise_anns: {self.batchwise_anns}, rank: {get_rank()}, "
            f"collect_dir: {self.collect_dir}"
        )
        if self.batchwise_anns:
            # the actual dataset length/size is the len(self.results)
            if self.collect_device == 'cpu':
                results = collect_results(self.results,
                                          len(self.results),
                                          self.collect_device,
                                          tmpdir=self.collect_dir)
            else:
                results = collect_results(self.results, len(self.results),
                                          self.collect_device)
        else:
            if self.collect_device == 'cpu':
                results = collect_results(self.results,
                                          size,
                                          self.collect_device,
                                          tmpdir=self.collect_dir)
            else:
                results = collect_results(self.results, size,
                                          self.collect_device)

        print_log(
            f"number of collected results: "
            f"{len(results) if results is not None else 'None'},"
            f"rank: {get_rank()}"
        )
        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            # import pdb; pdb.set_trace()
            if self.save_result_path is not None:
                import pickle
                import copy
                _ret = []
                for x in results:
                    x = list(copy.deepcopy(x))
                    x[0]["gt_bboxes_3d"] = x[0]["gt_bboxes_3d"].cpu().numpy()
                    x[1]["bboxes_3d"] = x[1]["bboxes_3d"].cpu().numpy()
                    x[1]["scores_3d"] = x[1]["scores_3d"].cpu().numpy()
                    x[1]["labels_3d"] = x[1]["labels_3d"].cpu().numpy()
                    x[1]["target_scores_3d"] = x[1]["target_scores_3d"].cpu().numpy()
                    _ret.append(x)
                pickle.dump(_ret, open(os.path.join(self.save_result_path, "results.pkl"), "wb"))

            _metrics = self.compute_metrics(results)  # type: ignore
            for part in self.eval_part:
            # for part in []:
                print(f"=================== {part} =========================")
                part_ret = [x for x in results if part in x[0]["scan_id"]]
                if len(part_ret) > 0:
                    _metrics_part = self.compute_metrics(part_ret, part=part)
                    _metrics_part = {f"{part}--{k}": v for k, v in _metrics_part.items()}
                    _metrics.update(_metrics_part)
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        if is_main_process():
            table = []
            header = ["Summary"]
            for i, iou_thresh in enumerate(self.iou_thr):
                header.append(f'AP_{iou_thresh:.2f}')
                header.append(f'AR_{iou_thresh:.2f}')
            table.append(header)

            for part in self.eval_part + ["Overall"]:
                table_data = [part]
                for i, iou_thresh in enumerate(self.iou_thr):
                    key = f"{part+'--' if part!='Overall' else ''}mAP_{iou_thresh:.2f}"
                    if key in _metrics:
                        table_data.append(f"{_metrics[key]:.4f}")
                    else:
                        table_data.append(f"0.xxxx")
                    key = f"{part+'--' if part!='Overall' else ''}mAR_{iou_thresh:.2f}"
                    if key in _metrics:
                        table_data.append(f"{_metrics[key]:.4f}")
                    else:
                        table_data.append(f"0.xxxx")
                table.append(table_data)
            table = AsciiTable(table)
            table.inner_footing_row_border = True
            logger: MMLogger = MMLogger.get_current_instance()
            print_log('\n' + table.table, logger=logger)

        # reset the results list
        self.results.clear()
        return metrics[0]


@METRICS.register_module()
class Indoor2DMetric(BaseMetric):
    """indoor 2d predictions evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 iou_thr: Union[float, List[float]] = [0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super(Indoor2DMetric, self).__init__(prefix=prefix,
                                             collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            eval_ann_info = data_sample['eval_ann_info']
            ann = dict(labels=eval_ann_info['gt_bboxes_labels'],
                       bboxes=eval_ann_info['gt_bboxes'])

            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)

            self.results.append((ann, dets))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        annotations, preds = zip(*results)
        eval_results = OrderedDict()
        for iou_thr_2d_single in self.iou_thr:
            mean_ap, _ = eval_map(preds,
                                  annotations,
                                  scale_ranges=None,
                                  iou_thr=iou_thr_2d_single,
                                  dataset=self.dataset_meta['classes'],
                                  logger=logger)
            eval_results['mAP_' + str(iou_thr_2d_single)] = mean_ap
        return eval_results
