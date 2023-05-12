# Copyright (c) Open-CD. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

#from mmengine.model import BaseModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
#from mmengine.structures import PixelData
from torch import Tensor, nn

# from mmseg.models import builder
from mmseg.ops import resize
#from mmseg.structures import SegDataSample
from mmseg.utils import ConfigType, SampleList, add_prefix
#from opencd.registry import MODELS
from mmseg.models.builder import HEADS
from addict import Dict
from typing import Dict, List, Optional, Sequence, Tuple, Union

class ConfigDict(Dict):
    """A dictionary for config which has the same interface as python's built-
    in dictionary and can be used as a normal dictionary.

    The Config class would transform the nested fields (dictionary-like fields)
    in config file into ``ConfigDict``.
    """

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no "
                                 f"attribute '{name}'")
        except Exception as e:
            raise e
        else:
            return value
        
ConfigType = Union[ConfigDict, dict]

@HEADS.register_module()
class MultiHeadDecoder(BaseDecodeHead):
    """Base class for MultiHeadDecoder.

    Args:
        binary_cd_head (dict): The decode head for binary change detection branch.
        binary_cd_neck (dict): The feature fusion part for binary \
            change detection branch
        semantic_cd_head (dict): The decode head for semantic change \
            detection `from` branch.
        semantic_cd_head_aux (dict): The decode head for semantic change \
            detection `to` branch. If None, the siamese semantic head will \
            be used. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 binary_cd_head,
                 binary_cd_neck=None,
                 semantic_cd_head=None,
                 semantic_cd_head_aux=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.binary_cd_head = MODELS.build(binary_cd_head)
        self.siamese_semantic_head = True
        if binary_cd_neck is not None:
            self.binary_cd_neck = MODELS.build(binary_cd_neck)
        if semantic_cd_head is not None:
            self.semantic_cd_head = MODELS.build(semantic_cd_head)
            if semantic_cd_head_aux is not None:
                self.siamese_semantic_head = False
                self.semantic_cd_head_aux = MODELS.build(semantic_cd_head_aux)
            else:
                self.semantic_cd_head_aux = self.semantic_cd_head

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function.
        The return value should be a dict() containing: 
        `seg_logits`, `seg_logits_from` and `seg_logits_to`.
        
        For example:
            return dict(
                seg_logits=out,
                seg_logits_from=out1, 
                seg_logits_to=out2)
        """
        pass

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses
    
    def predict(self, inputs, batch_img_metas: List[dict], test_cfg,
                **kwargs) -> List[Tensor]:
        """Forward function for testing."""
        seg_logits = self.forward(inputs)
        return self.predict_by_feat(seg_logits, batch_img_metas, **kwargs)
        
    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        assert ['seg_logits', 'seg_logits_from', 'seg_logits_to'] \
            == list(seg_logits.keys()), "`seg_logits`, `seg_logits_from` \
            and `seg_logits_to` should be contained."

        self.align_corners = {
            'seg_logits': self.binary_cd_head.align_corners,
            'seg_logits_from': self.semantic_cd_head.align_corners,
            'seg_logits_to': self.semantic_cd_head_aux.align_corners}

        for seg_name, seg_logit in seg_logits.items():
            seg_logits[seg_name] = resize(
                input=seg_logit,
                size=batch_img_metas[0]['img_shape'],
                mode='bilinear',
                align_corners=self.align_corners[seg_name])
        return seg_logits
    
    def get_sub_batch_data_samples(self, batch_data_samples: SampleList, 
                                   sub_metainfo_name: str,
                                   sub_data_name: str) -> list:
        sub_batch_sample_list = []
        for i in range(len(batch_data_samples)):
            data_sample = SegDataSample()

            gt_sem_seg_data = dict(
                data=batch_data_samples[i].get(sub_data_name).data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

            img_meta = {}
            seg_map_path = batch_data_samples[i].metainfo.get(sub_metainfo_name)
            for key in batch_data_samples[i].metainfo.keys():
                if not 'seg_map_path' in key:
                    img_meta[key] = batch_data_samples[i].metainfo.get(key)
            img_meta['seg_map_path'] = seg_map_path
            data_sample.set_metainfo(img_meta)

            sub_batch_sample_list.append(data_sample)
        return sub_batch_sample_list

    def loss_by_feat(self, seg_logits: dict,
                     batch_data_samples: SampleList, **kwargs) -> dict:
        """Compute segmentation loss."""
        assert ['seg_logits', 'seg_logits_from', 'seg_logits_to'] \
            == list(seg_logits.keys()), "`seg_logits`, `seg_logits_from` \
            and `seg_logits_to` should be contained."

        losses = dict()
        binary_cd_loss_decode = self.binary_cd_head.loss_by_feat(
            seg_logits['seg_logits'],
            self.get_sub_batch_data_samples(batch_data_samples,
                                            sub_metainfo_name='seg_map_path',
                                            sub_data_name='gt_sem_seg'))
        losses.update(add_prefix(binary_cd_loss_decode, 'binary_cd'))

        if getattr(self, 'semantic_cd_head'):
            semantic_cd_loss_decode_from = self.semantic_cd_head.loss_by_feat(
                seg_logits['seg_logits_from'],
                self.get_sub_batch_data_samples(batch_data_samples,
                                                sub_metainfo_name='seg_map_path_from',
                                                sub_data_name='gt_sem_seg_from'))
            losses.update(add_prefix(semantic_cd_loss_decode_from, 'semantic_cd_from'))

            semantic_cd_loss_decode_to = self.semantic_cd_head_aux.loss_by_feat(
                seg_logits['seg_logits_to'],
                self.get_sub_batch_data_samples(batch_data_samples,
                                                sub_metainfo_name='seg_map_path_to',
                                                sub_data_name='gt_sem_seg_to'))
            losses.update(add_prefix(semantic_cd_loss_decode_to, 'semantic_cd_to'))

        return losses
    
class BaseDataElement:
    """A base data interface that supports Tensor-like and dict-like
    operations.

    A typical data elements refer to predicted results or ground truth labels
    on a task, such as predicted bboxes, instance masks, semantic
    segmentation masks, etc. Because groundtruth labels and predicted results
    often have similar properties (for example, the predicted bboxes and the
    groundtruth bboxes), MMEngine uses the same abstract data interface to
    encapsulate predicted results and groundtruth labels, and it is recommended
    to use different name conventions to distinguish them, such as using
    ``gt_instances`` and ``pred_instances`` to distinguish between labels and
    predicted results. Additionally, we distinguish data elements at instance
    level, pixel level, and label level. Each of these types has its own
    characteristics. Therefore, MMEngine defines the base class
    ``BaseDataElement``, and implement ``InstanceData``, ``PixelData``, and
    ``LabelData`` inheriting from ``BaseDataElement`` to represent different
    types of ground truth labels or predictions.

    Another common data element is sample data. A sample data consists of input
    data (such as an image) and its annotations and predictions. In general,
    an image can have multiple types of annotations and/or predictions at the
    same time (for example, both pixel-level semantic segmentation annotations
    and instance-level detection bboxes annotations). All labels and
    predictions of a training sample are often passed between Dataset, Model,
    Visualizer, and Evaluator components. In order to simplify the interface
    between components, we can treat them as a large data element and
    encapsulate them. Such data elements are generally called XXDataSample in
    the OpenMMLab. Therefore, Similar to `nn.Module`, the `BaseDataElement`
    allows `BaseDataElement` as its attribute. Such a class generally
    encapsulates all the data of a sample in the algorithm library, and its
    attributes generally are various types of data elements. For example,
    MMDetection is assigned by the BaseDataElement to encapsulate all the data
    elements of the sample labeling and prediction of a sample in the
    algorithm library.

    The attributes in ``BaseDataElement`` are divided into two parts,
    the ``metainfo`` and the ``data`` respectively.

        - ``metainfo``: Usually contains the
          information about the image such as filename,
          image_shape, pad_shape, etc. The attributes can be accessed or
          modified by dict-like or object-like operations, such as
          ``.`` (for data access and modification), ``in``, ``del``,
          ``pop(str)``, ``get(str)``, ``metainfo_keys()``,
          ``metainfo_values()``, ``metainfo_items()``, ``set_metainfo()`` (for
          set or change key-value pairs in metainfo).

        - ``data``: Annotations or model predictions are
          stored. The attributes can be accessed or modified by
          dict-like or object-like operations, such as
          ``.``, ``in``, ``del``, ``pop(str)``, ``get(str)``, ``keys()``,
          ``values()``, ``items()``. Users can also apply tensor-like
          methods to all :obj:`torch.Tensor` in the ``data_fields``,
          such as ``.cuda()``, ``.cpu()``, ``.numpy()``, ``.to()``,
          ``to_tensor()``, ``.detach()``.

    Args:
        metainfo (dict, optional): A dict contains the meta information
            of single image, such as ``dict(img_shape=(512, 512, 3),
            scale_factor=(1, 1, 1, 1))``. Defaults to None.
        kwargs (dict, optional): A dict contains annotations of single image or
            model predictions. Defaults to None.

    Examples:
        >>> import torch
        >>> from mmengine.structures import BaseDataElement
        >>> gt_instances = BaseDataElement()
        >>> bboxes = torch.rand((5, 4))
        >>> scores = torch.rand((5,))
        >>> img_id = 0
        >>> img_shape = (800, 1333)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=img_shape),
        ...     bboxes=bboxes, scores=scores)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=(640, 640)))

        >>> # new
        >>> gt_instances1 = gt_instances.new(
        ...     metainfo=dict(img_id=1, img_shape=(640, 640)),
        ...                   bboxes=torch.rand((5, 4)),
        ...                   scores=torch.rand((5,)))
        >>> gt_instances2 = gt_instances1.new()

        >>> # add and process property
        >>> gt_instances = BaseDataElement()
        >>> gt_instances.set_metainfo(dict(img_id=9, img_shape=(100, 100)))
        >>> assert 'img_shape' in gt_instances.metainfo_keys()
        >>> assert 'img_shape' in gt_instances
        >>> assert 'img_shape' not in gt_instances.keys()
        >>> assert 'img_shape' in gt_instances.all_keys()
        >>> print(gt_instances.img_shape)
        (100, 100)
        >>> gt_instances.scores = torch.rand((5,))
        >>> assert 'scores' in gt_instances.keys()
        >>> assert 'scores' in gt_instances
        >>> assert 'scores' in gt_instances.all_keys()
        >>> assert 'scores' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.scores)
        tensor([0.5230, 0.7885, 0.2426, 0.3911, 0.4876])
        >>> gt_instances.bboxes = torch.rand((5, 4))
        >>> assert 'bboxes' in gt_instances.keys()
        >>> assert 'bboxes' in gt_instances
        >>> assert 'bboxes' in gt_instances.all_keys()
        >>> assert 'bboxes' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.bboxes)
        tensor([[0.0900, 0.0424, 0.1755, 0.4469],
                [0.8648, 0.0592, 0.3484, 0.0913],
                [0.5808, 0.1909, 0.6165, 0.7088],
                [0.5490, 0.4209, 0.9416, 0.2374],
                [0.3652, 0.1218, 0.8805, 0.7523]])

        >>> # delete and change property
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=0, img_shape=(640, 640)),
        ...     bboxes=torch.rand((6, 4)), scores=torch.rand((6,)))
        >>> gt_instances.set_metainfo(dict(img_shape=(1280, 1280)))
        >>> gt_instances.img_shape  # (1280, 1280)
        >>> gt_instances.bboxes = gt_instances.bboxes * 2
        >>> gt_instances.get('img_shape', None)  # (1280, 1280)
        >>> gt_instances.get('bboxes', None)  # 6x4 tensor
        >>> del gt_instances.img_shape
        >>> del gt_instances.bboxes
        >>> assert 'img_shape' not in gt_instances
        >>> assert 'bboxes' not in gt_instances
        >>> gt_instances.pop('img_shape', None)  # None
        >>> gt_instances.pop('bboxes', None)  # None

        >>> # Tensor-like
        >>> cuda_instances = gt_instances.cuda()
        >>> cuda_instances = gt_instances.to('cuda:0')
        >>> cpu_instances = cuda_instances.cpu()
        >>> cpu_instances = cuda_instances.to('cpu')
        >>> fp16_instances = cuda_instances.to(
        ...     device=None, dtype=torch.float16, non_blocking=False,
        ...     copy=False, memory_format=torch.preserve_format)
        >>> cpu_instances = cuda_instances.detach()
        >>> np_instances = cpu_instances.numpy()

        >>> # print
        >>> metainfo = dict(img_shape=(800, 1196, 3))
        >>> gt_instances = BaseDataElement(
        ...     metainfo=metainfo, det_labels=torch.LongTensor([0, 1, 2, 3]))
        >>> sample = BaseDataElement(metainfo=metainfo,
        ...                          gt_instances=gt_instances)
        >>> print(sample)
        <BaseDataElement(
            META INFORMATION
            img_shape: (800, 1196, 3)
            DATA FIELDS
            gt_instances: <BaseDataElement(
                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    DATA FIELDS
                    det_labels: tensor([0, 1, 2, 3])
                ) at 0x7f0ec5eadc70>
        ) at 0x7f0fea49e130>

        >>> # inheritance
        >>> class DetDataSample(BaseDataElement):
        ...     @property
        ...     def proposals(self):
        ...         return self._proposals
        ...     @proposals.setter
        ...     def proposals(self, value):
        ...         self.set_field(value, '_proposals', dtype=BaseDataElement)
        ...     @proposals.deleter
        ...     def proposals(self):
        ...         del self._proposals
        ...     @property
        ...     def gt_instances(self):
        ...         return self._gt_instances
        ...     @gt_instances.setter
        ...     def gt_instances(self, value):
        ...         self.set_field(value, '_gt_instances',
        ...                        dtype=BaseDataElement)
        ...     @gt_instances.deleter
        ...     def gt_instances(self):
        ...         del self._gt_instances
        ...     @property
        ...     def pred_instances(self):
        ...         return self._pred_instances
        ...     @pred_instances.setter
        ...     def pred_instances(self, value):
        ...         self.set_field(value, '_pred_instances',
        ...                        dtype=BaseDataElement)
        ...     @pred_instances.deleter
        ...     def pred_instances(self):
        ...         del self._pred_instances
        >>> det_sample = DetDataSample()
        >>> proposals = BaseDataElement(bboxes=torch.rand((5, 4)))
        >>> det_sample.proposals = proposals
        >>> assert 'proposals' in det_sample
        >>> assert det_sample.proposals == proposals
        >>> del det_sample.proposals
        >>> assert 'proposals' not in det_sample
        >>> with self.assertRaises(AssertionError):
        ...     det_sample.proposals = torch.rand((5, 4))
    """

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:

        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        """Set or change key-value pairs in ``metainfo_field`` by parameter
        ``metainfo``.

        Args:
            metainfo (dict): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
        """
        assert isinstance(
            metainfo,
            dict), f'metainfo should be a ``dict`` but got {type(metainfo)}'
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type='metainfo', dtype=None)

    def set_data(self, data: dict) -> None:
        """Set or change key-value pairs in ``data_field`` by parameter
        ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data,
                          dict), f'data should be a `dict` but got {data}'
        for k, v in data.items():
            # Use `setattr()` rather than `self.set_field` to allow `set_data`
            # to set property method.
            setattr(self, k, v)

    def update(self, instance: 'BaseDataElement') -> None:
        """The update() method updates the BaseDataElement with the elements
        from another BaseDataElement object.

        Args:
            instance (BaseDataElement): Another BaseDataElement object for
                update the current object.
        """
        assert isinstance(
            instance, BaseDataElement
        ), f'instance should be a `BaseDataElement` but got {type(instance)}'
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self,
            *,
            metainfo: Optional[dict] = None,
            **kwargs) -> 'BaseDataElement':
        """Return a new data element with same type. If ``metainfo`` and
        ``data`` are None, the new data element will have same metainfo and
        data. If metainfo or data is not None, the new result will overwrite it
        with the input value.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
                Defaults to None.
            kwargs (dict): A dict contains annotations of image or
                model predictions.

        Returns:
            BaseDataElement: A new data element with same type.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        """Deep copy the current data element.

        Returns:
            BaseDataElement: The copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) -> list:
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        # We assume that the name of the attribute related to property is
        # '_' + the name of the property. We use this rule to filter out
        # private keys.
        # TODO: Use a more robust way to solve this problem
        private_keys = {
            '_' + key
            for key in self._data_fields
            if isinstance(getattr(type(self), key, None), property)
        }
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo_fields.
        """
        return list(self._metainfo_fields)

    def values(self) -> list:
        """
        Returns:
            list: Contains all values in data.
        """
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo and data.
        """
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo and data.
        """
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo`` and ``data``.
        """
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``data``.
        """
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo``.
        """
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    @property
    def metainfo(self) -> dict:
        """dict: A dict contains metainfo of current data element."""
        return dict(self.metainfo_items())

    def __setattr__(self, name: str, value: Any):
        """setattr is only used to set data."""
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')
        else:
            self.set_field(
                name=name, value=value, field_type='data', dtype=None)

    def __delattr__(self, item: str):
        """Delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a '
                                 'private attribute, which is immutable.')
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    # dict-like methods
    __delitem__ = __delattr__

    def get(self, key, default=None) -> Any:
        """Get property in data and metainfo as the same as python."""
        # Use `getattr()` rather than `self.__dict__.get()` to allow getting
        # properties.
        return getattr(self, key, default)

    def pop(self, *args) -> Any:
        """Pop property in data and metainfo as the same as python."""
        assert len(args) < 3, '``pop`` get more than 2 arguments'
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            # don't just use 'self.__dict__.pop(*args)' for only popping key in
            # metainfo or data
            raise KeyError(f'{args[0]} is not contained in metainfo or data')

    def __contains__(self, item: str) -> bool:
        """Whether the item is in dataelement.

        Args:
            item (str): The key to inquire.
        """
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(self,
                  value: Any,
                  name: str,
                  dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
                  field_type: str = 'data') -> None:
        """Special method for set union field, used as property.setter
        functions."""
        assert field_type in ['metainfo', 'data']
        if dtype is not None:
            assert isinstance(
                value,
                dtype), f'{value} should be a {dtype} but got {type(value)}'

        if field_type == 'metainfo':
            if name in self._data_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of metainfo '
                    f'because {name} is already a data field')
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of data '
                    f'because {name} is already a metainfo field')
            self._data_fields.add(name)
        super().__setattr__(name, value)

    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'BaseDataElement':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cpu(self) -> 'BaseDataElement':
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cuda(self) -> 'BaseDataElement':
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def npu(self) -> 'BaseDataElement':
        """Convert all tensors to NPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.npu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def mlu(self) -> 'BaseDataElement':
        """Convert all tensors to MLU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.mlu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def detach(self) -> 'BaseDataElement':
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def numpy(self) -> 'BaseDataElement':
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) -> 'BaseDataElement':
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data[k] = v
            elif isinstance(v, BaseDataElement):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) -> dict:
        """Convert BaseDataElement to dict."""
        return {
            k: v.to_dict() if isinstance(v, BaseDataElement) else v
            for k, v in self.all_items()
        }

    def __repr__(self) -> str:
        """Represent the object."""

        def _addindent(s_: str, num_spaces: int) -> str:
            """This func is modified from `pytorch` https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29.

            Args:
                s_ (str): The string to add spaces.
                num_spaces (int): The num of space to add.

            Returns:
                str: The string after add indent.
            """
            s = s_.split('\n')
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)  # type: ignore
            s = first + '\n' + s  # type: ignore
            return s  # type: ignore

        def dump(obj: Any) -> str:
            """Represent the object.

            Args:
                obj (Any): The obj to represent.

            Returns:
                str: The represented str.
            """
            _repr = ''
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f'\n{k}: {_addindent(dump(v), 4)}'
            elif isinstance(obj, BaseDataElement):
                _repr += '\n\n    META INFORMATION'
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += '\n\n    DATA FIELDS'
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f'<{classname}({_repr}\n) at {hex(id(obj))}>'
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)
    
class PixelData(BaseDataElement):
    """Data structure for pixel-level annotations or predictions.

    All data items in ``data_fields`` of ``PixelData`` meet the following
    requirements:

    - They all have 3 dimensions in orders of channel, height, and width.
    - They should have the same height and width.

    Examples:
        >>> metainfo = dict(
        ...     img_id=random.randint(0, 100),
        ...     img_shape=(random.randint(400, 600), random.randint(400, 600)))
        >>> image = np.random.randint(0, 255, (4, 20, 40))
        >>> featmap = torch.randint(0, 255, (10, 20, 40))
        >>> pixel_data = PixelData(metainfo=metainfo,
        ...                        image=image,
        ...                        featmap=featmap)
        >>> print(pixel_data.shape)
        (20, 40)

        >>> # slice
        >>> slice_data = pixel_data[10:20, 20:40]
        >>> assert slice_data.shape == (10, 20)
        >>> slice_data = pixel_data[10, 20]
        >>> assert slice_data.shape == (1, 1)

        >>> # set
        >>> pixel_data.map3 = torch.randint(0, 255, (20, 40))
        >>> assert tuple(pixel_data.map3.shape) == (1, 20, 40)
        >>> with self.assertRaises(AssertionError):
        ...     # The dimension must be 3 or 2
        ...     pixel_data.map2 = torch.randint(0, 255, (1, 3, 20, 40))
    """

    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        """Set attributes of ``PixelData``.

        If the dimension of value is 2 and its shape meet the demand, it
        will automatically expand its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `PixelData`.
            value (Union[torch.Tensor, np.ndarray]): The value to store in.
                The type of value must be `torch.Tensor` or `np.ndarray`,
                and its shape must meet the requirements of `PixelData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), \
                f'Can not set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray)}'

            if self.shape:
                assert tuple(value.shape[-2:]) == self.shape, (
                    'The height and width of '
                    f'values {tuple(value.shape[-2:])} is '
                    'not consistent with '
                    'the shape of this '
                    ':obj:`PixelData` '
                    f'{self.shape}')
            assert value.ndim in [
                2, 3
            ], f'The dim of value must be 2 or 3, but got {value.ndim}'
            if value.ndim == 2:
                value = value[None]
                warnings.warn('The shape of value will convert from '
                              f'{value.shape[-2:]} to {value.shape}')
            super().__setattr__(name, value)

    # TODO torch.Long/bool
    def __getitem__(self, item: Sequence[Union[int, slice]]) -> 'PixelData':
        """
        Args:
            item (Sequence[Union[int, slice]]): Get the corresponding values
                according to item.

        Returns:
            :obj:`PixelData`: Corresponding values.
        """

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, tuple):

            assert len(item) == 2, 'Only support to slice height and width'
            tmp_item: List[slice] = list()
            for index, single_item in enumerate(item[::-1]):
                if isinstance(single_item, int):
                    tmp_item.insert(
                        0, slice(single_item, None, self.shape[-index - 1]))
                elif isinstance(single_item, slice):
                    tmp_item.insert(0, single_item)
                else:
                    raise TypeError(
                        'The type of element in input must be int or slice, '
                        f'but got {type(single_item)}')
            tmp_item.insert(0, slice(None, None, None))
            item = tuple(tmp_item)
            for k, v in self.items():
                setattr(new_data, k, v[item])
        else:
            raise TypeError(
                f'Unsupported type {type(item)} for slicing PixelData')
        return new_data

    @property
    def shape(self):
        """The shape of pixel data."""
        if len(self._data_fields) > 0:
            return tuple(self.values()[0].shape[-2:])
        else:
            return None