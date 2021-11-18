__all__ = ("Prune",)

import torch
import numpy


class Prune:
    def __init__(
        self,
        model,
        pretrain_step: int = 0,
        sparse_step: int = 0,
        current_step: int = 0,
        frequency: int = 100,
        prune_dict: dict = {},
        restore_sparsity: bool = False,
        fix_sparsity: bool = False,
        prune_device: str = "default",
        deploy_device: str = "none",
        group_size: int = 64,
        fixed_mask=None,
        mask=None
    ):
        self._model = model
        self._t = current_step 
        self._initial_sparsity = {}
        self._pretrain_step = pretrain_step
        self._sparse_step = sparse_step
        self._frequency = frequency
        self._prune_dict = prune_dict
        self._restore_sparsity = restore_sparsity
        self._fix_sparsity = fix_sparsity
        self._prune_device = prune_device
        self._deploy_device = deploy_device
        self._fpga_input_group = 4
        # self._asic_input_gloup = 8
        self._group_size = group_size
        self._asic_input_gloup = 512 // group_size
        self._check_parameter()
        self.fixed_mask = fixed_mask
        self.mask = mask 
        self._mask = {}
        self._prepare()
        if self.fixed_mask:
            self._mask = torch.load(self.fixed_mask)
            self._fix_sparsity = True

        if self.mask:
            self._mask = torch.load(self.mask)

    def _check_parameter(self):
        assert isinstance(self._pretrain_step, int)
        assert isinstance(self._sparse_step, int)
        assert isinstance(self._frequency, int)
        assert isinstance(self._prune_dict, dict)
        assert isinstance(self._restore_sparsity, bool)
        assert isinstance(self._fix_sparsity, bool)
        assert self._prune_device in ["default", "cpu"]
        assert self._deploy_device in ["none", "fpga", "asic"]

    def _prepare(self):
        with torch.no_grad():
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    if (
                        (self._deploy_device == "fpga")
                        and (len(parameter.shape) == 4)
                        and (parameter.shape[1] < self._fpga_input_group)
                    ):
                        self._prune_dict.pop(name)
                        print(
                            "For %s, the parameter %s cannot be balanced pruned and will be deleted from the prune_dict."
                            % (self._deploy_device, name)
                        )
                        continue
                    elif (
                        (self._deploy_device == "asic")
                        and (len(parameter.shape) == 4)
                        and (parameter.shape[1] < self._asic_input_gloup)
                        and ([parameter.shape[2], parameter.shape[3]] == [1, 1])
                    ):
                        self._prune_dict.pop(name)
                        print(
                            "For %s, the parameter %s cannot be balanced pruned and will be deleted from the prune_dict."
                            % (self._deploy_device, name)
                        )
                        continue
                    weight = self._get_weight(parameter)
                    if self._restore_sparsity == True:
                        mask = torch.where(
                            weight == 0,
                            torch.zeros_like(weight),
                            torch.ones_like(weight),
                        )
                        self._initial_sparsity[name] = (
                            1
                            - mask.cpu().numpy().astype(numpy.float32).sum()
                            / weight.cpu().numpy().size
                        )
                        self._mask[name] = mask
                    else:
                        self._initial_sparsity[name] = 0
                        self._mask[name] = torch.ones_like(weight)

    def _update_mask(self, name, weight, keep_k):
        if keep_k >= 1:
            reshape_weight = weight.reshape(-1)
            index = torch.topk(reshape_weight.abs(), keep_k)[1].cpu().numpy().tolist()
            mask = numpy.zeros(reshape_weight.shape)
            mask[index] = 1
            mask = mask.reshape(weight.shape)
            mask = torch.as_tensor(mask, dtype=weight.dtype, device=weight.device)
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_fpga(self, name, weight, keep_k):
        def _block_sparsity_balance(transpose_weight, keep_k, inc_group):
            reshape_weight = transpose_weight.reshape(
                [
                    -1,
                    transpose_weight.shape[-2]
                    * transpose_weight.shape[-1]
                    // inc_group,
                ]
            )
            base_k = keep_k // reshape_weight.shape[0]
            remain_k = keep_k % reshape_weight.shape[0]
            if remain_k > 0:
                index = torch.topk(reshape_weight.abs(), base_k + 1)[1]
            else:
                index = torch.topk(reshape_weight.abs(), base_k)[1]
            dim1 = []
            dim2 = []
            for i, temp in enumerate(index.cpu().numpy().tolist()):
                for j in temp:
                    dim1.append(i)
                    dim2.append(j)
            mask = numpy.zeros(reshape_weight.shape)
            mask[dim1, dim2] = 1
            mask = mask.reshape(transpose_weight.shape)
            mask = mask.transpose([0, 2, 1, 3])
            mask = torch.as_tensor(
                mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            return mask

        if keep_k >= 1:
            transpose_weight = weight.permute([0, 2, 1, 3])
            if transpose_weight.shape[-2] % self._fpga_input_group == 0:
                mask = _block_sparsity_balance(
                    transpose_weight, keep_k, self._fpga_input_group
                )
            else:
                temp1 = transpose_weight.shape[-2]
                temp4 = (self._fpga_input_group - 1) * (
                    temp1 // self._fpga_input_group + 1
                )
                keep_k_1 = int(temp4 / temp1 * keep_k)
                keep_k_2 = keep_k - keep_k_1
                transpose_weight_1 = transpose_weight[:, :, :temp4, :]
                transpose_weight_2 = transpose_weight[:, :, temp4:, :]
                mask_1 = _block_sparsity_balance(
                    transpose_weight_1, keep_k_1, self._fpga_input_group - 1
                )
                mask_2 = _block_sparsity_balance(transpose_weight_2, keep_k_2, 1)
                mask = torch.cat([mask_1, mask_2], 1)
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_asic_4d(self, name, weight, keep_k):
        def _block_sparsity_balance(transpose_weight, keep_k):
            reshape_weight = transpose_weight.reshape([-1, transpose_weight.shape[-1]])
            base_k = keep_k // reshape_weight.shape[0]
            remain_k = keep_k % reshape_weight.shape[0]
            if remain_k > 0:
                index = torch.topk(reshape_weight.abs(), base_k + 1)[1]
            else:
                index = torch.topk(reshape_weight.abs(), base_k)[1]
            dim1 = []
            dim2 = []
            for i, temp in enumerate(index.cpu().numpy().tolist()):
                for j in temp:
                    dim1.append(i)
                    dim2.append(j)
            mask = numpy.zeros(reshape_weight.shape)
            mask[dim1, dim2] = 1
            mask = mask.reshape(transpose_weight.shape)
            mask = mask.transpose([0, 3, 1, 2])
            mask = torch.as_tensor(
                mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            return mask

        def _block_1x1(transpose_weight, keep_k):
            temp1 = transpose_weight.shape[-1] // self._asic_input_gloup
            temp2 = transpose_weight.shape[-1] % self._asic_input_gloup
            for i in range(self._asic_input_gloup):
                locals()["list%s" % i] = []
            for i in range(temp1):
                for j in range(
                    i * self._asic_input_gloup, (i + 1) * self._asic_input_gloup
                ):
                    locals()["list%s" % (j % self._asic_input_gloup)].append(j)
            for i in range(temp1 * self._asic_input_gloup, transpose_weight.shape[-1]):
                locals()["list%s" % (i % self._asic_input_gloup)].append(i)
            temp3 = []
            for i in range(self._asic_input_gloup):
                temp3.append(
                    int(
                        len(locals()["list%s" % i])
                        / transpose_weight.shape[-1]
                        * keep_k
                    )
                )
            group_mask = numpy.ones(transpose_weight.shape).transpose([0, 3, 1, 2])
            for i in range(self._asic_input_gloup):
                temp4 = torch.cat(
                    [
                        transpose_weight[:, :, :, one : one + 1]
                        for one in locals()["list%s" % i]
                    ],
                    3,
                )
                mask = _block_sparsity_balance(temp4, temp3[i])
                for one, two in enumerate(locals()["list%s" % i]):
                    group_mask[:, two : two + 1, :, :] = (
                        mask[:, one : one + 1, :, :].cpu().numpy()
                    )
            group_mask = torch.as_tensor(
                group_mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            return group_mask

        if keep_k >= 1:
            transpose_weight = weight.permute([0, 2, 3, 1])
            if transpose_weight.shape[1] == 1 and transpose_weight.shape[2] == 1:
                group_size = 512
                temp1 = transpose_weight.shape[-1] // group_size
                temp2 = transpose_weight.shape[-1] % group_size
                keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
                keep_k_2 = keep_k - keep_k_1
                mask = numpy.ones(weight.shape)
                if temp1 > 0:
                    for i in range(temp1):
                        transpose_weight_1 = transpose_weight[
                            :, :, :, i * group_size : (i + 1) * group_size
                        ]
                        mask_1 = _block_1x1(transpose_weight_1, keep_k_1 // temp1)
                        mask[
                            :, i * group_size : (i + 1) * group_size, :, :
                        ] = mask_1.cpu().numpy()
                if temp2 > 0:
                    transpose_weight_2 = transpose_weight[:, :, :, temp1 * group_size :]
                    if transpose_weight_2.shape[-1] >= self._asic_input_gloup:
                        mask_2 = _block_1x1(transpose_weight_2, keep_k_2)
                        mask[:, temp1 * group_size :, :, :] = mask_2.cpu().numpy()
                    else:
                        pass
                mask = torch.as_tensor(
                    mask, dtype=transpose_weight.dtype, device=transpose_weight.device
                )
            else:
                group_size = self._group_size
                temp1 = transpose_weight.shape[-1] // group_size
                temp2 = transpose_weight.shape[-1] % group_size
                keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
                keep_k_2 = keep_k - keep_k_1
                mask = numpy.ones(weight.shape)
                if temp1 > 0:
                    for i in range(temp1):
                        transpose_weight_1 = transpose_weight[
                            :, :, :, i * group_size : (i + 1) * group_size
                        ]
                        mask_1 = _block_sparsity_balance(
                            transpose_weight_1, keep_k_1 // temp1
                        )
                        mask[
                            :, i * group_size : (i + 1) * group_size, :, :
                        ] = mask_1.cpu().numpy()
                if temp2 > 0:
                    transpose_weight_2 = transpose_weight[:, :, :, temp1 * group_size :]
                    mask_2 = _block_sparsity_balance(transpose_weight_2, keep_k_2)
                    mask[:, temp1 * group_size :, :, :] = mask_2.cpu().numpy()
                mask = torch.as_tensor(
                    mask, dtype=transpose_weight.dtype, device=transpose_weight.device
                )
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_asic_2d(self, name, weight, keep_k):
        def _block_sparsity_balance(transpose_weight, keep_k):
            reshape_weight = transpose_weight
            base_k = keep_k // reshape_weight.shape[0]
            remain_k = keep_k % reshape_weight.shape[0]
            if remain_k > 0:
                index = torch.topk(reshape_weight.abs(), base_k + 1)[1]
            else:
                index = torch.topk(reshape_weight.abs(), base_k)[1]
            dim1 = []
            dim2 = []
            for i, temp in enumerate(index.cpu().numpy().tolist()):
                for j in temp:
                    dim1.append(i)
                    dim2.append(j)
            mask = numpy.zeros(reshape_weight.shape)
            mask[dim1, dim2] = 1
            mask = torch.as_tensor(
                mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            return mask

        def _block_1x1(transpose_weight, keep_k):
            temp1 = transpose_weight.shape[-1] // self._asic_input_gloup
            temp2 = transpose_weight.shape[-1] % self._asic_input_gloup
            for i in range(self._asic_input_gloup):
                locals()["list%s" % i] = []
            for i in range(temp1):
                for j in range(
                    i * self._asic_input_gloup, (i + 1) * self._asic_input_gloup
                ):
                    locals()["list%s" % (j % self._asic_input_gloup)].append(j)
            for i in range(temp1 * self._asic_input_gloup, transpose_weight.shape[-1]):
                locals()["list%s" % (i % self._asic_input_gloup)].append(i)
            temp3 = []
            for i in range(self._asic_input_gloup):
                temp3.append(
                    int(
                        len(locals()["list%s" % i])
                        / transpose_weight.shape[-1]
                        * keep_k
                    )
                )
            group_mask = numpy.ones(transpose_weight.shape)
            for i in range(self._asic_input_gloup):
                temp4 = torch.cat(
                    [
                        transpose_weight[:, one : one + 1]
                        for one in locals()["list%s" % i]
                    ],
                    1,
                )
                mask = _block_sparsity_balance(temp4, temp3[i])
                for one, two in enumerate(locals()["list%s" % i]):
                    group_mask[:, two : two + 1] = mask[:, one : one + 1].cpu().numpy()
            group_mask = torch.as_tensor(
                group_mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            return group_mask

        if keep_k >= 1:
            transpose_weight = weight
            group_size = 512
            temp1 = transpose_weight.shape[-1] // group_size
            temp2 = transpose_weight.shape[-1] % group_size
            keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
            keep_k_2 = keep_k - keep_k_1
            mask = numpy.ones(weight.shape)
            if temp1 > 0:
                for i in range(temp1):
                    transpose_weight_1 = transpose_weight[
                        :, i * group_size : (i + 1) * group_size
                    ]
                    mask_1 = _block_1x1(transpose_weight_1, keep_k_1 // temp1)
                    mask[
                        :, i * group_size : (i + 1) * group_size
                    ] = mask_1.cpu().numpy()
            if temp2 > 0:
                transpose_weight_2 = transpose_weight[:, temp1 * group_size :]
                if transpose_weight_2.shape[-1] >= self._asic_input_gloup:
                    mask_2 = _block_1x1(transpose_weight_2, keep_k_2)
                    mask[:, temp1 * group_size :] = mask_2.cpu().numpy()
                else:
                    pass
            mask = torch.as_tensor(
                mask, dtype=transpose_weight.dtype, device=transpose_weight.device
            )
            self._mask[name][:] = mask
        else:
            self._mask[name][:] = 0

    def _update_mask_conditions(self):
        condition1 = self._fix_sparsity == False
        condition2 = (
            self._pretrain_step < self._t <= self._pretrain_step + self._sparse_step
        )
        condition3 = (self._t - self._pretrain_step) % self._frequency == 0
        return condition1 and condition2 and condition3

    def _get_weight(self, parameter):
        if self._prune_device == "default":
            weight = parameter.data
        elif self._prune_device == "cpu":
            weight = parameter.data.to(device=torch.device("cpu"))
        return weight

    def prune(self):
        current_sparsity = None

        with torch.no_grad():
            self._t = self._t + 1
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    weight = self._get_weight(parameter)
                    if self._update_mask_conditions():
                        weight = weight * self._mask[name]
                        target_sparsity = self._prune_dict[name]
                        current_sparse_step = (
                            self._t - self._pretrain_step
                        ) // self._frequency
                        total_srarse_step = self._sparse_step // self._frequency
                        current_sparsity = (
                            target_sparsity
                            + (self._initial_sparsity[name] - target_sparsity)
                            * (1.0 - current_sparse_step / total_srarse_step) ** 3
                        )
                        keep_k = int(
                            weight.cpu().numpy().size * (1.0 - current_sparsity)
                        )
                        if self._deploy_device == "none":
                            self._update_mask(name, weight, keep_k)
                        elif self._deploy_device == "fpga":
                            if len(weight.shape) == 4:
                                self._update_mask_fpga(name, weight, keep_k)
                            else:
                                self._update_mask(name, weight, keep_k)
                        elif self._deploy_device == "asic":
                            if len(weight.shape) == 4:
                                self._update_mask_asic_4d(name, weight, keep_k)
                            elif len(weight.shape) == 2:
                                self._update_mask_asic_2d(name, weight, keep_k)
                            else:
                                self._update_mask(name, weight, keep_k)

                    mask = self._mask[name]
                    if not mask.is_cuda:
                        mask = torch.as_tensor(
                            mask, dtype=weight.dtype, device=weight.device
                        )
                        self._mask[name] = mask
                    parameter.mul_(self._mask[name])

        return current_sparsity

    def sparsity(self):
        total_param = 0
        total_nonezero = 0
        layer_sparse_rate = {}
        for name, parameter in self._model.named_parameters():
            if any(name == one for one in self._prune_dict):
                temp = parameter.data.cpu().numpy()
                total_param = total_param + temp.size
                total_nonezero = total_nonezero + numpy.flatnonzero(temp).size
                layer_sparse_rate[name] = 1 - numpy.flatnonzero(temp).size / temp.size
        total_sparse_rate = 1 - total_nonezero / total_param
        return layer_sparse_rate, total_sparse_rate

    def check(self):
        def _check_weight(weight, keep_k):
            qualify = numpy.flatnonzero(weight).size <= keep_k
            return qualify

        def _check_weight_fpga(weight, keep_k):
            def _check_block_sparsity_balance(transpose_weight, keep_k, inc_group):
                reshape_weight = transpose_weight.reshape(
                    [
                        -1,
                        transpose_weight.shape[-2]
                        * transpose_weight.shape[-1]
                        // inc_group,
                    ]
                )
                base_k = keep_k // reshape_weight.shape[0]
                remain_k = keep_k % reshape_weight.shape[0]
                k = base_k + 1 if remain_k > 0 else base_k
                qualify_list = []
                for one in reshape_weight:
                    qualify_list.append(numpy.flatnonzero(one).size <= k)
                return all(qualify_list)

            transpose_weight = weight.transpose([0, 2, 1, 3])
            if transpose_weight.shape[-2] % self._fpga_input_group == 0:
                qualify = _check_block_sparsity_balance(
                    transpose_weight, keep_k, self._fpga_input_group
                )
            else:
                temp1 = transpose_weight.shape[-2]
                temp4 = (self._fpga_input_group - 1) * (
                    temp1 // self._fpga_input_group + 1
                )
                keep_k_1 = int(temp4 / temp1 * keep_k)
                keep_k_2 = keep_k - keep_k_1
                transpose_weight_1 = transpose_weight[:, :, :temp4, :]
                transpose_weight_2 = transpose_weight[:, :, temp4:, :]
                qualify_1 = _check_block_sparsity_balance(
                    transpose_weight_1, keep_k_1, self._fpga_input_group - 1
                )
                qualify_2 = _check_block_sparsity_balance(
                    transpose_weight_2, keep_k_2, 1
                )
                qualify = all([qualify_1, qualify_2])
            return qualify

        def _check_weight_asic_4d(weight, keep_k):
            def _check_block_sparsity_balance(transpose_weight, keep_k):
                reshape_weight = transpose_weight.reshape(
                    [-1, transpose_weight.shape[-1]]
                )
                base_k = keep_k // reshape_weight.shape[0]
                remain_k = keep_k % reshape_weight.shape[0]
                k = base_k + 1 if remain_k > 0 else base_k
                qualify_list = []
                for one in reshape_weight:
                    qualify_list.append(numpy.flatnonzero(one).size <= k)
                return all(qualify_list)

            def _check_block_1x1(transpose_weight, keep_k):
                temp1 = transpose_weight.shape[-1] // self._asic_input_gloup
                temp2 = transpose_weight.shape[-1] % self._asic_input_gloup
                for i in range(self._asic_input_gloup):
                    locals()["list%s" % i] = []
                for i in range(temp1):
                    for j in range(
                        i * self._asic_input_gloup, (i + 1) * self._asic_input_gloup
                    ):
                        locals()["list%s" % (j % self._asic_input_gloup)].append(j)
                for i in range(
                    temp1 * self._asic_input_gloup, transpose_weight.shape[-1]
                ):
                    locals()["list%s" % (i % self._asic_input_gloup)].append(i)
                temp3 = []
                for i in range(self._asic_input_gloup):
                    temp3.append(
                        int(
                            len(locals()["list%s" % i])
                            / transpose_weight.shape[-1]
                            * keep_k
                        )
                    )
                qualify_list = []
                for i in range(self._asic_input_gloup):
                    temp4 = numpy.concatenate(
                        [
                            transpose_weight[:, :, :, one : one + 1]
                            for one in locals()["list%s" % i]
                        ],
                        3,
                    )
                    qualify_list.append(_check_block_sparsity_balance(temp4, temp3[i]))
                return all(qualify_list)

            transpose_weight = weight.transpose([0, 2, 3, 1])
            if transpose_weight.shape[1] == 1 and transpose_weight.shape[2] == 1:
                group_size = 512
                temp1 = transpose_weight.shape[-1] // group_size
                temp2 = transpose_weight.shape[-1] % group_size
                keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
                keep_k_2 = keep_k - keep_k_1
                if temp1 > 0:
                    for i in range(temp1):
                        transpose_weight_1 = transpose_weight[
                            :, :, :, i * group_size : (i + 1) * group_size
                        ]
                        qualify_1 = _check_block_1x1(
                            transpose_weight_1, keep_k_1 // temp1
                        )
                if temp2 > 0:
                    transpose_weight_2 = transpose_weight[:, :, :, temp1 * group_size :]
                    if transpose_weight_2.shape[-1] >= self._asic_input_gloup:
                        qualify_2 = _check_block_1x1(transpose_weight_2, keep_k_2)
                    else:
                        pass
                qualify_list = []
                try:
                    qualify_list.append(qualify_1)
                except:
                    pass
                try:
                    qualify_list.append(qualify_2)
                except:
                    pass
                qualify = all(qualify_list)
            else:
                group_size = self._group_size
                temp1 = transpose_weight.shape[-1] // group_size
                temp2 = transpose_weight.shape[-1] % group_size
                keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
                keep_k_2 = keep_k - keep_k_1
                if temp1 > 0:
                    for i in range(temp1):
                        transpose_weight_1 = transpose_weight[
                            :, :, :, i * group_size : (i + 1) * group_size
                        ]
                        qualify_1 = _check_block_sparsity_balance(
                            transpose_weight_1, keep_k_1 // temp1
                        )
                if temp2 > 0:
                    transpose_weight_2 = transpose_weight[:, :, :, temp1 * group_size :]
                    qualify_2 = _check_block_sparsity_balance(
                        transpose_weight_2, keep_k_2
                    )
                qualify_list = []
                try:
                    qualify_list.append(qualify_1)
                except:
                    pass
                try:
                    qualify_list.append(qualify_2)
                except:
                    pass
                qualify = all(qualify_list)
            return qualify

        def _check_weight_asic_2d(weight, keep_k):
            def _check_block_sparsity_balance(transpose_weight, keep_k):
                reshape_weight = transpose_weight
                base_k = keep_k // reshape_weight.shape[0]
                remain_k = keep_k % reshape_weight.shape[0]
                k = base_k + 1 if remain_k > 0 else base_k
                qualify_list = []
                for one in reshape_weight:
                    qualify_list.append(numpy.flatnonzero(one).size <= k)
                return all(qualify_list)

            def _check_block_1x1(transpose_weight, keep_k):
                temp1 = transpose_weight.shape[-1] // self._asic_input_gloup
                temp2 = transpose_weight.shape[-1] % self._asic_input_gloup
                for i in range(self._asic_input_gloup):
                    locals()["list%s" % i] = []
                for i in range(temp1):
                    for j in range(
                        i * self._asic_input_gloup, (i + 1) * self._asic_input_gloup
                    ):
                        locals()["list%s" % (j % self._asic_input_gloup)].append(j)
                for i in range(
                    temp1 * self._asic_input_gloup, transpose_weight.shape[-1]
                ):
                    locals()["list%s" % (i % self._asic_input_gloup)].append(i)
                temp3 = []
                for i in range(self._asic_input_gloup):
                    temp3.append(
                        int(
                            len(locals()["list%s" % i])
                            / transpose_weight.shape[-1]
                            * keep_k
                        )
                    )
                qualify_list = []
                for i in range(self._asic_input_gloup):
                    temp4 = numpy.concatenate(
                        [
                            transpose_weight[:, one : one + 1]
                            for one in locals()["list%s" % i]
                        ],
                        1,
                    )
                    qualify_list.append(_check_block_sparsity_balance(temp4, temp3[i]))
                return all(qualify_list)

            transpose_weight = weight
            group_size = 512
            temp1 = transpose_weight.shape[-1] // group_size
            temp2 = transpose_weight.shape[-1] % group_size
            keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
            keep_k_2 = keep_k - keep_k_1
            if temp1 > 0:
                for i in range(temp1):
                    transpose_weight_1 = transpose_weight[
                        :, i * group_size : (i + 1) * group_size
                    ]
                    qualify_1 = _check_block_1x1(transpose_weight_1, keep_k_1 // temp1)
            if temp2 > 0:
                transpose_weight_2 = transpose_weight[:, temp1 * group_size :]
                if transpose_weight_2.shape[-1] >= self._asic_input_gloup:
                    qualify_2 = _check_block_1x1(transpose_weight_2, keep_k_2)
                else:
                    pass
            qualify_list = []
            try:
                qualify_list.append(qualify_1)
            except:
                pass
            try:
                qualify_list.append(qualify_2)
            except:
                pass
            qualify = all(qualify_list)
            return qualify

        with torch.no_grad():
            layer_sparse_qualify = {}
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    weight = parameter.data.cpu().numpy()
                    target_sparsity = self._prune_dict[name]
                    keep_k = int(weight.size * (1.0 - target_sparsity))
                    if self._deploy_device == "none":
                        qualify = _check_weight(weight, keep_k)
                    elif self._deploy_device == "fpga":
                        if len(weight.shape) == 4:
                            qualify = _check_weight_fpga(weight, keep_k)
                        else:
                            qualify = _check_weight(weight, keep_k)
                    elif self._deploy_device == "asic":
                        if len(weight.shape) == 4:
                            qualify = _check_weight_asic_4d(weight, keep_k)
                        elif len(weight.shape) == 2:
                            qualify = _check_weight_asic_2d(weight, keep_k)
                        else:
                            qualify = _check_weight(weight, keep_k)
                    layer_sparse_qualify[name] = qualify
        total_sparse_qualify = all(one for one in layer_sparse_qualify.values())
        return layer_sparse_qualify, total_sparse_qualify

"""
(deberta): DebertaModel(
    (embeddings): DebertaEmbeddings(
      (word_embeddings): Embedding(50265, 1024, padding_idx=0)
      (LayerNorm): DebertaLayerNorm()
      (dropout): StableDropout()
    )
    (encoder): DebertaEncoder(
      (layer): ModuleList(
        (0): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (1): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (2): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (3): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (4): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (5): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (6): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (7): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (8): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (9): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (10): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (11): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (12): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (13): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (14): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (15): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (16): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (17): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (18): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (19): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (20): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (21): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (22): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
        (23): DebertaLayer(
          (attention): DebertaAttention(
            (self): DisentangledSelfAttention(
              (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
              (pos_dropout): StableDropout()
              (pos_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (pos_q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): StableDropout()
            )
            (output): DebertaSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): DebertaLayerNorm()
              (dropout): StableDropout()
            )
          )
          (intermediate): DebertaIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (output): DebertaOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): DebertaLayerNorm()
            (dropout): StableDropout()
          )
        )
      )
      (rel_embeddings): Embedding(1024, 1024)
    )
  )
  (pooler): ContextPooler(
    (dense): Linear(in_features=1024, out_features=1024, bias=True)
    (dropout): StableDropout()
  )
  (classifier): Linear(in_features=1024, out_features=3, bias=True)
  (dropout): StableDropout()
)
"""
