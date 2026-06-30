"""
guard: shared per-device helpers for the kernels in this package.
  - device_guard: run a Triton kernel launch in the accelerator context of a chosen device, so a kernel
        operating on a tensor that lives on a non-default device (e.g. cuda:1) executes there instead of on
        the current/default device (cuda:0). Every kernel wraps its launch in it.
  - has_capability: the shared compute-capability arch gate (min / max bounds).

Unit test (multi-GPU)  (run from the source root:  CUDA_VISIBLE_DEVICES=<gpu0>,<gpu1> python -m triton_kernels.guard)
"""

import torch


def device_guard(device: torch.device):
    """
    Context manager that makes `device` the current accelerator device for the duration of the block, then
    restores the previous one. Triton launches on the ambient (current) device, so wrap every kernel launch in
    `with device_guard(input_tensor.device):` to keep launches correct when the data lives on a non-default
    device (e.g. cuda:1 while cuda:0 is current).
    """
    device_type: str = device.type
    module = getattr(torch, device_type)  # torch.cuda (or torch.xpu)
    return module.device(device)


def has_capability(
        min_major: int | None = None,
        min_minor: int = 0,
        max_major: int | None = None,
        max_minor: int = 0,
        device_index: int | None = None,
) -> bool:
    """
    True if the CUDA device's compute capability is within [(min_major, min_minor), (max_major, max_minor)].
    Each bound is only applied when its major is not None: omit the upper bound for an open-ended floor (the
    common case), omit the lower bound for an open-ended ceiling, or omit both -> always True. The shared arch
    gate for the whole package -- callers pass the result as a `tl.constexpr` flag so the kernel specializes on
    it (the unsupported branch, e.g. a Blackwell-only instruction, is compiled out on other arches).
    """
    cc: tuple[int, int] = torch.cuda.get_device_capability(device_index)
    return (min_major is None or cc >= (min_major, min_minor)) and (max_major is None or cc <= (max_major, max_minor))


# =========================================================================== #
# Unit test (multi-GPU)  (run from the source root:  CUDA_VISIBLE_DEVICES=<gpu0>,<gpu1> python -m triton_kernels.guard)
# =========================================================================== #
def _unit_test() -> None:
    """
    Self-contained check of device_guard, independent of the other kernels in this package -- it uses only a
    trivial local Triton kernel to exercise the launch protection. With the ambient device pinned to cuda:0
    the guard must (a) switch the current device to the target inside the block and restore it afterwards,
    even when the block raises, and (b) let a Triton kernel launched on a cuda:1 tensor actually run on cuda:1
    and produce the right result. As a control, the same launch WITHOUT the guard must fail, since Triton
    would otherwise launch it against cuda:0's context. Needs >= 2 visible GPUs (else skipped).
    """
    import triton
    import triton.language as tl

    n_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus < 2:
        print(f'Device guard multi-GPU test SKIPPED (needs >= 2 CUDA devices, saw {n_gpus}).')
        return

    @triton.jit
    def _add_one_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(x_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + 1, mask=mask)

    def launch(x: torch.Tensor, guarded: bool) -> None:
        grid = (triton.cdiv(x.numel(), 1024),)
        if guarded:
            with device_guard(x.device):
                _add_one_kernel[grid](x, x.numel(), BLOCK_SIZE=1024)
        else:
            _add_one_kernel[grid](x, x.numel(), BLOCK_SIZE=1024)

    home, other = torch.device('cuda:0'), torch.device('cuda:1')
    torch.cuda.set_device(home)

    # (a) switches the current device inside the block and restores it after -- including when the block raises.
    assert torch.cuda.current_device() == 0
    with device_guard(other):
        assert torch.cuda.current_device() == 1, 'device_guard did not switch the current device'
    assert torch.cuda.current_device() == 0, 'device_guard did not restore the current device'
    try:
        with device_guard(other):
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    assert torch.cuda.current_device() == 0, 'device_guard did not restore the current device after an exception'

    # (b) a real Triton launch on a non-ambient (cuda:1) tensor runs correctly under the guard.
    x: torch.Tensor = torch.arange(4096, dtype=torch.float32, device=other)
    expected: torch.Tensor = x + 1
    launch(x, guarded=True)
    torch.cuda.synchronize(other)
    assert x.equal(expected), 'guarded cuda:1 launch produced the wrong result'
    assert torch.cuda.current_device() == 0, 'device_guard did not restore the current device after a launch'

    # (c) control: the same launch WITHOUT the guard must fail (ambient cuda:0, data on cuda:1). Run last --
    #     Triton rejects the mismatched pointer before the CUDA launch, so this raises cleanly.
    failed: bool = False
    try:
        launch(torch.arange(4096, dtype=torch.float32, device=other), guarded=False)
        torch.cuda.synchronize(other)
    except Exception:
        failed: bool = True
    assert failed, 'expected the UNguarded cuda:1 launch to fail, but it succeeded'

    print('Unit test passed.')


if __name__ == '__main__':
    _unit_test()
