from prime_rl.inference.vllm.worker.filesystem import FileSystemWeightUpdateWorker
from prime_rl.inference.vllm.worker.nccl import NCCLWeightUpdateWorker


class FakeCommunicator:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class FakeReceiver:
    def __init__(self) -> None:
        self.communicator = FakeCommunicator()


def test_nccl_destroy_broadcaster_releases_and_clears_receiver():
    worker = object.__new__(NCCLWeightUpdateWorker)
    receiver = FakeReceiver()
    worker.nccl_broadcast_receiver = receiver

    worker.destroy_broadcaster()

    assert receiver.communicator.destroyed
    assert not hasattr(worker, "nccl_broadcast_receiver")
    worker.destroy_broadcaster()


def test_filesystem_destroy_broadcaster_is_idempotent():
    worker = object.__new__(FileSystemWeightUpdateWorker)

    worker.destroy_broadcaster()
    worker.destroy_broadcaster()
