import numpy as np

from conftest import CentreDetector, write_video
from deepfake_detector.faces import FaceExtractor, crop_face, read_frames, sample_frame_indices


def test_frame_indices_are_spread_and_inside_the_clip():
    idx = sample_frame_indices(300, 16)
    assert len(idx) == 16
    assert idx == sorted(idx)
    assert 0 < idx[0] and idx[-1] < 300
    assert sample_frame_indices(5, 16) == [0, 1, 2, 3, 4]
    assert sample_frame_indices(0, 16) == []


def test_crop_stays_inside_the_frame_near_an_edge():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_face(frame, (0, 0, 40, 40), margin=0.5, size=32)
    assert crop.shape == (32, 32, 3)


def test_read_frames_returns_the_requested_frames(tmp_path):
    path = tmp_path / "clip.mp4"
    write_video(path, frames=20)
    got = read_frames(str(path), [2, 7, 15])
    assert [i for i, _ in got] == [2, 7, 15]
    assert got[0][1].shape[2] == 3


def test_extractor_keeps_the_largest_confident_face():
    class TwoFaces:
        def detect_faces(self, image):
            return [
                {"box": [0, 0, 30, 30], "confidence": 0.99},
                {"box": [40, 40, 50, 50], "confidence": 0.99},
                {"box": [10, 10, 90, 90], "confidence": 0.5},
            ]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[40:90, 40:90] = 255
    face = FaceExtractor(margin=0.0, size=16, detector=TwoFaces()).largest_face(frame)
    assert face.mean() > 250


def test_extractor_skips_frames_without_a_face(tmp_path):
    path = tmp_path / "clip.mp4"
    write_video(path)
    none = FaceExtractor(detector=CentreDetector(confidence=0.1))
    assert none.from_video(str(path), 4) == []
    some = FaceExtractor(size=32, detector=CentreDetector())
    assert len(some.from_video(str(path), 4)) == 4


def test_detection_on_a_downscaled_copy_maps_back_to_full_resolution():
    seen = {}

    class Recording:
        def detect_faces(self, image):
            seen["shape"] = image.shape
            h, w = image.shape[:2]
            return [{"box": [w // 4, h // 4, w // 2, h // 2], "confidence": 0.99}]

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, 480:1440] = 200
    face = FaceExtractor(margin=0.0, size=64, detect_max_side=960, detector=Recording()).largest_face(frame)
    assert seen["shape"][:2] == (540, 960)
    assert face.shape == (64, 64, 3)
    assert face.mean() > 150


def test_a_detector_error_on_an_empty_frame_counts_as_no_face():
    class EmptyBatchError:
        def detect_faces(self, image):
            raise ValueError("The convolution operation resulted in an empty output")

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert FaceExtractor(detector=EmptyBatchError()).largest_face(frame) is None
