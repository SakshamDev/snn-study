import dataclasses
import pytest
from surj_snn.config import Config
from surj_snn.data import load_data
from surj_snn.experiment import check_protocol, check_test_ready, freeze_protocol, initialize_study

def small_config(tmp_path):
    return Config(dataset="synthetic", data_dir=str(tmp_path / "data"), results_dir=str(tmp_path / "runs"),
                  train_size=200, val_size=50, seeds=[0], timesteps=[4], primary_timestep=4).validate()

def test_final_testing_requires_frozen_protocol(tmp_path):
    cfg = small_config(tmp_path)
    _, manifest = load_data(cfg)
    with pytest.raises(RuntimeError, match="Freeze"):
        check_test_ready(cfg, manifest)
    freeze_protocol(cfg)
    with pytest.raises(RuntimeError, match="selection"):
        check_test_ready(cfg, manifest)

def test_frozen_protocol_rejects_configuration_changes(tmp_path):
    cfg = small_config(tmp_path)
    freeze_protocol(cfg)
    _, manifest = load_data(cfg)
    altered = dataclasses.replace(cfg, learning_rate=.01)
    with pytest.raises(RuntimeError, match="does not match"):
        check_protocol(altered, manifest)

def test_study_prevents_mixing_different_grids(tmp_path):
    cfg = small_config(tmp_path)
    _, manifest = load_data(cfg)
    initialize_study(cfg, manifest)
    with pytest.raises(RuntimeError, match="changed"):
        initialize_study(dataclasses.replace(cfg, epochs=9), manifest)

