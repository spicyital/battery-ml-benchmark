from battery_ml.config import SimulationConfig
from battery_ml.simulation import simulate_battery_data
from battery_ml.splits import split_by_cell, split_chronological


def test_held_out_cells_do_not_overlap_train() -> None:
    frame = simulate_battery_data(
        SimulationConfig(n_cells=4, cycles_per_cell=3, samples_per_cycle=10)
    )
    split = split_by_cell(frame, test_fraction=0.25, validation_fraction=0.25, seed=3)
    assert set(split.train.cell_id).isdisjoint(split.validation.cell_id)
    assert set(split.train.cell_id).isdisjoint(split.test.cell_id)
    assert set(split.validation.cell_id).isdisjoint(split.test.cell_id)


def test_chronological_split_keeps_test_after_training() -> None:
    frame = simulate_battery_data(
        SimulationConfig(n_cells=1, cycles_per_cell=6, samples_per_cycle=8)
    )
    split = split_chronological(frame, validation_fraction=0.17, test_fraction=0.33)
    assert split.train.cycle_id.max() < split.validation.cycle_id.min()
    assert split.validation.cycle_id.max() < split.test.cycle_id.min()
