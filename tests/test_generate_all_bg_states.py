from generate_all_bg_states import highest_resolution_atlases, state_paths


def test_highest_resolution_atlases_keeps_smallest_voxels():
    names = [
        "allen_mouse_100um",
        "allen_mouse_10um",
        "allen_mouse_25um",
        "allen_human_500um",
        "atlas_without_resolution",
    ]

    assert highest_resolution_atlases(names) == [
        "allen_human_500um",
        "allen_mouse_10um",
        "atlas_without_resolution",
    ]


def test_state_paths_pair_versioned_file_with_stable_latest(tmp_path):
    assert state_paths("duke_mouse_15um", "3.1", tmp_path) == (
        tmp_path / "duke_mouse_v3.1.json",
        tmp_path / "duke_mouse.json",
    )
