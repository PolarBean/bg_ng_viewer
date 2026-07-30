from generate_all_bg_states import highest_resolution_atlases


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
