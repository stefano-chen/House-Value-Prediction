from geo import utils

def test_calculate_haversine_distance_in_the_same_point():
    assert 0 == utils.calculate_haversine_distance(0,0,0,0)

def test_calculate_haversine_distance_is_always_positive():
    assert utils.calculate_haversine_distance(1, 1, 1, 1) >= 0
    assert utils.calculate_haversine_distance(1, 1, 1, -1) >= 0
    assert utils.calculate_haversine_distance(1, 1, -1, 1) >= 0
    assert utils.calculate_haversine_distance(1, 1, -1, -1) >= 0
    assert utils.calculate_haversine_distance(1, -1, 1, 1) >= 0
    assert utils.calculate_haversine_distance(1, -1, 1, -1) >= 0
    assert utils.calculate_haversine_distance(1, -1, -1, 1) >= 0
    assert utils.calculate_haversine_distance(1, -1, -1, -1) >= 0
    assert utils.calculate_haversine_distance(-1, 1, 1, 1) >= 0
    assert utils.calculate_haversine_distance(-1, 1, 1, -1) >= 0
    assert utils.calculate_haversine_distance(-1, 1, -1, 1) >= 0
    assert utils.calculate_haversine_distance(-1, 1, -1, -1) >= 0
    assert utils.calculate_haversine_distance(-1, -1, 1, 1) >= 0
    assert utils.calculate_haversine_distance(-1, -1, 1, -1) >= 0
    assert utils.calculate_haversine_distance(-1, -1, -1, 1) >= 0
    assert utils.calculate_haversine_distance(-1, -1, -1, -1) >= 0

def test_calculate_haversine_distance_has_commutative_propriety():
    assert utils.calculate_haversine_distance(10,5,20,10) == utils.calculate_haversine_distance(20,10,10,5)