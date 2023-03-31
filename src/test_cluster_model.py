import pytest
import cluster_model
import numpy as np

test_arr = np.ones((2, 128))
test_arr[0][0] = 1

cluster_test = [
    ([{ "sift": np.ones((2, 128)), "landmark": "landmark1", "gps_latitude": 1, "gps_longitude": 1, "gps_altitude": 2 }, 
      { "sift": 2 * np.ones((2, 128)), "landmark": "landmark2", "gps_latitude": 0, "gps_longitude": 0, "gps_altitude": 1 },
      { "sift": 3 * np.ones((2, 128)), "landmark": "landmark3", "gps_latitude": -1, "gps_longitude": 0, "gps_altitude": 3 }], 
      ("testsample", { "sift": test_arr }), 
      "landmark1", (1, 1, 2))
]

class TestEvaluation:
    @pytest.mark.parametrize("xt, xp, ypl, ypd", cluster_test)
    def test_cluster_fit(self, xt, xp, ypl, ypd):
        cluster_model.reset_models(1, 3)
        cluster_model.fit(xt)
        label, dist = cluster_model.predict(xp)
        assert label[0] == ypl
        assert tuple(dist[0]) == ypd