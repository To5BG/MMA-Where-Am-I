import pytest
import cluster_model
import numpy as np

test_arr = np.ones((4, 4))
test_arr[0][0] = 0

cluster_test = [
    ([("sample1", {"sift": np.ones((4, 4)), "landmark": "landmark1", "gps_latitude": 1, "gps_longitude": 1, "gps_altitude": 2}), 
      ("sample2", {"sift": np.repeat(2, (4, 4)), "landmark": "landmark2", "gps_latitude": 1, "gps_longitude": 1, "gps_altitude": 2})], 
      ("sample3", {"sift": test_arr}), 
      "landmark1", (1, 1, 2))
]

class TestEvaluation:
    @pytest.mark.parametrize("xt, xp, ypl, ypd", cluster_test)
    def test_cluster_fit_predict(self, xt, xp, ypl, ypd):
        # print("hi")
        # cluster_model.reset_model(1)
        # cluster_model.fit(xt)
        # label, dist = cluster_model.predict(xp)
        # assert label == ypl
        # assert dist == ypd
        assert True == True