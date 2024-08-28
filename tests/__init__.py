from unittest import TestCase

from polars import DataFrame
from chow_test import chow_test, _calculate_rss

data = [[11, 10, 9], [11,  15, 9], [12, 14, 16], [11, 10, 9], [11,  15, 9],
        [12, 14, 16], [11, 10, 9], [11,  15, 9], [12, 14, 16], [11, 10, 9],
        [11,  15, 9], [12, 14, 16], [11, 10, 9], [11,  15, 9], [12, 14, 16]]

class TestBase(TestCase):

    def test_x_series(self):
        new_data = DataFrame(data, schema=["Y", "X", "Z"])
        self.assertEqual(list(new_data["X"]), [10, 15, 14, 10, 15, 14, 10, 15, 14, 10, 15, 14, 10, 15, 14])

    def test_base(self):

        new_data = DataFrame(data, schema=["Y", "X", "Z"])
        chow, p_val = chow_test(new_data, 8, 9, 0.01, x_field='X', y_field='Y')

        self.assertEqual(chow,0.7978723404255487)
        self.assertEqual(p_val, 0.4769872586883571)

    def test_rss(self):

        new_data = DataFrame(data, schema=["Y", "X", "Z"], orient="row")
        _, rss_pooled = _calculate_rss(new_data, x_field='X', y_field='Y')

        self.assertEqual(list(_['y_actual']), [11, 11, 12, 11, 11, 12, 11, 11, 12, 11, 11, 12, 11, 11, 12])    


        self.assertEqual(list(_['y_hat']), [
            11.11904761904762,11.476190476190478,
            11.404761904761905,
            11.11904761904762,
            11.476190476190478,
            11.404761904761905,
            11.11904761904762,
            11.476190476190478,
            11.404761904761905,
            11.11904761904762,
            11.476190476190478,
            11.404761904761905,
            11.11904761904762,
            11.476190476190478,
            11.404761904761905
        ])
        self.assertEqual(rss_pooled, 2.9761904761904843)