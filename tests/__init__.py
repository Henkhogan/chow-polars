from unittest import TestCase

from polars import DataFrame
from chow_test import chow_test

class TestBase(TestCase):
    def test_base(self):

        data = [[11, 10, 9], [11,  15, 9], [12, 14, 16], [11, 10, 9], [11,  15, 9],
                [12, 14, 16], [11, 10, 9], [11,  15, 9], [12, 14, 16], [11, 10, 9],
                [11,  15, 9], [12, 14, 16], [11, 10, 9], [11,  15, 9], [12, 14, 16]]
        new_data = DataFrame(data, schema=["A", "B", "C"])
        chow, p_val = chow_test(new_data["B"], new_data["A"], 8, 9, 0.01)

        self.assertEqual(chow,0.7978723404255487)
        self.assertEqual(p_val, 0.4769872586883571)