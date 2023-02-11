import unittest
from spreadsheet3 import SpreadSheet
import csv

index = 'hoge,fuga,piyo,foo,bar\n'
data_part = [[str(n * i) for n in range(5)] for i in range(5)]
data_part_str = [','.join(d) for d in data_part]
csv_data = index + '\n'.join(data_part_str)

class TestStringMethods(unittest.TestCase):

    def test_load_data(self):
        data = SpreadSheet().load_data(csv_data.split('\n'))
        self.assertSequenceEqual(data.data, data_part)

    def test_get_item(self):
        data = SpreadSheet().load_data(csv_data.split('\n'))
        self.assertSequenceEqual(
            data[0],
            dict(hoge='0', fuga='0', piyo='0', foo='0', bar='0'))
        self.assertSequenceEqual(
            data[1],
            dict(hoge='0', fuga='1', piyo='2', foo='3', bar='4'))
        self.assertSequenceEqual(
            data[3],
            dict(hoge='0', fuga='3', piyo='6', foo='9', bar='12'))
        self.assertEqual(list(data['piyo'].data)[3], '6')
        self.assertSequenceEqual(
            data['piyo'],
            ['0', '3', '6', '9', '12'])

    def test_map(self):
        data = SpreadSheet().load_data(csv_data.split('\n'))
        self.assertSequenceEqual(
            data.map('hoge', lambda x: str(int(x) + 1))['hoge'],
            ['1'] * 5)
        self.assertSequenceEqual(
            data.map('hoge', lambda x: str(int(x) + 1))['hoge'],
            [['1'] * 5, ['2'] * 5])

    def test_filter(self):
        data = SpreadSheet().load_data(csv_data.split('\n'))
        self.assertSequenceEqual(
            data.filter('bar', lambda x: int(x) > 10).calc().data,
            [['0', '3', '6', '9', '12'], ['0', '4', '8', '12', '16']]
        )

    def test_len(self):
        index = 'hoge,fuga,piyo,foo,bar\n'
        data_part = [[str(n * i) for n in range(5)] for i in range(5)]
        data_part_str = [','.join(d) for d in data_part]
        csv_data = index + '\n'.join(data_part_str)
        self.assertEqual(len(SpreadSheet().load_data(csv_data.split('\n'))), 5)
        data_part = [[str(n * i) for n in range(5)] for i in range(6)]
        data_part_str = [','.join(d) for d in data_part]
        csv_data = index + '\n'.join(data_part_str)
        self.assertEqual(len(SpreadSheet().load_data(csv_data.split('\n'))), 6)

    def test_add_dict(self):
        data = SpreadSheet().load_data(csv_data.split('\n'))
        data.add_dict({'hogehoge': 'hoge'})
        self.assertSequenceEqual(
            data.data,
            [
                ['0', '0', '0', '0', '0', None],
                ['0', '1', '2', '3', '4', None],
                ['0', '2', '4', '6', '8', None],
                ['0', '3', '6', '9', '12', None],
                ['0', '4', '8', '12', '16', None],
                [None,None,None,None,None,'hoge']
            ]
        )

    def test_add(self):
        data = SpreadSheet().load_data(csv_data.split('\n'))
        self.assertSequenceEqual(
            (data + data).data,
            [
                ['0', '0', '0', '0', '0'],
                ['0', '1', '2', '3', '4'],
                ['0', '2', '4', '6', '8'],
                ['0', '3', '6', '9', '12'],
                ['0', '4', '8', '12', '16'],
                ['0', '0', '0', '0', '0'],
                ['0', '1', '2', '3', '4'],
                ['0', '2', '4', '6', '8'],
                ['0', '3', '6', '9', '12'],
                ['0', '4', '8', '12', '16'],
            ]
        )


if __name__ == '__main__':
    unittest.main()
