import unittest
from utils.data_preprocessing import preprocess_data, load_data, convert_to_integers

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.file_path = 'data/raw/test_data.txt'
        with open(self.file_path, 'w') as f:
            f.write("0 1 2 1 0")

    def test_preprocess_data(self):
        observations = preprocess_data(self.file_path)
        expected = [0, 1, 2, 1, 0]
        self.assertEqual(observations, expected, "Preprocess data did not return expected result")

    def test_load_data(self):
        data = load_data(self.file_path)
        expected = ["0", "1", "2", "1", "0"]
        self.assertEqual(data, expected, "Load data did not return expected result")

    def test_convert_to_integers(self):
        data = ["a", "b", "c", "b", "a"]
        converted = convert_to_integers(data)
        expected = [0, 1, 2, 1, 0]
        self.assertEqual(converted, expected, "Convert to integers did not return expected result")

    def test_preprocess_data_generic(self):
        observations = preprocess_data(self.file_path)
        expected = [0, 1, 2, 1, 0]
        self.assertEqual(observations, expected, "Preprocess data generic did not return expected result")

if __name__ == '__main__':
    unittest.main()
