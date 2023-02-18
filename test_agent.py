import unittest
import agent

class TestAgentMethods(unittest.TestCase):
    
    def test_read_parameter_from_repo(self):
        result = agent.read_parameters_from_repo(["innodb_buffer_pool_size", "innodb_io_capacity"])
        print(result)
        self.assertEqual(result, [1109659400, 200])

    def test_read_threshold_from_repo(self):
        result = agent.read_threshold_from_repo()
        print(result)
        self.assertEqual(result, (62, 1862))

if __name__ == '__main__':
    unittest.main()