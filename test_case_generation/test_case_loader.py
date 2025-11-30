import json


class TestCaseLoader:
    """A class to load and manage test cases from JSON files"""

    def __init__(self, file_path=".data/generated_order_test_cases.json"):
        self.file_path = file_path
        self._test_cases = None
        self._load_test_cases()

    def _load_test_cases(self):
        """Load test cases from the JSON file"""
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
            self._test_cases = data.get("test_cases", [])
        except Exception as e:
            print(f"Error loading test cases from {self.file_path}: {e}")
            self._test_cases = []

    def remove_guid_keys(self, obj):
        """Recursively remove all keys that contain 'guid' from a dictionary or list"""
        if isinstance(obj, dict):
            return {
                key: self.remove_guid_keys(value)
                for key, value in obj.items()
                if "guid" not in key
            }
        elif isinstance(obj, list):
            return [self.remove_guid_keys(item) for item in obj]
        else:
            return obj

    def get_first_test_case(self, remove_guids=True):
        """Get the first test case, optionally removing GUID keys"""
        if not self._test_cases:
            return "{}"

        test_case = self._test_cases[0]

        if remove_guids:
            test_case = self.remove_guid_keys(test_case)

        return json.dumps(test_case, indent=2)

    def get_test_case_by_index(self, index, remove_guids=True):
        """Get a test case by index, optionally removing GUID keys"""
        if not self._test_cases or index >= len(self._test_cases):
            return "{}"

        test_case = self._test_cases[index]

        if remove_guids:
            test_case = self.remove_guid_keys(test_case)

        return json.dumps(test_case, indent=2)

    def get_all_test_cases(self, remove_guids=True):
        """Get all test cases, optionally removing GUID keys"""
        if not self._test_cases:
            return []

        if remove_guids:
            return [self.remove_guid_keys(test_case) for test_case in self._test_cases]

        return self._test_cases

    def get_test_case_count(self):
        """Get the total number of test cases"""
        return len(self._test_cases) if self._test_cases else 0
