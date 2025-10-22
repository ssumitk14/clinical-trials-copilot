import json

class Utility:
    @staticmethod
    def _safe_cell(x):
        if x is None:
            return ''
        if isinstance(x, (list, dict)):
            return json.dumps(x, ensure_ascii=False)
        return str(x)
