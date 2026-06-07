class AutolysisErrors(Exception):
    def __init__(self, message: str):
        super().__init__(message)
    
    

class APICallerError(AutolysisErrors):
    def __init__(self, message: str = 'Error in making call to API'):
        super().__init__(message)