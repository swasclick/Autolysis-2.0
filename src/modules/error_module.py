class AutolysisErrors(Exception):
    def __init__(self, message: str):
        super().__init__(message)
    
class APICallerError(AutolysisErrors):
    def __init__(self, message: str = 'Error in making call to API'):
        super().__init__(message)
        
class ResponseParseError(AutolysisErrors):
    def __init__(self, message: str = 'Error in parsing LLM response'):
        super().__init__(message)