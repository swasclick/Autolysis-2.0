import logging


class Logger:
    
    def __init__(self, name= 'Autolysis'):
        
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s"
                )
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
    def get_logger(self):
        return self.logger  