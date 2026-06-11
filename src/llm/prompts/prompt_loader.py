import pathlib

class PromptLoader:
    
    def __init__(self):
        pass
    
    def load_prompt(self, prompt_name: str) -> str:
        '''
        Function to load a prompt string by its name
        '''
        PROMPT_DIR = pathlib.Path(__file__).parent
        # print(f'PROMPT_DIR : {PROMPT_DIR}')
        prompt = (PROMPT_DIR/f'templates/{prompt_name}.txt').read_text(encoding='utf-8')
        return prompt