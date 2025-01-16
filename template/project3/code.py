from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
#from langchain.prompts import load_prompt
os.environ['OPENAI_API_KEY'] = 'sk-######'
prompt_str = '''테스트 {test}'''
template = PromptTemplate.from_template(prompt_str)
#template = load_prompt('myprompt.json') # PromptTemplate
test = '''테스트'''

prompt = template.format(test=test)
json_schema = {'title': 'joke', 'description': 'Joke to tell user.', 'type': 'object', 'properties': {'property_1': {'type': 'string', 'description': 'Description for property_1'}}, 'required': ['property_1']}
llm = ChatOpenAI(model='gpt-4o-mini-2024-07-18', temperature=0.0, max_tokens=1000)
llm_structured = llm.with_structured_output(json_schema)
llm_result = llm_structured.invoke(prompt)
