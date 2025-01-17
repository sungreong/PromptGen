import streamlit as st
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from prompt_example_main import prompt_template_folder_view , PromptEvalTemplate , user_guide_view
import os
from langchain.callbacks import get_openai_callback
import json
from dotenv import load_dotenv
import logging
from datetime import datetime
import pathlib

load_dotenv()

def setup_logger(username):
    """사용자별 로거 설정"""
    try:
        # 로그 디렉토리 생성
        log_dir = pathlib.Path("logs") / username
        log_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(log_dir, 0o777)  # 디렉토리 권한 설정
        
        # 현재 날짜로 로그 파일 생성
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{current_date}.log"
        
        # 로그 파일이 없으면 생성하고 권한 설정
        if not log_file.exists():
            log_file.touch()
            os.chmod(log_file, 0o666)  # 파일 권한 설정
        
        # 로거 설정
        logger = logging.getLogger(username)
        if not logger.handlers:  # 핸들러가 없을 때만 추가
            logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    except Exception as e:
        st.error(f"로그 설정 중 오류 발생: {str(e)}")
        # 임시 대체 로거 반환
        console_handler = logging.StreamHandler()
        logger = logging.getLogger(f"temp_{username}")
        logger.addHandler(console_handler)
        return logger

def make_json_schema(key="make_json_schema"):
    # JSON 스키마 생성
    st.header("JSON Schema 생성")
    folder_name = st.session_state.prompt_template.get("folder_name", "")
    
    # template 디렉토리 및 하위 디렉토리 생성 및 권한 설정
    if folder_name:
        template_dir = pathlib.Path("template")
        folder_dir = template_dir / folder_name
        
        # 디렉토리 생성
        template_dir.mkdir(exist_ok=True)
        folder_dir.mkdir(exist_ok=True)
        
        # 권한 설정
        try:
            os.chmod(template_dir, 0o777)
            os.chmod(folder_dir, 0o777)
        except Exception as e:
            st.warning(f"권한 설정 중 오류 발생: {str(e)}")
    
    if folder_name and os.path.exists(f"./template/{folder_name}/json_schema.json"):
        try:
            with open(f"./template/{folder_name}/json_schema.json", "r") as f:
                json_schema = json.load(f)
        except Exception as e:
            st.error(f"JSON Schema 파일 읽기 오류: {str(e)}")
            json_schema = {}
    else:
        json_schema = {}

    title = st.text_input("Title", json_schema.get("title", "joke"), key=f"prompt_app_title_{key}")
    description = st.text_input("Description", json_schema.get("description", "Joke to tell user."), key=f"prompt_app_description_{key}")
    template_property = {
        "rewrited_query": {
            "type": "string",
            "description": "신규로 생성된 쿼리를 저장하는 곳 "
        }
    }
    properties = json_schema.get("properties", template_property)

    num_properties = st.number_input("Number of properties", min_value=1, max_value=10, value=len(properties),key=f"prompt_app_num_properties_{key}")
    tab_list = st.tabs([f"Property {i+1}" for i in range(num_properties)])
    property_name_list = list(properties.keys())
    for i in range(num_properties):
        with tab_list[i]:
            if i < len(property_name_list):
                prop_name = st.text_input(f"Property {i+1} Name", property_name_list[i], key=f"prompt_app_prop_name_{key}_{i}")
                prop_type = st.selectbox(f"Property {i+1} Type", properties[prop_name].get("type", "string"), index=0, key=f"prompt_app_prop_type_{key}_{i}")
                prop_description = st.text_input(f"Property {i+1} Description", properties[prop_name].get("description", ""), key=f"prompt_app_prop_description_{key}_{i}")
            else :
                prop_name = st.text_input(f"Property {i+1} Name", f"property_{i+1}", key=f"prompt_app_prop_name_{key}_{i}")
                prop_type = st.selectbox(f"Property {i+1} Type", ["string", "integer", "boolean"], index=0, key=f"prompt_app_prop_type_{key}_{i}")
                prop_description = st.text_input(f"Property {i+1} Description", f"Description for {prop_name}", key=f"prompt_app_prop_description_{key}_{i}")
            properties[prop_name] = {
                "type": prop_type,
                "description": prop_description
            }

    json_schema = {
        "title": title,
        "description": description,
        "type": "object",
        "properties": properties,
        "required": list(properties.keys())
    }
    st.json(json_schema)
    return json_schema


def llm_test_view(key="llm_test_view"):
    st.header("LLM Test View")
    sample_result = {}
    input_list = st.session_state.prompt_template.get("input_list", [])
    input_values = st.session_state.prompt_template.get("input_values", {})
    folder_name = st.session_state.prompt_template.get("folder_name", "")
    temperature = st.session_state.prompt_template.get("temperature", 0.0)
    max_tokens = st.session_state.prompt_template.get("max_tokens", 1000)
    model = st.session_state.prompt_template.get("model", "gpt-4o-mini-2024-07-18")
    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input("API 키를 입력하세요:", type="password", key=f"prompt_app_api_key_{key}")
    with col2:
        model = st.selectbox(
            "모델 선택",
            ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06","gpt-4o-2024-11-20"],
            key=f"prompt_app_model_{key}",
        )
        st.session_state.prompt_template["model"] = model
    with st.expander("모델 파라미터 설정"):
        temperature = st.number_input("Temperature", 0.0, 1.0, 0.0, key=f"prompt_app_temperature_{key}")
        max_tokens = st.number_input("Max Tokens", 100, 10000, 1000, key=f"prompt_app_max_tokens_{key}")
        st.session_state.prompt_template["temperature"] = temperature
        st.session_state.prompt_template["max_tokens"] = max_tokens
    tab1, json_tab, tab2 , code_tab = st.tabs(["Test Input", "JSON Schema", "LLM Eval" , "Code"])
    with tab1:
        if len(input_list) > 0:
            input_values = {}
            # 입력 필드를 두 개씩 나누어 배치
            tab_name_list = [col for col in input_list]
            tab_list = st.tabs(tab_name_list + [":star: Final PROMPT :star:"])
            for tab_name, tab in zip(tab_name_list, tab_list[:-1]):
                col = tab_name
                with tab:
                    height = st.number_input("Height", 100, 1000, 300, key=f"{col}_height_{key}")
                    input_values[col] = st.text_area(f"{col}", "", height=height, key=f"{col}_input_{key}")
            st.session_state.prompt_template["input_values"] = input_values
        else:
            tab_list = st.tabs([":star: Final PROMPT :star:"])
        
        prompt_tab = tab_list[-1]
        with prompt_tab:
            height = st.number_input("Height", 100, 1000, 400, key=f"prompt_app_height_{key}")
            prompt_string = st.session_state.prompt_template.get("prompt_string", "")
            prompt_template = PromptTemplate.from_template(template=prompt_string)
            if len(input_list) > 0:
                print(input_values)
                try:
                    prompt = prompt_template.format(**input_values)
                except Exception as e:
                    prompt = f"Error occurred: {e}"
            else:
                prompt = prompt_template.format()

            prompt = st.text_area("Prompt Example", prompt, height=height, key=f"prompt_example_{key}")

    with json_tab:
        json_schema = make_json_schema(key)
        json_schema_str = json.dumps(json_schema, indent=4, ensure_ascii=False)  # dict를 JSON 문자열로 변환
        st.session_state.prompt_template["json_schema"] = json_schema
        # save json schema
        if folder_name is not None :
            if st.button("Save JSON Schema"):
                with open(f"./template/{folder_name}/json_schema.json", "w") as f:
                    f.write(json_schema_str)
                st.success(f"JSON Schema 저장 완료 (./template/{folder_name}/json_schema.json)")

    with tab2:
        llm = ChatOpenAI(openai_api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens) if api_key else None
        on = st.toggle("Show Prompt", value=False, key=f"prompt_app_toggle_{key}")
        if on:
            st.text_area("Prompt", prompt, height=height, key=f"prompt_app_prompt_{key}")
        with st.form(key=f"prompt_app_form_{key}"):
            if len(json_schema) > 0:
                if llm:
                    llm_structured = llm.with_structured_output(json_schema)
                else :
                    st.error("LLM API KEY를 입력해주세요.")
                    st.stop()
            else :
                st.error("JSON Schema를 생성해주세요.")
                st.stop()
            col1, col2 = st.columns(2)
            with col1:
                button_llm_test = st.form_submit_button("테스트")
            with col2:
                button_llm_eval = st.form_submit_button("프롬프트 평가")
            if button_llm_test and llm:
                with get_openai_callback() as cb:
                    if llm_structured:
                        sample_result = llm_structured.invoke(prompt)
                        # 로그 기록
                        if 'username' in st.session_state:
                            logger.info(f"""
                            Action: LLM Test
                            Model: {model}
                            Temperature: {temperature}
                            Max Tokens: {max_tokens}
                            Prompt: {prompt}
                            Response: {json.dumps(sample_result, ensure_ascii=False)}
                            Total Tokens: {cb.total_tokens}
                            Total Cost: {cb.total_cost}
                            """)
                    else:
                        sample_result = llm.invoke(prompt)
                    # Using beta_columns to create a more structured layout
                    with st.container(border=True):
                        st.markdown("**Response**")
                        st.json(sample_result)
                    with st.container(border=True):
                        st.markdown("**Token Details**")
                        st.write(f"Total Tokens: {cb.total_tokens}")
                        st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                        st.write(f"Completion Tokens: {cb.completion_tokens}")
                        st.markdown("**Cost Details**")
                        st.markdown(
                            f"<h4 style='color: red;'>Total Cost: {cb.total_cost:.6f}$</h4>", unsafe_allow_html=True
                        )

            if button_llm_eval and llm:
                with get_openai_callback() as cb:
                    prompt_eval_template = PromptTemplate.from_template(template=PromptEvalTemplate)

                    eval_prompt = prompt_eval_template.format(prompt_template=prompt_string)
                    for input in input_list:
                        eval_prompt = eval_prompt.format(**{input: "[variable]"})
                    sample_result = llm.invoke(eval_prompt).content
                    # Using beta_columns to create a more structured layout

                    with st.container(border=True):
                        st.markdown("**Response**")
                        st.markdown(sample_result)
                    with st.container(border=True):
                        st.markdown("**Token Details**")
                        st.write(f"Total Tokens: {cb.total_tokens}")
                        st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                        st.write(f"Completion Tokens: {cb.completion_tokens}")
                        st.markdown("**Cost Details**")
                        st.markdown(
                            f"<h4 style='color: red;'>Total Cost: {cb.total_cost:.6f}$</h4>", unsafe_allow_html=True
                        )
                    if 'username' in st.session_state:
                        logger.info(f"""
                        Action: Prompt Evaluation
                        Model: {model}
                        Temperature: {temperature}
                        Max Tokens: {max_tokens}
                        Eval Prompt: {eval_prompt}
                        Response: {sample_result}
                        Total Tokens: {cb.total_tokens}
                        Total Cost: {cb.total_cost}
                        """)
    with code_tab:
        user_guide_view(key)
    return sample_result

def load_users():
    """사용자 정보 로드"""
    try:
        with open('users.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 기본 관리자 계정으로 초기화
        default_users = {
            "users": [
                {
                    "username": os.getenv('user_name', 'admin'),
                    "password": os.getenv('password', 'agilesoda$01'),
                    "role": "admin"
                }
            ]
        }
        with open('users.json', 'w', encoding='utf-8') as f:
            json.dump(default_users, f, indent=4)
        return default_users

def authenticate_user(username, password):
    """사용자 인증"""
    users_data = load_users()
    for user in users_data['users']:
        if user['username'] == username and user['password'] == password:
            return True, user
    return False, None

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Prompt Generator") 
    def prompt_view(key="general"):
        prompt_gene_col , prompt_test_col = st.columns(2)
        with prompt_gene_col:
            prompt_template_folder_view(key)
        with prompt_test_col:
            sample_result = llm_test_view(key)
        return sample_result

    # 로그인
    username = st.sidebar.text_input('Username')
    password = st.sidebar.text_input('Password', type='password')

    is_authenticated, user_info = authenticate_user(username, password)
    
    if is_authenticated:
        st.success('Logged in successfully!')
        # 사용자 정보를 세션 상태에 저장
        st.session_state.username = username
        st.session_state.user_role = user_info['role']
        
        # 로거 설정
        logger = setup_logger(username)
        logger.info(f"User {username} logged in with role {user_info['role']}")
        
        # 관리자 기능 추가
        if user_info['role'] == 'admin':
            with st.sidebar.expander("User Management"):
                st.write("Add New User")
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                new_role = st.selectbox("Role", ["user", "admin"])
                
                if st.button("Add User"):
                    users_data = load_users()
                    # 중복 사용자 확인
                    if any(user['username'] == new_username for user in users_data['users']):
                        st.error("Username already exists!")
                    else:
                        users_data['users'].append({
                            "username": new_username,
                            "password": new_password,
                            "role": new_role
                        })
                        with open('users.json', 'w', encoding='utf-8') as f:
                            json.dump(users_data, f, indent=4)
                        st.success(f"User {new_username} added successfully!")
                        logger.info(f"Admin {username} added new user: {new_username}")
        
        with st.container(border=True):
            result_dict = prompt_view("general")
    else:
        if username or password:  # 사용자가 로그인을 시도했을 때만 에러 메시지 표시
            st.error('Invalid username or password')
            logger = setup_logger('system')
            logger.warning(f"Failed login attempt with username: {username}") 
