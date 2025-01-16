from langchain.prompts import PromptTemplate

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
import json
from langchain.prompts.loading import load_prompt_from_config
from langchain.prompts import load_prompt
from pathlib import Path

PromptEvalTemplate = """
I want you to become my Prompt Template  Creator. Your goal is to help me craft the best possible prompt for my needs. The prompt will be used by you, ChatGPT. You will follow the following process:

I will provide my prompt, but we will need to improve it.

Based on my input, you will generate 3 sections;

a) Revised prompt (provide your rewritten prompt; it should be clear, concise, and easily understood by you)

b) Suggestions (provide suggestions on what details to include in the prompt to improve it)

My Prompt Template is 
'''
{prompt_template}
'''
"""

import zipfile  # zipfile 모듈 추가
import os

os.makedirs("./template", exist_ok=True)


def save_prompt_template_local(prompt_template, file_name):
    prompt_template.save(f"./template/{file_name}.json")


def load_prompt_template_local(file_name):
    prompt_template = load_prompt(f"./template/{file_name}")
    return prompt_template


def prompt_template_view(key="general"):
    st.header("Prompt Template View")
    st.session_state.prompt_template = {}
    on = st.toggle("Prompt Template", value=False, key=f"prompt_app_on_{key}")
    input_list = []
    uploaded_file = None
    prompt_string = ""
    if on:
        col1, col2 = st.columns(2)

        with col1:
            on_json = st.toggle("Upload Prompt Template JSON", value=False, key=f"prompt_app_on_json_{key}")
        with col2:
            on_local = st.toggle("Load Prompt Template Local", value=False, key=f"prompt_app_on_local_{key}")
        if on_local:
            file_list = [i.name for i in Path("./template").glob("*.json")]
            file_name = st.selectbox("파일 선택", file_list, index=None, key=f"prompt_app_file_name_{key}")
            if file_name is None:
                st.stop()
            prompt_template = load_prompt_template_local(file_name)
            prompt_string = prompt_template.dict()["template"]
            input_list = prompt_template.input_variables

        if on_json:
            uploaded_file = st.file_uploader("Upload a Prompt Template JSON file", type="json", key=f"prompt_app_uploaded_file_{key}")

            if uploaded_file is not None:
                stringio = uploaded_file.getvalue()
                string_data = stringio.decode("utf-8")

                # JSON으로 변환
                json_data = json.loads(string_data)
                on = st.toggle("Show Prompt Template JSON", value=False, key=f"prompt_app_on_json_{key}")
                if on:
                    st.json(json_data)
                prompt_template = load_prompt_from_config(json_data)
                save_prompt_template_local(prompt_template, uploaded_file.name)
                input_list = prompt_template.input_variables
                prompt_string = prompt_template.dict()["template"]

        prompt_string = st.text_area(
            "Prompt Template",
            value=prompt_string,
            height=500,
            max_chars=None,
            placeholder="Prompt Template을 입력해주세요.",
            key=f"prompt_app_prompt_string_{key}",
        )

        prompt_template = PromptTemplate.from_template(prompt_string)
        input_list = prompt_template.input_variables
        st.session_state.prompt_template["prompt_string"] = prompt_string
        st.session_state.prompt_template["input_list"] = input_list

        new_file_name = st.text_input(label="파일 이름을 입력하세요:", 
                                      placeholder="예) myprompt", key=f"prompt_app_new_file_name_{key}")
        button = st.button("Save Prompt Template", key=f"prompt_app_button_{key}")
        if button:
            file_list = [i.stem for i in Path("./template").glob("*.json")]
            if (new_file_name in file_list) & (len(new_file_name) > 0):
                st.error(f"이미 존재하는 파일 이름입니다.")
                st.error(f"기존 파일 이름: {file_list}")
                st.stop()
            else:
                if len(new_file_name) == 0:
                    st.error("파일 이름을 입력해주세요.")
                    st.stop()
                elif len(prompt_string) == 0:
                    st.error("Prompt Template을 입력해주세요.")
                    st.stop()
                elif (len(prompt_string) > 0) & (len(new_file_name) > 0):
                    save_prompt_template_local(prompt_template, new_file_name)
                    st.download_button(
                        label="Download Promt Template JSON",
                        data=json.dumps(prompt_template.dict()),
                        file_name=f"{new_file_name}.json",
                        mime="application/json",
                        key=f"prompt_app_download_button_{key}",
                    )
                    st.success(
                        f"Prompt Template이 저장되었습니다. {new_file_name}.json(./template/{new_file_name}.json)"
                    )
                else:
                    st.error("Prompt Template 혹은 파일 이름을 입력해주세요.")
                    st.stop()


def prompt_template_folder_view(key="general"):
    st.header("Prompt Template View")
    st.session_state.prompt_template = {}
    input_list = []
    uploaded_file = None
    prompt_string = ""
    folder_list = [d.name for d in Path("./template").iterdir() if d.is_dir()]
    if len(folder_list) == 0:
        st.error("폴더가 없습니다.")
        new_folder_name = st.text_input("폴더 이름을 입력하세요:", placeholder="예) myprompt", key=f"prompt_app_new_folder_name_{key}")
        button = st.button("Create Folder", key=f"prompt_app_button_{key}")
        if button:
            if new_folder_name in folder_list:
                st.error(f"이미 존재하는 폴더 이름입니다.")
                st.error(f"기존 폴더 이름: {folder_list}")
                st.stop()
            else:
                os.makedirs(f"./template/{new_folder_name}", exist_ok=True)
                st.success(f"폴더가 생성되었습니다. {new_folder_name}(./template/{new_folder_name})")
                st.rerun()
    else:
        if folder_list is None:
            folder_list = ["신규 프로젝트 폴더 추가"]
        else :
            folder_list = folder_list + ["신규 프로젝트 폴더 추가"]
        folder_name = st.selectbox("폴더 선택", folder_list, index=None, key=f"prompt_app_folder_name_{key}")
        if folder_name == "신규 프로젝트 폴더 추가":
            new_folder_name = st.text_input("폴더 이름을 입력하세요:", placeholder="예) myprompt", key=f"prompt_app_new_folder_name_{key}_2")
            button = st.button("Create Folder", key=f"prompt_app_button_{key}_2")
            if button:
                os.makedirs(f"./template/{new_folder_name}", exist_ok=True)
                st.success(f"폴더가 생성되었습니다. {new_folder_name}(./template/{new_folder_name})")
                st.rerun()
        else :
            folder_name = folder_name
    col1, col2 = st.columns(2)
    with col1:
        on = st.toggle("Show File List in Folder", value=False, key=f"prompt_app_on_file_list_{key}")
    with col2:
        on_download = st.toggle("Download Folder", value=False, key=f"prompt_app_on_download_{key}")
    if on:
        file_list = [i.name for i in Path(f"./template/{folder_name}").glob("*.json")]
        st.markdown(f"### {folder_name} 폴더 내의 파일 목록")
        if file_list:
            for file in file_list:
                st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px; margin: 5px; border-radius: 5px;'>"
                             f"<strong>파일 이름:</strong> {file}</div>", unsafe_allow_html=True)
        else:
            st.markdown("파일이 없습니다.")

    if on_download:
        zip_file_path = f"./template/{folder_name}.zip"  # 압축 파일 경로 설정
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for json_file in Path(f"./template/{folder_name}").glob("*.json"):  # 폴더 내의 모든 JSON 파일 추가
                zipf.write(json_file, arcname=json_file.name)  # 파일 이름만 저장
        with open(zip_file_path, "rb") as f:
            st.download_button(label="Download Folder as ZIP",
                            data=f,
                            file_name=f"{folder_name}.zip",
                            mime="application/zip")

    st.session_state.prompt_template["folder_name"] = folder_name
    on = st.toggle("Prompt Template", value=False, key=f"prompt_app_on_{key}")
    if on:
        col1, col2 = st.columns(2)

        with col1:
            on_json = st.toggle("Upload Prompt Template JSON", value=False, key=f"prompt_app_on_json_{key}")
        with col2:
            on_local = st.toggle("Load Prompt Template Local", value=False, key=f"prompt_app_on_local_{key}")
        if on_local:
            file_list = [i.name for i in Path("./template").joinpath(folder_name).glob("prompt*.json")]            
            file_name = st.selectbox("파일 선택", file_list, index=None, key=f"prompt_app_file_name_{key}")
            if file_name is None:
                st.stop()
            prompt_template = load_prompt_template_local(f"{folder_name}/{file_name}")
            prompt_string = prompt_template.dict()["template"]
            input_list = prompt_template.input_variables

        if on_json:
            uploaded_file = st.file_uploader("Upload a Prompt Template JSON file", type="json", key=f"prompt_app_uploaded_file_{key}")

            if uploaded_file is not None:
                stringio = uploaded_file.getvalue()
                string_data = stringio.decode("utf-8")

                # JSON으로 변환
                json_data = json.loads(string_data)
                on = st.toggle("Show Prompt Template JSON", value=False, key=f"prompt_app_on_json_{key}")
                if on:
                    st.json(json_data)
                prompt_template = load_prompt_from_config(json_data)
                save_prompt_template_local(prompt_template, f"{folder_name}/prompt.json")
                input_list = prompt_template.input_variables
                prompt_string = prompt_template.dict()["template"]

        prompt_string = st.text_area(
            "Prompt Template",
            value=prompt_string,
            height=300,
            max_chars=None,
            placeholder="Prompt Template을 입력해주세요.",
            key=f"prompt_app_prompt_string_{key}",
        )

        prompt_template = PromptTemplate.from_template(prompt_string)
        input_list = prompt_template.input_variables
        st.session_state.prompt_template["prompt_string"] = prompt_string
        st.session_state.prompt_template["input_list"] = input_list
        new_file_name = st.text_input("파일 이름을 입력하세요:", placeholder="예) prompt_1", key=f"prompt_app_new_file_name_{key}")
        button = st.button("Save Prompt Template", key=f"prompt_app_button_{key}")
        if button:
            file_list = [i.stem for i in Path("./template").glob("*.json")]
            
            if len(prompt_string) == 0:
                st.error("Prompt Template을 입력해주세요.")
                st.stop()
            elif not new_file_name.startswith("prompt") :
                st.error("파일 이름은 prompt로 시작해야 합니다.")
                st.stop()
            elif (len(prompt_string) > 0) :
                file_list = [i.stem for i in Path("./template").joinpath(folder_name).glob("prompt*.json")]
                if new_file_name in file_list:
                    st.error(f"이미 존재하는 파일 이름입니다.")
                    st.error(f"기존 파일 이름: {file_list}")
                    st.stop()
                save_prompt_template_local(prompt_template, f"{folder_name}/{new_file_name}")
                st.download_button(
                    label="Download Promt Template JSON",
                    data=json.dumps(prompt_template.dict()),
                    file_name=f"{folder_name}/{new_file_name}.json",
                    mime="application/json",
                    key=f"prompt_app_download_button_{key}",
                )
                st.success(
                    f"Prompt Template이 저장되었습니다. {folder_name}/prompt.json(./template/{folder_name}/prompt.json)"
                )
            else:
                st.error("Prompt Template 혹은 파일 이름을 입력해주세요.")
                st.stop()


def llm_test_view():
    st.header("LLM Test View")
    input_list = st.session_state.prompt_template.get("input_list", [])
    input_values = st.session_state.prompt_template.get("input_values", {})

    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input("API 키를 입력하세요:", type="password", key="prompt_app_api_key")
    with col2:
        model = st.selectbox(
            "모델 선택",
            ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4-1106-preview"],
            key="prompt_app_model",
        )

    tab1, tab2 = st.tabs(["Prompt Example", "LLM Eval"])
    with tab1:
        if len(input_list) > 0:
            input_values = {}
            # 입력 필드를 두 개씩 나누어 배치
            tab_name_list = [col for col in input_list]
            tab_list = st.tabs(tab_name_list + [":star: Final PROMPT :star:"])
            for tab_name, tab in zip(tab_name_list, tab_list[:-1]):
                col = tab_name
                with tab:
                    height = st.number_input("Height", 100, 1000, 300, key=f"{col}_height")
                    input_values[col] = st.text_area(f"{col}", "", height=height)
            st.session_state.prompt_template["input_values"] = input_values
        else:
            tab_list = st.tabs([":star: Final PROMPT :star:"])
        tab = tab_list[-1]
        with tab:
            height = st.number_input("Height", 100, 1000, 400)
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

            prompt = st.text_area("Prompt Example", prompt, height=height)

    with tab2:
        llm = ChatOpenAI(openai_api_key=api_key, model=model) if api_key else None
        col1, col2 = st.columns(2)
        on = st.toggle("Show Prompt", value=False)
        if on:
            st.text_area("Prompt", prompt, height=height)
        with col1:
            button_llm_test = st.button("테스트")
        with col2:
            button_llm_eval = st.button("프롬프트 평가")
        if button_llm_test and llm:
            with get_openai_callback() as cb:
                sample_result = llm.invoke(prompt).content
                # Using beta_columns to create a more structured layout
                st.session_state.prompt_template['output'] = sample_result
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

        if button_llm_eval and llm:
            with get_openai_callback() as cb:
                prompt_eval_template = PromptTemplate.from_template(template=PromptEvalTemplate)

                eval_prompt = prompt_eval_template.format(prompt_template=prompt_string)
                for input in input_list:
                    eval_prompt = eval_prompt.format(**{input: "[variable]"})
                print(eval_prompt)
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


def user_guide_view(key="general"):
    prompt_str = st.session_state.prompt_template.get("prompt_string", "")
    input_list = st.session_state.prompt_template.get("input_list", [])
    input_values = st.session_state.prompt_template.get("input_values", {})
    st.header("User Guide(LangChain)")
    temperature = st.session_state.prompt_template.get("temperature", 0.0)
    max_tokens = st.session_state.prompt_template.get("max_tokens", 1000)
    model = st.session_state.prompt_template.get("model", "gpt-4o-mini-2024-07-18")
    folder_name = st.session_state.prompt_template.get("folder_name", "general")
    input_text = ""
    input_arg_text = ""
    for input in input_list:
        input_text += f"{input} = '''{input_values.get(input, '')}'''\n"
        input_arg_text += f"{input}={input}, "
    input_arg_text = input_arg_text.rsplit(",", 1)[0]
    if len(prompt_str) > 0:
        code = (
            "from langchain.prompts import PromptTemplate\n"
            "from langchain_openai import ChatOpenAI\n"
            "import os\n"
            "#from langchain.prompts import load_prompt\n"
            "os.environ['OPENAI_API_KEY'] = 'sk-######'\n"
            f"prompt_str = '''{prompt_str}'''\n"
            "template = PromptTemplate.from_template(prompt_str)\n"
            "#template = load_prompt('myprompt.json') # PromptTemplate\n"
            f"{input_text}\n"
            f"prompt = template.format({input_arg_text})\n"
            f"json_schema = {st.session_state.prompt_template.get('json_schema', {})}\n"
            f"llm = ChatOpenAI(model='{model}', temperature={temperature}, max_tokens={max_tokens})\n"
            "llm_structured = llm.with_structured_output(json_schema)\n"
            "llm_result = llm_structured.invoke(prompt)\n"
        )
        st.code(code, language="python", line_numbers=True)
        # save code
        if st.button("Save Code", key=f"prompt_app_save_code_{key}"):
            with open(f"./template/{folder_name}/code.py", "w") as f:
                f.write(code)
            st.success(f"Code가 저장되었습니다. {folder_name}/code.py(./template/{folder_name}/code.py)")
    else:
        st.error("Prompt Template을 입력해주세요.")


import pandas as pd


def make_prompt_file():
    prompt_str = st.session_state.prompt_template.get("prompt_string", "")
    if len(prompt_str) == 0:
        st.error("Prompt Template을 입력해주세요.")
        st.stop()
    input_list = st.session_state.prompt_template.get("input_list", [])
    if len(input_list) == 0:
        st.error("해당 Prompt Template에 입력 변수가 없습니다.")
        st.stop()
    input_values = st.session_state.prompt_template.get("input_values", {})
    st.header("Generate Prompt File Based on Prompt Template")
    # csv (input variables)
    encoding_options = ["utf-8", "utf-8-sig", "ISO-8859-1", "cp1252", "utf-16", "ASCII", "cp949", "latin1"]
    selected_encoding = st.selectbox("Choose file encoding", options=encoding_options, index=0)  # 기본값은 'utf-8'
    uploaded_csv_file = st.file_uploader("Upload a Input Variables CSV file", type="csv")
    if uploaded_csv_file is None:
        st.write("Input Variables CSV 파일을 업로드해주세요.")
        st.stop()
    else:
        try:
            data = pd.read_csv(uploaded_csv_file, encoding=selected_encoding)
        except Exception as e:
            st.error(f"CSV 파일을 읽는데 실패했습니다. {e}")
            st.stop()
        candidate_input_list = data.columns.tolist()
        intersection_cols = list(set(input_list) & set(candidate_input_list))
        if set(input_list) != set(intersection_cols):
            st.error(f"Input Variables CSV 파일의 컬럼명이 Prompt Template의 입력 변수와 일치하지 않습니다.")
            st.error(f"Prompt Template의 입력 변수: {input_list}")
            st.error(f"Input Variables CSV 파일의 컬럼명: {candidate_input_list}")
            st.stop()
        else:
            on_test = st.toggle("Test Prompt", value=False)
            if on_test:
                select_row_idx = st.number_input(
                    "테스트할 행을 선택하세요:",
                    min_value=0,
                    max_value=len(data) - 1,
                    value=0,
                    key="prompt_app_templated_select_row_idx",
                )
                prompt_template = PromptTemplate.from_template(template=prompt_str)
                sample_prompt = prompt_template.format(**data.iloc[select_row_idx].to_dict())
                tab1, tab2 = st.tabs(["Template", "Sample Prompt"])
                with tab1:
                    with st.container(border=True):
                        st.markdown(f"Template: {prompt_template.template}")
                        st.markdown(f"Input Variables: {input_list}")
                with tab2:
                    with st.container(border=True):
                        st.markdown(f"Sample Prompt: {sample_prompt}")
        col1, col2 = st.columns(2)

        on_run = st.toggle("Run(Make Prompt File)")
        col1, col2 = st.columns(2)

        if on_run:
            prompt_template = PromptTemplate.from_template(template=prompt_str)
            prompt_list = []
            for idx, row in data.iterrows():
                prompt = prompt_template.format(**row.to_dict())
                prompt_list.append(prompt)
            data["prompt"] = prompt_list
            st.session_state.prompt_template["data"] = data
            st.dataframe(data)
            selected_encoding = st.selectbox(
                "Choose file encoding", options=encoding_options, index=0
            )  # 기본값은 'utf-8'
            with col1:
                st.download_button(
                    label="Download Promt CSV",
                    data=data.to_csv(index=False, encoding=selected_encoding),
                    file_name="prompt.csv",
                    mime="text/csv",
                )
            with col2:
                file_name = st.text_input(label="파일 이름을 입력하세요:", placeholder="prompt.csv")
                upload_button = st.button("Upload Prompt File for Data Generation")

            if file_name is None:
                st.stop()
            elif ".csv" not in file_name:
                st.error("파일 이름에 .csv를 포함해주세요.")
            else:
                pass

            if upload_button:
                file_list = [i.name for i in Path("./csv").glob("*.csv")]
                if file_name in file_list:
                    st.error(f"이미 존재하는 파일 이름입니다.")
                    st.error(f"기존 파일 이름: {file_name}")
                    st.error(f"기존 파일 리스트: {file_list}")
                else:
                    data.to_csv(f"./csv/{file_name}", index=False, encoding=selected_encoding)
                    st.success(f"Prompt CSV가 저장되었습니다. {file_name}(./csv/{file_name})")


def prompt_example_list():
    # 특정 경로 설정
    path = Path("./prompt_store/")

    # 해당 경로에서 하위 폴더만 뽑아 리스트로 만들기
    folder_list = [d.name for d in path.iterdir() if d.is_dir()]

    folder_select = st.selectbox("Select Folder", folder_list, index=None)
    if folder_select is None:
        st.warning("폴더를 선택해주세요.")
    prompt_file_path_list = list(path.joinpath(f"{folder_select}").glob("*.md"))
    if len(prompt_file_path_list) == 0:
        prompt_name_list = []
        st.warning("해당 폴더에 프롬프트 파일이 없습니다.")
    else:
        prompt_name_list = [i.stem for i in prompt_file_path_list]
    prompt_name = st.selectbox("Select Prompt", prompt_name_list, index=None)
    if prompt_name is None:
        st.warning("프롬프트를 선택해주세요.")
    # 파일 읽기
    if path.joinpath(f"{folder_select}").joinpath(f"{prompt_name}.md").is_file():

        with open(path.joinpath(f"{folder_select}").joinpath(f"{prompt_name}.md"), "r", encoding="utf-8-sig") as file:
            md_content = file.read()

        # Streamlit 애플리케이션에 Markdown 내용 표시
        st.markdown(md_content, unsafe_allow_html=True)
    else:
        pass


def prompt_example_main():
    st.session_state.prompt_template = {
        "prompt_string": "",
        "input_list": [],
        "input_values": {},
    }
    with st.expander("프롬프트 예시"):
        prompt_example_list()
    with st.expander("프롬프트 템플릿"):
        prompt_template_view()
    with st.expander("프롬프트 테스트"):
        llm_test_view()
    with st.expander("프롬프트 사용자 가이드"):
        user_guide_view()
    with st.expander("프롬프트 파일 생성"):
        make_prompt_file()

    # st.sidebar.title("Prompt Navigation")
    # selection = st.sidebar.radio("Go to", ["프롬프트 템플릿", "프롬프트 테스트"])

    # if selection == "프롬프트 템플릿":
    #     prompt_template_view()
    # elif selection == "프롬프트 테스트":
    #     llm_test_view()


if __name__ == "__main__":
    prompt_example_main()
