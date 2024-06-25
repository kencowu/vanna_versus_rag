import time
import streamlit as st
from random import randint
# from code_editor import code_editor
from vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached
)
from hackthon_faiss_embedding import *
from hackthon_vanna_rag import *
from hackthon_langchain_retriever import *
import certifi
certifi.where()

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("API_KEY")

avatar_url = "https://vanna.ai/img/vanna.svg"


st.set_page_config(layout="wide")

st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
# st.sidebar.checkbox("Show Summary", value=True, key="show_summary")
# st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")
st.sidebar.checkbox("GPT Integration", value=True, key="gpt_integration")
st.sidebar.checkbox("RAG Integration", value=True, key="rag_integration")
st.sidebar.button("Reset", on_click=lambda: set_question(None), use_container_width=True)

st.title("Vanna AI")
# st.sidebar.write(st.session_state)


st.session_state["user_input_cursor"] = 1


def set_question(question):
    st.session_state["my_question"] = question


assistant_message_suggested = st.chat_message(
    "assistant", avatar=avatar_url
)
if assistant_message_suggested.button("Click to show suggested questions"):
    st.session_state["my_question"] = None
    questions = generate_questions_cached()
    for i, question in enumerate(questions):
        time.sleep(0.05)
        button = st.button(
            question,
            on_click=set_question,
            args=(question,),
        )

my_question = st.session_state.get("my_question", default=None)

if my_question is None and st.session_state["user_input_cursor"] == 1:
    my_question = st.chat_input(
        "Ask me a question to create your dataframe.",
        key=1
    )


if my_question:
    st.session_state["my_question"] = my_question
    user_message = st.chat_message("user")
    user_message.write(f"{my_question}")

    sql = generate_sql_cached(question=my_question)

    if sql:
        if is_sql_valid_cached(sql=sql):
            if st.session_state.get("show_sql", True):
                assistant_message_sql = st.chat_message(
                    "assistant", avatar=avatar_url
                )
                assistant_message_sql.code(sql, language="sql", line_numbers=True)
        else:
            assistant_message = st.chat_message(
                "assistant", avatar=avatar_url
            )
            assistant_message.write(sql)
            st.stop()

        df = run_sql_cached(sql=sql)

        if df is not None:
            st.session_state["df"] = df

            #################### Hackthon ####################

            # st.session_state["user_input_cursor"] = 2

            #################### Hackthon ####################

        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                assistant_message_table = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )
                if len(df) > 10:
                    assistant_message_table.text("First 10 rows of data")
                    assistant_message_table.dataframe(df.head(10))
                    st.session_state["user_input_cursor"] = 2
                else:
                    assistant_message_table.dataframe(df)

            if should_generate_chart_cached(question=my_question, sql=sql, df=df):

                code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                if st.session_state.get("show_plotly_code", False):
                    assistant_message_plotly_code = st.chat_message(
                        "assistant",
                        avatar=avatar_url,
                    )
                    assistant_message_plotly_code.code(
                        code, language="python", line_numbers=True
                    )

                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                        assistant_message_chart = st.chat_message(
                            "assistant",
                            avatar=avatar_url,
                        )
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            assistant_message_chart.plotly_chart(fig)
                        else:
                            assistant_message_chart.error("I couldn't generate a chart")

            # if st.session_state.get("show_summary", False):
            #     assistant_message_summary = st.chat_message(
            #         "assistant",
            #         avatar=avatar_url,
            #     )
            #     summary = generate_summary_cached(question=my_question, df=df)
                
            #     if summary is not None:
            #         assistant_message_summary.text(summary)


                #################### hackthon_vanna_rag ####################

            # Add GPT integration session
            if st.session_state.get("gpt_integration", True):
                assistant_message_gpt = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )

                # st.session_state["gpt_question"] = None
                gpt_question = st.session_state.get("gpt_question", default=None)
                input_key=2
                
                if gpt_question is None and st.session_state["user_input_cursor"] == 2:
                    print('A')
                    gpt_question = st.chat_input(
                        "Ask a question about your generated data",
                        key=input_key
                        )
                    st.session_state["gpt_question"] = gpt_question
                    print('B')
                    print(gpt_question)
                if gpt_question:
                    gpt_answer = get_gpt_response(df=df, user_prompt=gpt_question)
                    input_key+=1
                    gpt_question=None
                    assistant_message_gpt.text(gpt_answer)
                    gpt_answer=None

                if st.sidebar.button('Generate Vector Database', use_container_width=True):
                    json_rows = df_to_structural_json(df)
                    create_faiss_embeddings(json_rows)
                    st.write('Vector database created.')
                    st.session_state["user_input_cursor"] = 3


            # # Add RAG integration session
            # if st.session_state.get("rag_integration", True):
            #     assistant_message_rag = st.chat_message(
            #         "assistant",
            #         avatar=avatar_url,
            #     )

            #     st.session_state["rag_question"] = None
            #     rag_question = st.session_state.get("rag_question", default=None)
            #     input_key=3
            #     print("00000")
            #     if rag_question is None and st.session_state["user_input_cursor"] == 3:
            #         print("00001")
            #         rag_question = st.chat_input(
            #             "Ask a question to you RAG model",
            #             key=input_key
            #             )
            #         print(rag_question)
            #         st.session_state["rag_question"] = rag_question
            #         print("00002")
            #         rag_question = 'Summarize the data'
            #     print("00003")
            #     print(rag_question)

            #     if rag_question:
            #         print("00004")
            #         rag_result = get_answer_from_faiss_gpt(rag_question)
            #         print("00005")
            #         assistant_message_rag.text(rag_result)

            # Add RAG integration session 
            if st.session_state.get("rag_integration", True):
                assistant_message_rag = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )

                # st.session_state["rag_question"] = None
                rag_question = st.session_state.get("rag_question", default=None)
                input_key=3
                print('D')
                if rag_question is None and st.session_state["user_input_cursor"] == 3:
                    rag_question = st.chat_input(
                        "Ask a question about your RAG data",
                        key=input_key
                        )
                    print(rag_question)
                    st.session_state["rag_question"] = rag_question
                print('E')
                print(rag_question)
                if rag_question:
                    rag_answer = get_answer_from_faiss_gpt(rag_question)
                    input_key+=1
                    rag_question=None
                    assistant_message_rag.text(rag_answer)
                    rag_answer=None




                #################### hackthon_vanna_rag ####################


            # if st.session_state.get("show_followup", False):
            #     assistant_message_followup = st.chat_message(
            #         "assistant",
            #         avatar=avatar_url,
            #     )
            #     followup_questions = generate_followup_cached(
            #         question=my_question, sql=sql, df=df
            #     )
            #     st.session_state["df"] = None

            #     if len(followup_questions) > 0:
            #         assistant_message_followup.text(
            #             "Here are some possible follow-up questions"
            #         )
            #         # Print the first 5 follow-up questions
            #         for question in followup_questions[:5]:
            #             assistant_message_followup.button(question, on_click=set_question, args=(question,))

    else:
        assistant_message_error = st.chat_message(
            "assistant", avatar=avatar_url
        )
        assistant_message_error.error("I wasn't able to generate SQL for that question")
