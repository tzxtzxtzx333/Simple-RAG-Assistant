import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量获取 Key，如果没找到，就设为 None
API_KEY = os.getenv("DEEPSEEK_API_KEY") 
BASE_URL = "https://api.deepseek.com"

# 增加一个检查：如果没 Key，提示用户
if not API_KEY:
    st.error("⚠️ 未检测到 API Key！请在项目根目录创建 .env 文件并配置 DEEPSEEK_API_KEY")
    st.stop()

st.set_page_config(page_title="RAG 智能知识库", page_icon="📂")
st.title("📂 RAG 智能知识库助手")

# === 核心函数：处理文件并初始化 RAG ===
# 这里去掉了 @st.cache_resource，因为每次换文件都要重新处理
def process_uploaded_file(uploaded_file):
    # 1. 把上传的文件（内存对象）存成临时文件（硬盘文件）
    # 这是软工里常见的 IO 操作，因为很多库只认文件路径
    temp_filename = "temp_uploaded.pdf"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    st.info(f"正在分析文档: {uploaded_file.name} ...")
    
    # 2. 加载 & 切片
    loader = PyPDFLoader(temp_filename)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # 3. 向量化 (这步最耗时)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # 为了演示简单，每次都新建一个临时的内存数据库
    db = Chroma.from_documents(texts, embeddings)
    
    # 4. 初始化 LLM
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0
    )
    
    # 5. 构建链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )
    
    return qa_chain

# === 侧边栏：文件上传 ===
with st.sidebar:
    st.header("1. 上传文档")
    uploaded_file = st.file_uploader("请上传 PDF 文件", type=["pdf"])
    
    # 初始化 session_state (状态管理)
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    # 如果用户上传了文件，就开始处理
    if uploaded_file is not None:
        # 避免重复处理：只有当现在的 chain 是空的时候，或者换了新文件时才处理
        # (这里简化逻辑，只要有文件就重新处理一次，确保稳定)
        if st.button("开始分析文档"):
            with st.spinner("AI 正在阅读，请稍等..."):
                try:
                    st.session_state.qa_chain = process_uploaded_file(uploaded_file)
                    st.success("✅ 分析完成！请在右侧提问。")
                except Exception as e:
                    st.error(f"出错啦: {e}")

# === 主界面：聊天 ===
st.header("2. 智能问答")

# 如果没有 Chain，提示用户先上传
if st.session_state.qa_chain is None:
    st.info("👈 请先在左侧上传一个 PDF 文件并点击“开始分析”。")
else:
    # 聊天记录显示
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 输入框
    prompt = st.chat_input("在这个文档里搜索...")

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke({"query": prompt})
                st.write(response['result'])
                st.session_state.messages.append({"role": "assistant", "content": response['result']})