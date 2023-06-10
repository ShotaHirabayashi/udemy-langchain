from typing import List

import langchain
from langchain import VectorDBQAWithSourcesChain
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool


langchain.verbose = True


def create_index() -> VectorStoreIndexWrapper:
    loader = DirectoryLoader("./data/", glob="**/*.txt")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
    )
    return VectorstoreIndexCreator(text_splitter=text_splitter).from_loaders([loader])


# def chat(message: str, history: ChatMessageHistory) -> str:
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#     messages = history.messages
#     messages.append(HumanMessage(content=message))
#     return llm(messages).content


def create_tools(index: VectorStoreIndexWrapper) -> List[BaseTool]:
    vectorstore_info = VectorStoreInfo(
        vectorstore=index.vectorstore,
        name="Wグループ従業員マニュアル",
        description="Wグループの従業員用のマニュアル",
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    return toolkit.get_tools()


def chat(message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper) -> str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, top_p=0.9)
    tools = create_tools(index)
    # memory = ConversationBufferMemory(
    #     chat_memory=history, memory_key="chat_history", return_messages=True,
    # )
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    return agent_chain.run(input=message)
