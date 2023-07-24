from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter 
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer,pipeline
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline,PromptTemplate
from peft import PeftConfig

import torch

from arkham.utils import create_sbert_mpnet
from arkham.constants import falcon_template

class QA_assistant():
    def __init__(self,file_path,model) -> None:
        self.file_path = file_path
        self.model = model
        self.models = ['Falcon7b-Tuned','Falcon7b-Instruct','GPT3.5']
        assert model in self.models, f'Current available models are {self.models}'

    def initiate_gpt(self):
        loader = TextLoader(self.file_path)
        self.index = VectorstoreIndexCreator().from_loaders([loader])
        
    def query_gpt(self,question):
        return self.index.query(question)
    
    def initate_Falcon(self,model):
        embedding = create_sbert_mpnet()
        
        loader = TextLoader(self.file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
        texts = text_splitter.split_documents(texts)

        vectordb_from_text = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=None)
        
        tokenizer = AutoTokenizer.from_pretrained(model)

        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                tokenizer = tokenizer,
                trust_remote_code = True,
                max_new_tokens=100,
                model_kwargs={
                        "device_map": 'auto', 
                        "load_in_8bit": True, 
                        "max_length": 512,
                        'pad_token_id': 11,
                        "torch_dtype":torch.bfloat16,
                        'temperature' : 0.0,
                        }
                )
        
        hf_llm = HuggingFacePipeline(pipeline=hf_pipeline)
        retriever = vectordb_from_text.as_retriever(search_kwargs={"k":4})
        self.qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff",retriever=retriever)

        QUESTION_FALCON_PROMPT = PromptTemplate(
            template=falcon_template, input_variables=["context","question"]
        )
        self.qa.combine_documents_chain.llm_chain.prompt = QUESTION_FALCON_PROMPT
        self.qa.combine_documents_chain.verbose = False
    
    def query_falcon(self,question):
        return self.qa({"query":question,})
    
    def get_querier(self):
        if self.model == 'GPT3.5':
            self.initiate_gpt()
        
        elif 'Falcon7b' in self.model:
            if self.model == 'Falcon7b-Instruct':
                model = "tiiuae/falcon-7b-instruct" 
            else:
                config = PeftConfig.from_pretrained('./checkpoint-240')
                model = config.base_model_name_or_path

            self.initate_Falcon(model=model)
            
        return self.query_manager(model=self.model)

    def query_manager(self,model):
        if model == 'GPT3.5':
            return self.query_gpt
        else:
            return self.query_falcon