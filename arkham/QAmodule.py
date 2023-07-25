from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline, PromptTemplate
from peft import PeftConfig

import torch

from arkham.utils import create_sbert_mpnet
from arkham.constants import falcon_template


class QA_assistant:
    """
    Class that holds the logic to create and mantain a Q&A with the chosen LLM regarding a given file.
    The current models available to use are `Falcon7b-Instruct`, a fine tuned `Falcon7b` and `GPT3.5`
    from OpenAI.
    """

    def __init__(self, file_path: str, model: str) -> None:
        """
        Method that initialates the class.

        Args:
            file_path (str): Path of the text file that will be queried on.
            model (str): Model that will be used to retrieve the information from the file.
        """
        self.file_path = file_path
        self.model = model
        self.models = ["Falcon7b-Tuned", "Falcon7b-Instruct", "GPT3.5"]
        assert model in self.models, f"Current available models are {self.models}"

    def initiate_gpt(self):
        """
        Method that initiates the gpt logic. HAS to be run before `query_gpt`.
        """
        loader = TextLoader(self.file_path)
        self.index = VectorstoreIndexCreator().from_loaders([loader])

    def query_gpt(self, question: str) -> str:
        """Method that wraps around langchain's index creator. This allows to retrieve answers
        from the `GPT3.5` model.

        Args:
            question (str): Question that's being asked to the LLM.

        Returns:
            str: Answer from the model.
        """
        return self.index.query(question)

    def initate_Falcon(self, model) -> None:
        """
        Method that initiates the gpt logic. HAS to be run before `query_falcon.`. It does the following steps:
            1. Loads the word embedding.
            2. Loads the document and tokenizes it.
            3. Splits the text file into tokens.
            4. Creates the vector space to model the tokens from the file.
            5. Creates the pipeline for the model. Loads all the parameters and loads the vector space.
            6. Loads the QA prompt and sets up the outputs.
        Args:
            model (str): Name of the model that will be loaded.
        """
        # Load embedding
        embedding = create_sbert_mpnet()
        # Load Files from txt
        loader = TextLoader(self.file_path)
        documents = loader.load()
        # Splitt text file
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        text_splitter = TokenTextSplitter(
            chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base"
        )  # This the encoding for text-embedding-ada-002
        texts = text_splitter.split_documents(texts)
        # Create vector space
        vectordb_from_text = Chroma.from_documents(
            documents=texts, embedding=embedding, persist_directory=None
        )
        # Define tokenizer for the pipeline
        tokenizer = AutoTokenizer.from_pretrained(model)
        # Create model pipeline
        hf_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            max_new_tokens=100,
            model_kwargs={
                "device_map": "auto",
                "load_in_8bit": True,
                "max_length": 512,
                "pad_token_id": 11,
                "torch_dtype": torch.bfloat16,
                "temperature": 0.0,
            },
        )

        hf_llm = HuggingFacePipeline(pipeline=hf_pipeline)
        retriever = vectordb_from_text.as_retriever(search_kwargs={"k": 4})
        self.qa = RetrievalQA.from_chain_type(
            llm=hf_llm, chain_type="stuff", retriever=retriever
        )
        # Define QA prompt
        QUESTION_FALCON_PROMPT = PromptTemplate(
            template=falcon_template, input_variables=["context", "question"]
        )
        self.qa.combine_documents_chain.llm_chain.prompt = QUESTION_FALCON_PROMPT
        self.qa.combine_documents_chain.verbose = False

    def query_falcon(self, question: str) -> str:
        """Method that wraps around HuggingFacePipeline. This allows to retrieve answers
        from the Falcon models.

        Args:
            question (str): Question that's being asked to the LLM.

        Returns:
            str: Answer from the model.
        """
        return self.qa(
            {
                "query": question,
            }
        )

    def get_querier(self):
        """Funciton that wraps the whole class. This function is used to get the proper querier
        method after instanciating the class.

        Returns:
            function: Method to call the queries on.
        """
        if self.model == "GPT3.5":
            self.initiate_gpt()

        elif "Falcon7b" in self.model:
            if self.model == "Falcon7b-Instruct":
                model = "tiiuae/falcon-7b-instruct"
            else:
                config = PeftConfig.from_pretrained("./checkpoint-240")
                model = config.base_model_name_or_path

            self.initate_Falcon(model=model)

        return self.query_manager(model=self.model)

    def query_manager(self, model: str):
        """Helper method to return the querier.

        Args:
            model (str): Model that's going to be used.

        Returns:
            function: Method to call the queries on.
        """
        if model == "GPT3.5":
            return self.query_gpt
        else:
            return self.query_falcon
