from src.pdf_http_converter import PdfHttpConverter
from src.llm_completition import DetectionDrugsLLMAction, DrugQALLMAction
from src.drug_database_info import DrugDatabaseInfo

from haystack.components.preprocessors import DocumentSplitter
# from src.document_splitter import

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

from huggingface_hub import InferenceClient

from haystack import Document

import re

class ChatBot:

    def __init__(self, name):
        print(f'Initializing {name}\n==================================', end='\n\n')
        self.name = name

        self.client = InferenceClient(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        
        self.retriever = None
        self.cached_file = None
        self.previous_drugs = None

    # detect drug
    def detect_drug(self, query):
        drug_detector = DetectionDrugsLLMAction(self.client)

        drugs = drug_detector.complete(query)

        return drugs

    # find info in the database
    def get_pdf_info(self, drugs_detected):
        db = DrugDatabaseInfo()

        drug_name = drugs_detected['substancias_listadas'][-1]

        return db.find_information(drug_name)

    def get_pdf_from_link(self, link):
        converter = PdfHttpConverter()

        document = converter.link_to_document(link)

        return document

    # download pdf
    def setup_retriever(self, pdf_file, pdf_info):

        splitter = DocumentSplitter(split_by="word", split_length=200, split_overlap=50)

        splits = splitter.run([pdf_file])['documents']

        document_store = InMemoryDocumentStore()

        document_store.write_documents(splits)

        self.retriever = InMemoryBM25Retriever(document_store=document_store)

    # split file

    # store docs

    # retriever
    def clear_response(self, text):
        text = text.replace('\n', ' ')
        text = re.sub(' +', ' ', text)

        parts = text.split('.')

        if len(parts) == 1:
            return parts[0] + '.' 

        return ".".join(parts[0:-1])+'.'


    def respond(self, message):
        drugs = self.detect_drug(message)

        pdf_info = self.get_pdf_info(drugs)

        if drugs != self.previous_drugs:
            documents = self.get_pdf_from_link(pdf_info.ficha_tecnica)
            self.setup_retriever(documents, pdf_info)

        context_documents =self.retriever.run(query=message, top_k=5)['documents']

        pdf_info = pdf_info.to_json().replace('_', ' ')
        context_documents = [Document(id='pdf_info', content=pdf_info)] + context_documents

        context = "\n\n".join(['<s>' + doc.content+ '</s>' for doc in context_documents])

        drug_qa = DrugQALLMAction(self.client)

        response = drug_qa.complete(message, context)

        return self.clear_response(response)
            
        
