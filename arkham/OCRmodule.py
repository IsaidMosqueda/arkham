from unstructured.partition.pdf import partition_pdf
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

from typing import Union
import os

class ocr_parse_file():
    def __init__(self,file_path:str) -> None:
        self.file_path = file_path
        self.methods = ['hi_res','fast','secure']
        self.paged_text = []

    def load_file(self,method:str='hi_res',file_language:str='spa',return_obj:bool = False):
        assert method in self.methods, f"The current reading methods available are {self.methods}"
        if method == 'hi_res':
            try:
                self.files = partition_pdf(self.file_path, ocr_languages=file_language, strategy="hi_res")
            except Exception as e:
                print(e) # this should go in debugging level
                print(f'Method {method} failed, atempting method "fast"')
                method = 'fast'

        if method == 'fast':
            try:
                self.files = partition_pdf(self.file_path, ocr_languages=file_language, strategy="fast")
            except Exception as e:
                method = 'secure'
                print(f'Method {method} failed, atempting method "secure"')

        if method == 'secure':
            pdf_reader = PyPDF2.PdfReader(open(self.file_path, 'rb'))
            self.files = []
            for page_num in range(len(pdf_reader.pages)):
                images = convert_from_path(self.pdf_path, first_page=page_num, last_page=page_num + 1)
                text = pytesseract.image_to_string(images[0], lang=file_language)
                self.files.append(text)

        if method == 'secure':
            for i, page in enumerate(self.files):
                self.paged_text.append(f'Pagina {i+1}\n {page}')
        elif method in ['hi_res','fast']:
            curr_page = 0
            curr_page_text = ''
            for line in self.files:
                # Separate between pages
                if curr_page != line.metadata.page_number:
                    self.paged_text.append(str(curr_page_text))
                    curr_page = line.metadata.page_number
                    curr_page_text = (f'Page {curr_page}\n')
                curr_page_text+=f'\n{line.text}'

        if return_obj:
                return self.paged_text
        
    def write_file(self,write_path:str='parsed/'):
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        write_path = write_path+self.file_path[:-4]+'.txt'
        with open(write_path,'w') as to_save:
            to_save.write('\n\n|'.join(self.paged_text))
            to_save.close()
        return write_path
    