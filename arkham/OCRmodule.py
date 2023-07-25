from unstructured.partition.pdf import partition_pdf
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

from typing import Union
import os


class ocr_parse_file:
    """
    This class contains the logic to parse pdf files and extract the text, then it is saved in disk.
    """

    def __init__(self, file_path: str) -> None:
        """Method to initiate the class.

        Args:
            file_path (str): Path to the pdf to parse.
        """
        self.file_path = file_path
        self.methods = ["hi_res", "fast", "secure"]
        self.paged_text = []

    def load_file(
        self,
        method: str = "hi_res",
        file_language: str = "spa",
        return_obj: bool = False,
    ) -> Union[None, list]:
        """Method that reads a pdf file using OCR and transforms it into a string.

        Args:
            method (str, optional): Method to use in OCR module, can be one of 'hi_res','fast','secure'. Defaults to 'hi_res'.
            file_language (str, optional): Language in wich the pdf is written. Defaults to 'spa'.
            return_obj (bool, optional): If parsed str list wants to be returned. Defaults to False.

        Returns:
            Union[None,list]: If retunr_obj returns the list of the parsed text from the file.
        """
        assert (
            method in self.methods
        ), f"The current reading methods available are {self.methods}"
        if method == "hi_res":
            try:
                # self.files will be the attribute where the parsed text will live.
                self.files = partition_pdf(
                    self.file_path, ocr_languages=file_language, strategy="hi_res"
                )
            except Exception as e:
                print(e)
                print(f'Method {method} failed, atempting method "fast"')
                method = "fast"

        if method == "fast":
            try:
                self.files = partition_pdf(
                    self.file_path, ocr_languages=file_language, strategy="fast"
                )
            except Exception as e:
                method = "secure"
                print(f'Method {method} failed, atempting method "secure"')

        if method == "secure":
            # Instanciate reader and files attribute
            pdf_reader = PyPDF2.PdfReader(open(self.file_path, "rb"))
            self.files = []
            # Iterate through each page
            for page_num in range(len(pdf_reader.pages)):
                # Convert page to image
                images = convert_from_path(
                    self.pdf_path, first_page=page_num, last_page=page_num + 1
                )
                # Convert image to text
                text = pytesseract.image_to_string(images[0], lang=file_language)
                self.files.append(text)
        # Add pages to each teact
        if method == "secure":
            for i, page in enumerate(self.files):
                self.paged_text.append(f"Pagina {i+1}\n {page}")
        elif method in ["hi_res", "fast"]:
            curr_page = 0
            curr_page_text = ""
            for line in self.files:
                # Separate between pages
                if curr_page != line.metadata.page_number:
                    self.paged_text.append(str(curr_page_text))
                    curr_page = line.metadata.page_number
                    curr_page_text = f"Page {curr_page}\n"
                curr_page_text += f"\n{line.text}"

        if return_obj:
            return self.paged_text

    def write_file(self, write_path: str = "parsed/") -> str:
        """Method to save the file in a given path.

        Args:
            write_path (str, optional): Path on which the file will be written. Defaults to 'parsed/'.

        Returns:
            str: Path to the saved file.
        """
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        write_path = write_path + self.file_path[:-4] + ".txt"
        with open(write_path, "w") as to_save:
            to_save.write("\n\n|".join(self.paged_text))
            to_save.close()
        return write_path
