from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.converters import PyPDFToDocument


class PdfHttpConverter:
    """
    Class that fetches PDF documents from HTTP links and converts them to text
    using PyPDFToDocument
    """

    def __init__(self) -> None:
        """
        Create a new PDFHttpConverter instance
        """
        self.fetcher = LinkContentFetcher()
        self.converter = PyPDFToDocument()

    def link_to_document(self, link: str) -> None:
        """
        Given a link to a PDF document, fetch it and convert it to text using PyPDFToDocument.
        The text is returned as a single string.

        Parameters
        ----------
        link: str
            The URL of the PDF document

        Returns
        -------
        str
            The text of the PDF document
        """
        streams = self.fetcher.run(urls=[link])["streams"]
        text_from_pdf = self.converter.run(streams)
        # The output of PyPDFToDocument is a dict containing a list of documents
        # We're only interested in the first document in the list
        return text_from_pdf["documents"][0]
