import json
from typing import Dict, List
from huggingface_hub import InferenceClient

class CompletitionLLMAction:
    """
    Base class for LLM-based completition actions.
    """
    def __init__(self, client: InferenceClient) -> None:
        """
        Initialize the action with a HuggingFace InferenceClient.

        Parameters
        ----------
        client: huggingface_hub.InferenceClient
            The HuggingFace InferenceClient to use for completition
        """
        self.client = client

    def complete(self, prompt: str, context: List[str] = None) -> Dict[str, List[str]]:
        """
        Perform the completition given the input prompt.

        Parameters
        ----------
        prompt: str
            The input prompt to generate completitions from

        Raises
        ------
        NotImplementedError
            If this method is called directly from this base class
        """
        raise NotImplementedError("complete() method must be implemented in a subclass")

        
class DetectionDrugsLLMAction(CompletitionLLMAction):
    """
    Class to perform completition of drug names in Spanish using a transformer-based language model.

    Attributes
    ----------
    client: huggingface_hub.InferenceClient
        The HuggingFace InferenceClient to use for completition
    """
    def complete(self, prompt: str, context: List[str] = None) -> Dict[str, List[str]]:
        prompt_with_template = f"""
            <s>
            [INST] 
            **Input:** El texto que desea analizar (sobre medicamentos em Spain).
            [/INST]
            </s>

            <s>
            **Instruction:**

            [INST] 
            1. Identifique todas las menciones de productos químicos o sustancias dentro del texto.
            2. Inclua a apenas el nombre del producto.
            [/INST]
            </s>
            
            <s>
            **Result:**

            [INST] 
            El resultado es uno JSON de una lista de las sustancias activas identificadas que se encuentran en el texto.
            [/INST]
            </s>
            
            **Example:**

            **Input:** "Al paciente se le recetó ibuprofeno o presar para aliviar el dolor".
            **Output:** {{"sustancias_listadas": ["ibuprofeno", "presar"]}}

            **Input:** "Al paciente se le recetó dipirona para aliviar el dolor".
            **Output:** {{"sustancias_listadas": ["dipirona"]}}

            **Input:** {prompt}
            **Output:** {{"sustancias_listadas": ["
            """


        result =  self.client.text_generation(
            prompt_with_template,
            max_new_tokens=150,
            stop_sequences=['\n\n'])
        
        result = '{"substancias_listadas": ["' +  result.strip()


        try:
            return json.loads(result)
        except Exception as e:
            print(f'Exception {e}')
            return json.loads('{"substancia_listadas": []}')


class DrugQALLMAction(CompletitionLLMAction):
   def complete(self, prompt: str, context: List[str] = None) -> Dict[str, List[str]]:
        prompt_with_template = f"""
            [INST] 
            Responda la pregunta según el contexto proporcionado. Responder No sé si el contexto no es suficiente:
            [/INST] 

            <s>CONTEXTO: {context}</s>
            <s>PREGUNTA: {prompt}</s>
            <s>RESPUESTA: </>
            """

        result =  self.client.text_generation(
            prompt_with_template,
            max_new_tokens=150)
            # stop_sequences=['\n\n'])

        
        return result.strip()

if __name__ == "__main__":

    client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token="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")

    drug_detection = DetectionDrugsLLMAction(client)

    result = drug_detection.complete('substancia activa presar?')
    print(result['substancias_listadas'])



