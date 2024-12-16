from typing import List
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain.utils.math import cosine_similarity

class DynamicRouteQuery(BaseModel):
    datasource: str = Field(
        description="The selected datasource that is most relevant for the given user query."
    )

class QueryRouter:
    """Custom router for routing user queries based on context."""

    def __init__(
        self,
        model,
        datasources: List[str],
        routing_logic: str = "Based on the context and keywords, choose the most relevant datasource."
    ):
        """
        Initialize the router with datasources and routing logic.

        Args:
            datasources (List[str]): List of datasources for routing.
            model : The LLM model to use.
            routing_logic (str): System prompt describing the routing logic.
        """
        self.datasources = datasources
        self.llm = model
        self.routing_logic = routing_logic

        self.system_prompt = (
            f"You are an expert at routing user queries.\n"
            f"{self.routing_logic}\n"
            f"Here is the list of datasources to choose from:\n"
            f"{', '.join(self.datasources)}."
        )

        self.structured_llm = self.llm.with_structured_output(DynamicRouteQuery)

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )

        self.router = self.prompt_template | self.structured_llm

    def route(self, question: str) -> str:
        """
        Route a question to the most relevant datasource.

        Args:
            question (str): The user question.

        Returns:
            str: The name of the selected datasource.
        """
        result = self.router.invoke({"question": question})
        return result.datasource

    def list_datasources(self) -> List[str]:
        """
        List all available datasources.

        Returns:
            List[str]: A list of datasource names.
        """
        return self.datasources



class EmbedRouter:
    """Route queries to templates/prompts using embedding-based similarity."""
    
    def __init__(self, embeddings_model, templates: List[str]):
        """
        Initialize with a list of templates and an embedding model.

        Args:
            templates (List[str]): List of templates for routing.
            embeddings_model : Model for embedding similarity.
        """
        self.templates = templates
        self.embeddings = embeddings_model
        self.template_embeddings = self.embeddings.embed_documents(self.templates)

    def get_most_similar_template(self, query: str) -> str:
        query_embedding = self.embeddings.embed_query(query)
        similarity = cosine_similarity([query_embedding], self.template_embeddings)[0]
        return self.templates[similarity.argmax()]

    def add_template(self, new_template: str):

        self.templates.append(new_template)
        self.template_embeddings = self.embeddings.embed_documents(self.templates)

    def route_query(self, query: str) -> PromptTemplate:

        most_similar_template = self.get_most_similar_template(query)
        return PromptTemplate.from_template(most_similar_template)