from typing import List, Literal, Union, Dict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel , Field

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
        self.llm =model
        self.routing_logic = routing_logic

        self.system_prompt = f"""You are an expert at routing user queries.
                    {self.routing_logic}
                    Here is the list of datasources to choose from:
                    {', '.join(self.datasources)}."""

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
