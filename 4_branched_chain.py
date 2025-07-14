from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from pydantic import BaseModel, Field
from typing import Annotated, Literal

load_dotenv()

class Review(BaseModel):
    sentiment : Annotated[Literal['positive', 'negative'], Field(description="The sentiment of the review")]

model = GoogleGenerativeAI(model='gemini-2.5-pro')

parser = PydanticOutputParser(pydantic_object= Review)
str_parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Analyze the below review provided by the user and based on the analysis provide the sentiment of that review.\n{review}\n\n{format_instruction}",
    input_variables= ['review'],
    partial_variables= {
        'format_instruction' : parser.get_format_instructions()
    }
)

prompt2 = PromptTemplate(
    template= "Generate a simple cutomer service response for the below positive review of the user in a standard and professional tone, and no need to provide any cutomization message like customer name, product name etc and dont provide any option just appropriatly respond to the review and make sure you include the key point that they are pointing in the review..\n{review}",
    input_variables= ['review']
)

prompt3 = PromptTemplate(
    template= "Generate a simple cutomer service response for the below negative review of the user in a standard and professional tone, and no need to provide any cutomization message like customer name, product name etc and dont provide any option just appropriatly respond to the review and make sure you include the key point that they are pointing in the review.\n{review}",
    input_variables= ['review']
)

sentiment_chain = prompt1 | model | parser

if_else_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | str_parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | str_parser),
    RunnableLambda(lambda x : "couldnt find the sentiment")
)

senti_brach_chain = RunnableParallel({
    'sentiment' : RunnableLambda(lambda x: x.sentiment),
    'response' : if_else_chain
})

chain = sentiment_chain | senti_brach_chain

review = "This product price and the quality of the product is like two side of the coin, absolutely terrible"

result = chain.invoke({'review' : review})
print(result)